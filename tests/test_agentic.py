from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visualcue.harness.systems.agentic import EVALUATE_SYSTEM_PROMPT, AgenticPipeline
from visualcue.harness.types import Instance


class FakeVLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []
        self.stages: list[str] = []

    @property
    def plan_calls(self) -> int:
        return self.stages.count("plan")

    @property
    def evaluate_calls(self) -> int:
        return self.stages.count("evaluate")

    @property
    def reason_calls(self) -> int:
        return self.stages.count("reason")

    def complete(self, system: str, user_text: str, image: Image.Image | None = None) -> str:
        del image
        self.calls.append((system, user_text))
        self.stages.append(_stage(system))
        return self.responses.pop(0)


class FakeSegmenter:
    def __init__(self, responses: list[list[Instance]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.to_cpu_calls = 0
        self.to_device_calls = 0

    def segment(self, image: Image.Image, prompt: str) -> list[Instance]:
        del image
        self.prompts.append(prompt)
        return self.responses.pop(0)

    def to_cpu(self) -> None:
        self.to_cpu_calls += 1

    def to_device(self) -> None:
        self.to_device_calls += 1


def test_agentic_refines_then_finalizes_and_reasons_on_latest_segment() -> None:
    first_segment = _instances(3)
    final_segment = _instances(1)
    vlm = FakeVLM([
        '{"intent":"locate","segmentation_prompt":"cow"}',
        '{"action":"refine","segmentation_prompt":"the front cow","reason":"too broad"}',
        '{"action":"finalize","reason":"good"}',
        '{"selected":[0],"answer":"front cow"}',
    ])
    segmenter = FakeSegmenter([first_segment, final_segment])

    output = AgenticPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find the front cow")

    assert segmenter.prompts == ["cow", "the front cow"]
    assert output.instances == final_segment
    assert output.count == 1
    assert output.answer == "front cow"
    assert output.intermediate["n_steps"] == 2
    assert output.intermediate["stop_reason"] == "finalize"
    assert output.intermediate["segmentation_prompt"] == "the front cow"
    assert output.intermediate["n_candidates"] == 1
    assert output.intermediate["steps"] == [
        {
            "step": 0,
            "segmentation_prompt": "cow",
            "n_candidates": 3,
            "action": "refine",
            "evaluate_raw": '{"action":"refine","segmentation_prompt":"the front cow","reason":"too broad"}',
        },
        {
            "step": 1,
            "segmentation_prompt": "the front cow",
            "n_candidates": 1,
            "action": "finalize",
            "evaluate_raw": '{"action":"finalize","reason":"good"}',
        },
    ]
    assert vlm.stages == ["plan", "evaluate", "evaluate", "reason"]
    assert "Candidates:" in vlm.calls[-1][1]
    assert "bbox=(0.0, 0.0, 2.0, 2.0)" in vlm.calls[-1][1]


def test_agentic_stops_at_max_steps_and_still_outputs() -> None:
    vlm = FakeVLM([
        '{"intent":"count","segmentation_prompt":"apple"}',
        '{"action":"refine","segmentation_prompt":"red apples","reason":"narrow"}',
        '{"count":2,"answer":"two apples"}',
    ])
    segmenter = FakeSegmenter([_instances(1), _instances(2)])

    output = AgenticPipeline(vlm=vlm, segmenter=segmenter, max_steps=2).run(_image(), "count red apples")

    assert segmenter.prompts == ["apple", "red apples"]
    assert output.count == 2
    assert output.intermediate["n_steps"] == 2
    assert output.intermediate["stop_reason"] == "max_steps"
    assert output.intermediate["segmentation_prompt"] == "red apples"
    assert output.intermediate["steps"][-1]["action"] == "max_steps"
    assert output.intermediate["steps"][-1]["evaluate_raw"] is None


def test_agentic_stops_on_unchanged_refine_prompt() -> None:
    vlm = FakeVLM([
        '{"intent":"locate","segmentation_prompt":"cow"}',
        '{"action":"refine","segmentation_prompt":" cow ","reason":"same"}',
        '{"selected":[0],"answer":"cow"}',
    ])
    segmenter = FakeSegmenter([_instances(1)])

    output = AgenticPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find cow")

    assert segmenter.prompts == ["cow"]
    assert output.intermediate["n_steps"] == 1
    assert output.intermediate["stop_reason"] == "no_change"
    assert output.intermediate["steps"][0]["action"] == "refine"


def test_agentic_bad_evaluate_json_finalizes_without_crashing() -> None:
    vlm = FakeVLM([
        '{"intent":"locate","segmentation_prompt":"cow"}',
        'not json',
        '{"selected":[0],"answer":"cow"}',
    ])
    segmenter = FakeSegmenter([_instances(1)])

    output = AgenticPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find cow")
    assert output.intermediate["stop_reason"] == "finalize"
    assert output.intermediate["evaluate_parse_fallback"] is True
    assert "evaluate_parse_error" in output.intermediate
    assert output.count == 1


def test_agentic_gold_targets_skip_plan_and_evaluate() -> None:
    vlm = FakeVLM(['{"selected":[0],"answer":"gold duck"}'])
    segmenter = FakeSegmenter([_instances(1)])

    output = AgenticPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find it", gold_targets=["duck"])
    assert vlm.plan_calls == 0
    assert vlm.evaluate_calls == 0
    assert vlm.reason_calls == 1
    assert segmenter.prompts == ["duck"]
    assert output.intermediate["plan_skipped_gold_targets"] is True
    assert output.intermediate["n_steps"] == 1
    assert output.intermediate["stop_reason"] == "finalize"


def test_agentic_config_includes_max_steps_and_evaluate_prompt() -> None:
    pipeline = AgenticPipeline(
        vlm=FakeVLM([]),
        segmenter=FakeSegmenter([]),
        max_steps=4,
        evaluate_system_prompt="EVAL",
    )
    config = pipeline.config()
    assert config["max_steps"] == 4
    assert config["evaluate_system_prompt"] == "EVAL"
    assert EVALUATE_SYSTEM_PROMPT.startswith("You are refining")


def test_agentic_rejects_non_positive_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        AgenticPipeline(vlm=FakeVLM([]), segmenter=FakeSegmenter([]), max_steps=0)


def _stage(system: str) -> str:
    if "refining an open-vocabulary segmentation" in system:
        return "evaluate"
    if '"intent": "count"|"locate"' in system:
        return "plan"
    return "reason"


def _image() -> Image.Image:
    return Image.new("RGB", (10, 10), "white")


def _instances(count: int) -> list[Instance]:
    instances: list[Instance] = []
    for index in range(count):
        mask = np.zeros((10, 10), dtype=bool)
        mask[index : index + 2, index : index + 2] = True
        instances.append(Instance(mask=mask, bbox=(float(index), float(index), 2.0, 2.0), label="fake", score=None))
    return instances
