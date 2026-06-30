from __future__ import annotations

import numpy as np
from PIL import Image

import pytest

from visualcue.harness.systems.sequential import (
    PLAN_SYSTEM_PROMPT_DETAILED,
    PLAN_SYSTEM_PROMPT_SHORT,
    SequentialPipeline,
)
from visualcue.harness.types import Instance


class FakeVLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    @property
    def plan_calls(self) -> int:
        return sum('"segmentation_prompt"' in system for system, _ in self.calls)

    @property
    def reason_calls(self) -> int:
        return len(self.calls) - self.plan_calls

    def complete(self, system: str, user_text: str, image: Image.Image | None = None) -> str:
        del image
        self.calls.append((system, user_text))
        return self.responses.pop(0)


class FakeSegmenter:
    def __init__(self, instances: list[Instance]) -> None:
        self.instances = instances
        self.prompts: list[str] = []
        self.to_cpu_calls = 0
        self.to_device_calls = 0

    def segment(self, image: Image.Image, prompt: str) -> list[Instance]:
        del image
        self.prompts.append(prompt)
        return self.instances

    def to_cpu(self) -> None:
        self.to_cpu_calls += 1

    def to_device(self) -> None:
        self.to_device_calls += 1


def test_sequential_count_intent_keeps_all_masks_and_vlm_count() -> None:
    vlm = FakeVLM([
        '{"intent":"count","segmentation_prompt":"bottle"}',
        '{"count":2,"answer":"There are two bottles."}',
    ])
    segmenter = FakeSegmenter(_instances(3))
    output = SequentialPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "how many bottles?")

    assert output.count == 2
    assert len(output.instances) == 3
    assert segmenter.prompts == ["bottle"]
    assert output.intermediate["segmentation_prompt"] == "bottle"
    assert vlm.plan_calls == 1
    assert vlm.reason_calls == 1


def test_sequential_locate_intent_selects_candidate() -> None:
    candidates = _instances(3)
    vlm = FakeVLM([
        '{"intent":"locate","segmentation_prompt":"red cup"}',
        '{"selected":[1],"answer":"The second candidate."}',
    ])
    segmenter = FakeSegmenter(candidates)
    output = SequentialPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find the red cup")

    assert output.instances == [candidates[1]]
    assert output.count == 1
    assert output.answer == "The second candidate."


def test_sequential_without_reasoning_skips_reason_call() -> None:
    vlm = FakeVLM(['{"intent":"count","segmentation_prompt":"bottle"}'])
    segmenter = FakeSegmenter(_instances(2))
    output = SequentialPipeline(vlm=vlm, segmenter=segmenter, enable_reasoning=False).run(_image(), "count bottles")

    assert output.count == 2
    assert len(output.instances) == 2
    assert output.answer == "2"
    assert vlm.plan_calls == 1
    assert vlm.reason_calls == 0


def test_sequential_bad_plan_json_falls_back_to_query() -> None:
    vlm = FakeVLM(["not json", '{"selected":[0],"answer":"fallback selected"}'])
    segmenter = FakeSegmenter(_instances(1))
    output = SequentialPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "raw query")

    assert segmenter.prompts == ["raw query"]
    assert output.intermediate["plan_parse_fallback"] is True
    assert output.intermediate["segmentation_prompt"] == "raw query"


def test_sequential_gold_targets_skip_plan() -> None:
    vlm = FakeVLM(['{"selected":[0],"answer":"gold target"}'])
    segmenter = FakeSegmenter(_instances(1))
    output = SequentialPipeline(vlm=vlm, segmenter=segmenter).run(_image(), "find it", gold_targets=["duck"])

    assert vlm.plan_calls == 0
    assert vlm.reason_calls == 1
    assert segmenter.prompts == ["duck"]
    assert output.count == 1


def test_sequential_config_includes_prompts_and_reasoning_flag() -> None:
    pipeline = SequentialPipeline(vlm=FakeVLM([]), segmenter=FakeSegmenter([]), enable_reasoning=False)
    config = pipeline.config()

    assert config["segmentation_prompt_style"] == "short"
    assert "plan_system_prompt" in config
    assert "count_reason_system_prompt" in config
    assert "locate_reason_system_prompt" in config
    assert config["enable_reasoning"] is False


def test_sequential_custom_plan_prompt_is_used_and_logged() -> None:
    vlm = FakeVLM(['{"intent":"count","segmentation_prompt":"bottle"}'])
    pipeline = SequentialPipeline(
        vlm=vlm,
        segmenter=FakeSegmenter(_instances(1)),
        enable_reasoning=False,
        plan_system_prompt="CUSTOM",
    )

    output = pipeline.run(_image(), "count bottles")

    assert pipeline.config()["plan_system_prompt"] == "CUSTOM"
    assert vlm.calls[0][0] == "CUSTOM"
    assert output.intermediate["segmentation_prompt"] == "bottle"


def test_sequential_prompt_style_selects_detailed_plan_prompt() -> None:
    vlm = FakeVLM(['{"intent":"locate","segmentation_prompt":"small red cup on the left"}'])
    pipeline = SequentialPipeline(
        vlm=vlm,
        segmenter=FakeSegmenter(_instances(1)),
        enable_reasoning=False,
        segmentation_prompt_style="detailed",
    )

    output = pipeline.run(_image(), "find the small red cup on the left")

    assert pipeline.config()["segmentation_prompt_style"] == "detailed"
    assert pipeline.config()["plan_system_prompt"] == PLAN_SYSTEM_PROMPT_DETAILED
    assert vlm.calls[0][0] == PLAN_SYSTEM_PROMPT_DETAILED
    assert output.intermediate["segmentation_prompt_style"] == "detailed"
    assert output.intermediate["segmentation_prompt"] == "small red cup on the left"


def test_sequential_prompt_style_defaults_to_short_plan_prompt() -> None:
    pipeline = SequentialPipeline(vlm=FakeVLM([]), segmenter=FakeSegmenter([]))

    assert pipeline.config()["segmentation_prompt_style"] == "short"
    assert pipeline.config()["plan_system_prompt"] == PLAN_SYSTEM_PROMPT_SHORT


def test_sequential_complex_prompt_style_aliases_detailed() -> None:
    pipeline = SequentialPipeline(vlm=FakeVLM([]), segmenter=FakeSegmenter([]), segmentation_prompt_style="complex")

    assert pipeline.config()["segmentation_prompt_style"] == "detailed"
    assert pipeline.config()["plan_system_prompt"] == PLAN_SYSTEM_PROMPT_DETAILED


def test_sequential_invalid_prompt_style_fails_fast() -> None:
    with pytest.raises(ValueError, match="segmentation_prompt_style"):
        SequentialPipeline(vlm=FakeVLM([]), segmenter=FakeSegmenter([]), segmentation_prompt_style="verbose")


def _image() -> Image.Image:
    return Image.new("RGB", (10, 10), "white")


def _instances(count: int) -> list[Instance]:
    instances: list[Instance] = []
    for index in range(count):
        mask = np.zeros((10, 10), dtype=bool)
        mask[index : index + 2, index : index + 2] = True
        instances.append(Instance(mask=mask, bbox=(float(index), float(index), 2.0, 2.0), label="fake", score=None))
    return instances
