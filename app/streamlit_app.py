"""Interactive Streamlit demo for VisualCue pipelines."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.systems._falcon import DEFAULT_DEVICE, DEFAULT_MODEL_ID, FalconSegmenter
from visualcue.harness.systems._pipeline import (
    PLAN_SYSTEM_PROMPT_DETAILED,
    PLAN_SYSTEM_PROMPT_SHORT,
    render_overlay,
)
from visualcue.harness.systems._vlm import VLMClient
from visualcue.harness.systems.agentic import AgenticPipeline
from visualcue.harness.systems.sequential import SequentialPipeline

SYSTEM_LABELS = {
    "sequential_pipeline": "Sequential Pipeline",
    "agentic_pipeline": "Agentic Pipeline",
}
PROMPT_STYLES = {
    "short": PLAN_SYSTEM_PROMPT_SHORT,
    "detailed": PLAN_SYSTEM_PROMPT_DETAILED,
}


@st.cache_resource(show_spinner="Loading Falcon segmenter...")
def build_segmenter(falcon_model_id: str, falcon_device: str) -> FalconSegmenter:
    """Load Falcon once and keep it resident across UI interactions."""

    return FalconSegmenter(model_id=falcon_model_id, device=falcon_device)


def main() -> None:
    """Render the VisualCue demo app."""

    st.set_page_config(page_title="VisualCue Demo", page_icon="VisualCue", layout="wide")
    st.title("VisualCue")
    st.caption("Ask visual questions and inspect which image regions the pipeline used.")

    settings = _sidebar_settings()
    image = _uploaded_image()
    query = st.text_input("Question", placeholder="How many watches are there? / Which umbrella is closest?")
    ask = st.button("Ask", type="primary")

    if image is not None:
        st.subheader("Input")
        st.image(image, caption="Original image", use_container_width=True)

    if not ask:
        st.info("Upload an image, enter a question, then click Ask.")
        return
    if image is None:
        st.warning("Please upload a JPG or PNG image first.")
        return
    if not query.strip():
        st.warning("Please enter a question first.")
        return

    try:
        segmenter = build_segmenter(DEFAULT_MODEL_ID, DEFAULT_DEVICE)
        pipeline = _make_pipeline(settings=settings, segmenter=segmenter)
        with st.spinner(_spinner_text(settings["system"])):
            output = pipeline.run(image, query.strip())
    except Exception as exc:
        st.error(f"VisualCue failed: {exc}")
        st.info("Is LM Studio running and the selected model loaded?")
        return

    overlay = render_overlay(
        image,
        output.instances,
        fill_masks=bool(settings["overlay_fill_masks"]),
        draw_mask_outlines=bool(settings["overlay_draw_mask_outlines"]),
        draw_boxes=bool(settings["overlay_draw_boxes"]),
        line_width=int(settings["overlay_line_width"]),
    )
    _show_result(image, overlay, output.answer, output.count)
    _show_trace(settings["system"], output.intermediate)


def _sidebar_settings() -> dict[str, Any]:
    st.sidebar.header("Settings")
    system = st.sidebar.radio(
        "System",
        options=list(SYSTEM_LABELS),
        format_func=lambda value: SYSTEM_LABELS[value],
    )
    prompt_style = st.sidebar.radio("Planning prompt style", options=list(PROMPT_STYLES), horizontal=True)
    enable_reasoning = st.sidebar.toggle("Enable reasoning", value=True)

    max_steps = 8
    include_prompt_history = False
    if system == "agentic_pipeline":
        max_steps = st.sidebar.slider("Max refine steps", min_value=1, max_value=12, value=8)
        include_prompt_history = st.sidebar.toggle("Show previous prompts to loop", value=False)
        st.sidebar.caption("Agentic is slower because it uses additional VLM calls per refine step.")

    with st.sidebar.expander("Connection settings"):
        vlm_base_url = st.text_input("VLM base URL", value="http://localhost:1234/v1")
        vlm_model = st.text_input("VLM model", value="google/gemma-4-26b-a4b")
        vlm_api_key = st.text_input("VLM API key", value="lm-studio", type="password")

    with st.sidebar.expander("Overlay settings"):
        overlay_fill_masks = st.toggle("Fill masks", value=False)
        overlay_draw_mask_outlines = st.toggle("Draw mask outlines", value=True)
        overlay_draw_boxes = st.toggle("Draw bounding boxes", value=True)
        overlay_line_width = st.slider("Line width", min_value=1, max_value=8, value=2)

    return {
        "system": system,
        "plan_system_prompt": PROMPT_STYLES[prompt_style],
        "prompt_style": prompt_style,
        "enable_reasoning": enable_reasoning,
        "max_steps": max_steps,
        "include_prompt_history": include_prompt_history,
        "vlm_base_url": vlm_base_url,
        "vlm_model": vlm_model,
        "vlm_api_key": vlm_api_key,
        "overlay_fill_masks": overlay_fill_masks,
        "overlay_draw_mask_outlines": overlay_draw_mask_outlines,
        "overlay_draw_boxes": overlay_draw_boxes,
        "overlay_line_width": overlay_line_width,
    }


def _uploaded_image() -> Image.Image | None:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return None
    data = uploaded.read()
    return Image.open(BytesIO(data)).convert("RGB")


def _make_pipeline(settings: dict[str, Any], segmenter: FalconSegmenter) -> SequentialPipeline | AgenticPipeline:
    vlm = VLMClient(
        base_url=str(settings["vlm_base_url"]),
        model=str(settings["vlm_model"]),
        api_key=str(settings["vlm_api_key"]),
    )
    common_kwargs = {
        "vlm_model": str(settings["vlm_model"]),
        "enable_reasoning": bool(settings["enable_reasoning"]),
        "plan_system_prompt": str(settings["plan_system_prompt"]),
        "segmentation_prompt_style": str(settings["prompt_style"]),
        "vlm": vlm,
        "segmenter": segmenter,
    }
    if settings["system"] == "agentic_pipeline":
        return AgenticPipeline(
            max_steps=int(settings["max_steps"]),
            include_prompt_history=bool(settings["include_prompt_history"]),
            **common_kwargs,
        )
    return SequentialPipeline(**common_kwargs)


def _spinner_text(system: str) -> str:
    if system == "agentic_pipeline":
        return "Analyzing with the agentic loop. This may take several VLM roundtrips..."
    return "Analyzing..."


def _show_result(image: Image.Image, overlay: Image.Image, answer: str, count: int | None) -> None:
    st.subheader("Result")
    original_col, overlay_col = st.columns(2)
    with original_col:
        st.image(image, caption="Original", use_container_width=True)
    with overlay_col:
        st.image(overlay, caption="VisualCue overlay", use_container_width=True)

    metric_col, answer_col = st.columns([1, 3])
    with metric_col:
        if count is not None:
            st.metric("Count", count)
        else:
            st.metric("Count", "n/a")
    with answer_col:
        st.markdown("**Answer**")
        st.success(answer or "No answer returned.")


def _show_trace(system: str, intermediate: dict[str, Any]) -> None:
    st.subheader("Prompt Trace")
    if system == "agentic_pipeline":
        _show_agentic_trace(intermediate)
    else:
        st.write(f"**Intent:** `{intermediate.get('intent', 'unknown')}`")
        st.write(f"**Segmentation prompt:** `{intermediate.get('segmentation_prompt', '')}`")
        st.write(f"**Candidates:** `{intermediate.get('n_candidates', 0)}`")

    with st.expander("Raw details"):
        st.json(intermediate)


def _show_agentic_trace(intermediate: dict[str, Any]) -> None:
    st.write(f"**Intent:** `{intermediate.get('intent', 'unknown')}`")
    st.write(f"**Steps:** `{intermediate.get('n_steps', 0)}`")
    st.write(f"**Stop reason:** `{intermediate.get('stop_reason', 'unknown')}`")

    steps = intermediate.get("steps", [])
    if not isinstance(steps, list) or not steps:
        st.info("No loop trace available.")
        return

    for step in steps:
        if not isinstance(step, dict):
            continue
        index = step.get("step", "?")
        prompt = step.get("segmentation_prompt", "")
        n_candidates = step.get("n_candidates", 0)
        action = step.get("action", "unknown")
        reason = _step_reason(step)
        title = f"Prompt {index}: {prompt}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(f"Candidates: `{n_candidates}`")
            st.write(f"Action: `{action}`")
            if reason:
                st.caption(reason)


def _step_reason(step: dict[str, Any]) -> str:
    direct_reason = step.get("reason")
    if direct_reason:
        return str(direct_reason)
    raw = step.get("evaluate_raw")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parsed = _parse_json_from_text(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    reason = parsed.get("reason")
    return "" if reason is None else str(reason)


def _parse_json_from_text(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("expected JSON object")
    return parsed


if __name__ == "__main__":
    main()
