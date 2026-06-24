"""Manual smoke check for SequentialPipeline with LM Studio and Falcon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.systems.sequential import SequentialPipeline, render_overlay


def main() -> None:
    """Run the real sequential pipeline and write a qualitative overlay."""

    parser = argparse.ArgumentParser(description="Smoke-test SequentialPipeline")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("query")
    parser.add_argument("--vlm-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--vlm-model", default="gemma")
    parser.add_argument("--vlm-api-key", default="lm-studio")
    parser.add_argument("--falcon-device", default="cuda:0")
    parser.add_argument(
        "--segmentation-prompt-style",
        choices=["short", "detailed", "complex"],
        default="short",
    )
    parser.add_argument("--disable-reasoning", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    try:
        pipeline = SequentialPipeline(
            vlm_base_url=args.vlm_base_url,
            vlm_model=args.vlm_model,
            vlm_api_key=args.vlm_api_key,
            falcon_device=args.falcon_device,
            enable_reasoning=not args.disable_reasoning,
            segmentation_prompt_style=args.segmentation_prompt_style,
        )
        output = pipeline.run(image, args.query)
    except Exception as exc:
        raise SystemExit(f"sequential smoke failed; check LM Studio, Falcon weights, and CUDA: {exc}") from exc

    out_path = ROOT / "results" / "smoke_sequential.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(image, output.instances).save(out_path)

    print(f"intent: {output.intermediate.get('intent')}")
    print(f"segmentation_prompt: {output.intermediate.get('segmentation_prompt')}")
    print(f"masks: {len(output.instances)}")
    print(f"count: {output.count}")
    print(f"answer: {output.answer}")
    print(f"overlay: {out_path}")


if __name__ == "__main__":
    main()
