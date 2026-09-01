"""Analyze skipped samples across VisualCue result logs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.datasets.fsc147 import FSC147Adapter

ERROR_CATEGORIES = ("n_ctx", "max_tokens", "parse", "other")


def main() -> None:
    """CLI entry point for skip analysis."""

    parser = argparse.ArgumentParser(description="Analyze skipped VisualCue samples")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--fsc147-root", type=Path, default=Path("data/fsc147"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/skips.json"))
    parser.add_argument("--force", action="store_true", help="Overwrite --out if it already exists")
    args = parser.parse_args()

    out_path = _validate_out_path(args.out, force=args.force)
    fsc147_gt = _load_fsc147_gt(args.fsc147_root, args.split)
    runs = [_analyze_log(path, fsc147_gt) for path in sorted(args.results_root.rglob("*__samples.jsonl"))]
    result = {"runs": runs, "summary": _summary(runs)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(runs)
    print(f"wrote {out_path}")


def _validate_out_path(path: Path, force: bool) -> Path:
    resolved = path.resolve()
    allowed_root = (ROOT / "results" / "analysis").resolve()
    if allowed_root not in [resolved, *resolved.parents]:
        raise SystemExit(f"--out must be under {allowed_root}")
    if resolved.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing output: {resolved}; pass --force to overwrite")
    return resolved


def _load_fsc147_gt(root: Path, split: str) -> dict[str, int]:
    try:
        return {str(sample.sample_id): int(sample.gt_count) for sample in FSC147Adapter(root, split=split) if sample.gt_count is not None}
    except FileNotFoundError as exc:
        print(f"warning: FSC-147 GT unavailable: {exc}", file=sys.stderr)
        return {}


def _analyze_log(path: Path, fsc147_gt: dict[str, int]) -> dict[str, Any]:
    total = 0
    skipped = 0
    categories: Counter[str] = Counter()
    skip_rows: list[dict[str, Any]] = []
    skipped_density: list[float] = []
    scored_density: list[float] = []
    skipped_prompt_tokens: list[int] = []
    scored_prompt_tokens: list[int] = []
    dataset = _dataset_from_path(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            is_skipped = bool(row.get("skipped")) or row.get("count") is None
            density = _density_proxy(row, sample_id, dataset, fsc147_gt)
            prompt_tokens = _prompt_tokens(row)
            if is_skipped:
                skipped += 1
                category = _error_category(str(row.get("error", "")), str(row.get("skip_reason", "")))
                categories[category] += 1
                if density is not None:
                    skipped_density.append(density)
                if prompt_tokens is not None:
                    skipped_prompt_tokens.append(prompt_tokens)
                skip_rows.append(
                    {
                        "sample_id": sample_id,
                        "line": line_number,
                        "category": category,
                        "skip_reason": row.get("skip_reason"),
                        "error": row.get("error"),
                        "prompt_tokens": prompt_tokens,
                        "density_proxy": density,
                    }
                )
            else:
                if density is not None:
                    scored_density.append(density)
                if prompt_tokens is not None:
                    scored_prompt_tokens.append(prompt_tokens)

    return {
        "path": str(path),
        "run": _run_name(path),
        "system": _system_from_path(path),
        "dataset": dataset,
        "total": total,
        "scored": total - skipped,
        "skipped": skipped,
        "skip_rate": 0.0 if total == 0 else skipped / total,
        "categories": {category: categories.get(category, 0) for category in ERROR_CATEGORIES},
        "density_proxy": {
            "skipped_mean": _mean(skipped_density),
            "scored_mean": _mean(scored_density),
            "skipped_n": len(skipped_density),
            "scored_n": len(scored_density),
        },
        "prompt_tokens": {
            "skipped_mean": _mean(skipped_prompt_tokens),
            "skipped_max": max(skipped_prompt_tokens) if skipped_prompt_tokens else None,
            "scored_mean": _mean(scored_prompt_tokens),
            "scored_max": max(scored_prompt_tokens) if scored_prompt_tokens else None,
        },
        "skips": skip_rows,
    }


def _dataset_from_path(path: Path) -> str:
    stem = path.name.removesuffix("__samples.jsonl")
    if "__" not in stem:
        return "unknown"
    return stem.rsplit("__", 1)[1].removesuffix("_2").removesuffix("_3")


def _system_from_path(path: Path) -> str:
    stem = path.name.removesuffix("__samples.jsonl")
    if "__" not in stem:
        return stem
    return stem.rsplit("__", 1)[0]


def _run_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT / "results"))
    except ValueError:
        return str(path)


def _density_proxy(row: dict[str, Any], sample_id: str, dataset: str, fsc147_gt: dict[str, int]) -> float | None:
    if dataset == "fsc147" and sample_id in fsc147_gt:
        return float(fsc147_gt[sample_id])
    intermediate = row.get("intermediate")
    if isinstance(intermediate, dict):
        n_candidates = intermediate.get("n_candidates") or intermediate.get("n_preds")
        if isinstance(n_candidates, int | float):
            return float(n_candidates)
        steps = intermediate.get("steps")
        if isinstance(steps, list) and steps:
            last_step = steps[-1]
            if isinstance(last_step, dict) and isinstance(last_step.get("n_candidates"), int | float):
                return float(last_step["n_candidates"])
    predictions = row.get("predictions")
    if isinstance(predictions, list):
        return float(len(predictions))
    return None


def _prompt_tokens(row: dict[str, Any]) -> int | None:
    usage = row.get("vlm_usage")
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        return int(usage["prompt_tokens"])
    intermediate = row.get("intermediate")
    if not isinstance(intermediate, dict):
        return None
    total = intermediate.get("vlm_usage_total")
    if isinstance(total, dict) and isinstance(total.get("prompt_tokens"), int):
        return int(total["prompt_tokens"])
    entries = intermediate.get("vlm_usage")
    if isinstance(entries, list):
        values = [entry.get("prompt_tokens") for entry in entries if isinstance(entry, dict)]
        int_values = [int(value) for value in values if isinstance(value, int)]
        if int_values:
            return sum(int_values)
    return None


def _error_category(error: str, skip_reason: str) -> str:
    text = f"{skip_reason} {error}".lower()
    if "n_keep" in text or "n_ctx" in text or "context length" in text:
        return "n_ctx"
    if "max_tokens" in text or "token_limit" in text or "finish_reason" in text:
        return "max_tokens"
    if "failed to parse input" in text or "<|channel>thought" in text:
        return "parse"
    return "other"


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    for run in runs:
        categories.update(run["categories"])
    return {
        "n_runs": len(runs),
        "total_samples": sum(run["total"] for run in runs),
        "total_skipped": sum(run["skipped"] for run in runs),
        "categories": {category: categories.get(category, 0) for category in ERROR_CATEGORIES},
    }


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _print_table(runs: list[dict[str, Any]]) -> None:
    headers = ["Run", "Scored/Total", "Skip%", "n_ctx", "max_tok", "parse", "other", "skip_prompt_mean", "skip_density_mean"]
    rows = []
    for run in runs:
        rows.append(
            [
                run["run"],
                f"{run['scored']}/{run['total']}",
                f"{run['skip_rate'] * 100:.1f}%",
                str(run["categories"]["n_ctx"]),
                str(run["categories"]["max_tokens"]),
                str(run["categories"]["parse"]),
                str(run["categories"]["other"]),
                _format_optional(run["prompt_tokens"]["skipped_mean"]),
                _format_optional(run["density_proxy"]["skipped_mean"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def _format_optional(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.1f}"


def _format_row(values: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))


if __name__ == "__main__":
    main()
