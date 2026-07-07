"""Recompute FSC-147 counting metrics on the intersection of scored samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.datasets.fsc147 import FSC147Adapter
from visualcue.harness.metrics import counting_errors


def main() -> None:
    """CLI entry point for intersection-based FSC-147 count evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate FSC-147 counts on the common scored sample set")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", action="append", required=True, help="System sample log as name=path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out_path = _validated_out_path(args.out)
    gt = _load_fsc147_gt(args.dataset_root, args.split)
    systems = _load_system_samples(args.samples)
    common = _common_scored_ids(systems, gt)

    if not common:
        print("No common scored FSC-147 samples found across all systems and ground truth.", file=sys.stderr)

    result = {
        "dataset": "fsc147",
        "intersection_size": len(common),
        "systems": {
            name: _system_result(samples, gt, common)
            for name, samples in systems.items()
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(result)
    print(f"wrote {out_path}")


def _validated_out_path(path: Path) -> Path:
    resolved = path.resolve()
    allowed_root = (ROOT / "results" / "intersection").resolve()
    if allowed_root not in [resolved, *resolved.parents]:
        raise SystemExit(f"--out must be under {allowed_root}")
    if resolved.exists():
        raise SystemExit(f"refusing to overwrite existing output: {resolved}")
    return resolved


def _load_fsc147_gt(root: Path, split: str) -> dict[str, int]:
    gt: dict[str, int] = {}
    for sample in FSC147Adapter(root, split=split):
        if sample.gt_count is not None:
            gt[str(sample.sample_id)] = int(sample.gt_count)
    return gt


def _load_system_samples(entries: list[str]) -> dict[str, dict[str, Any]]:
    systems: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name, path = _parse_sample_arg(entry)
        if name in systems:
            raise SystemExit(f"duplicate system name in --samples: {name}")
        systems[name] = _read_samples(path)
    return systems


def _parse_sample_arg(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise SystemExit(f"--samples must be name=path, got: {entry}")
    name, raw_path = entry.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"empty system name in --samples: {entry}")
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"samples file does not exist: {path}")
    return name, path


def _read_samples(path: Path) -> dict[str, Any]:
    scored: dict[str, int] = {}
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            count = row.get("count")
            if count is not None:
                try:
                    scored[sample_id] = int(count)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"invalid count in {path}:{line_number}: {count!r}") from exc
    return {"path": str(path), "scored": scored, "total": total}


def _common_scored_ids(systems: dict[str, dict[str, Any]], gt: dict[str, int]) -> list[str]:
    if not systems:
        return []
    sets = [set(system["scored"]) for system in systems.values()]
    common = set.intersection(*sets) & set(gt)
    return sorted(common)


def _system_result(samples: dict[str, Any], gt: dict[str, int], common: list[str]) -> dict[str, Any]:
    scored: dict[str, int] = samples["scored"]
    own_ids = sorted(set(scored) & set(gt))
    total = int(samples["total"])
    scored_count = len(scored)
    coverage = {
        "scored": scored_count,
        "total": total,
        "skip_rate": 0.0 if total == 0 else 1.0 - scored_count / total,
    }
    return {
        "coverage": coverage,
        "intersection": _count_metrics(scored, gt, common),
        "own_scored": {**_count_metrics(scored, gt, own_ids), "n": len(own_ids)},
    }


def _count_metrics(scored: dict[str, int], gt: dict[str, int], sample_ids: list[str]) -> dict[str, float]:
    pred = [scored[sample_id] for sample_id in sample_ids]
    gt_counts = [gt[sample_id] for sample_id in sample_ids]
    return counting_errors(pred, gt_counts)


def _print_table(result: dict[str, Any]) -> None:
    rows = []
    for name, system in result["systems"].items():
        coverage = system["coverage"]
        rows.append(
            [
                name,
                f"{coverage['scored']}/{coverage['total']}",
                f"{coverage['skip_rate'] * 100:.1f}%",
                f"{system['intersection']['mae']:.3f}",
                f"{system['intersection']['rmse']:.3f}",
                f"{system['own_scored']['mae']:.3f}",
            ]
        )

    headers = ["System", "Coverage", "Skip%", "MAE_common", "RMSE_common", "MAE_own"]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print(f"Intersection size: {result['intersection_size']}")
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def _format_row(values: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))


if __name__ == "__main__":
    main()
