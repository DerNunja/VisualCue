"""Download FSC-147 annotations and 384px images without density maps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

import requests

ANNOTATION_FILES = (
    "annotation_FSC147_384.json",
    "Train_Test_Val_FSC_147.json",
    "ImageClasses_FSC147.txt",
)
OFFICIAL_RAW_BASE = "https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data"
HF_DATASET = "isentropic/FSC147"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
HF_TREE_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/images_384_VarV2?recursive=1"
IMAGE_DIR_NAME = "images_384_VarV2"
REQUEST_TIMEOUT_SECONDS = 120
CHUNK_SIZE_BYTES = 1024 * 1024
MAX_RETRIES = 3

DownloadStatus = Literal["downloaded", "skipped", "failed"]


def main() -> None:
    """CLI entry point for preparing local FSC-147 data."""

    parser = argparse.ArgumentParser(description="Download FSC-147 annotations and 384px images")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    annotation_summary = ensure_annotations(args.root)
    image_summary = ensure_images(args.root)
    print(
        "summary: "
        f"downloaded={annotation_summary['downloaded'] + image_summary['downloaded']} "
        f"skipped={annotation_summary['skipped'] + image_summary['skipped']} "
        f"failed={annotation_summary['failed'] + image_summary['failed']}"
    )
    if annotation_summary["failed"] or image_summary["failed"]:
        raise SystemExit(1)


def ensure_annotations(root: Path) -> dict[DownloadStatus, int]:
    """Download the three small FSC-147 annotation files if missing."""

    summary: dict[DownloadStatus, int] = {"downloaded": 0, "skipped": 0, "failed": 0}
    for file_name in ANNOTATION_FILES:
        path = root / file_name
        if path.exists():
            summary["skipped"] += 1
            print(f"skipped annotation {file_name}")
            continue
        try:
            _download_from_urls(_annotation_urls(file_name), path)
        except requests.RequestException as exc:
            summary["failed"] += 1
            print(f"failed annotation {file_name}: {exc}")
        else:
            summary["downloaded"] += 1
            print(f"downloaded annotation {file_name}")
    return summary


def ensure_images(root: Path) -> dict[DownloadStatus, int]:
    """Download only images_384_VarV2 files from the Hugging Face mirror."""

    image_dir = root / IMAGE_DIR_NAME
    image_dir.mkdir(exist_ok=True)
    entries = _hf_image_entries()
    summary: dict[DownloadStatus, int] = {"downloaded": 0, "skipped": 0, "failed": 0}
    print(f"needed images: {len(entries)}")
    for index, entry in enumerate(entries, start=1):
        relative_path = Path(entry["path"]).relative_to(IMAGE_DIR_NAME)
        path = image_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            summary["skipped"] += 1
            continue
        url = f"{HF_RESOLVE_BASE}/{entry['path']}"
        try:
            _download_from_urls((url,), path)
        except requests.RequestException as exc:
            summary["failed"] += 1
            print(f"[{index}/{len(entries)}] failed {entry['path']}: {exc}")
        else:
            summary["downloaded"] += 1
            print(f"[{index}/{len(entries)}] downloaded {entry['path']}")
    print(
        "images: "
        f"downloaded={summary['downloaded']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    return summary


def _hf_image_entries() -> list[dict[str, Any]]:
    with requests.get(HF_TREE_URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        entries = response.json()
    if not isinstance(entries, list):
        raise RuntimeError("unexpected Hugging Face tree response for FSC-147 images")
    image_entries = [entry for entry in entries if entry.get("type") == "file" and _is_image_path(entry.get("path"))]
    if not image_entries:
        raise RuntimeError("Hugging Face mirror did not list any images_384_VarV2 image files")
    return sorted(image_entries, key=lambda entry: entry["path"])


def _is_image_path(path: object) -> bool:
    if not isinstance(path, str):
        return False
    suffix = Path(path).suffix.lower()
    return path.startswith(f"{IMAGE_DIR_NAME}/") and suffix in {".jpg", ".jpeg", ".png"}


def _annotation_urls(file_name: str) -> tuple[str, ...]:
    return (f"{OFFICIAL_RAW_BASE}/{file_name}", f"{HF_RESOLVE_BASE}/{file_name}")


def _download_from_urls(urls: tuple[str, ...], path: Path) -> None:
    part_path = path.with_name(f"{path.name}.part")
    last_error: requests.RequestException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        for url in urls:
            try:
                _download_url_once(url, part_path, path)
                return
            except requests.RequestException as exc:
                last_error = exc
                part_path.unlink(missing_ok=True)
        if attempt == MAX_RETRIES and last_error is not None:
            raise last_error


def _download_url_once(url: str, part_path: Path, path: Path) -> None:
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                if chunk:
                    handle.write(chunk)
    os.replace(part_path, path)


if __name__ == "__main__":
    main()
