"""Download FSC-147 annotations and 384px images without density maps."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
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
GOOGLE_DRIVE_FILE_ID = "1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S"
GOOGLE_DRIVE_ARCHIVE = "fsc147_images_384.zip"
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
    """Download 384px images from HF first, then official Drive archive if needed."""

    image_dir = root / IMAGE_DIR_NAME
    image_dir.mkdir(exist_ok=True)
    required = _required_image_names(root)
    summary: dict[DownloadStatus, int] = {"downloaded": 0, "skipped": 0, "failed": 0}
    print(f"needed split images: {len(required)}")
    try:
        entries = _hf_image_entries()
    except (requests.RequestException, RuntimeError) as exc:
        entries = []
        print(f"Hugging Face image mirror unavailable, using Google Drive fallback: {exc}")
    else:
        print(f"Hugging Face mirror images listed: {len(entries)}")
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
                print(f"[{index}/{len(entries)}] failed {entry['path']}: {exc}")
            else:
                summary["downloaded"] += 1
                print(f"[{index}/{len(entries)}] downloaded {entry['path']}")

    missing = _missing_images(image_dir, required)
    if missing:
        print(f"missing after Hugging Face mirror: {len(missing)}")
        try:
            summary["downloaded"] += _download_and_extract_drive_archive(root, missing)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            print(f"failed Google Drive image fallback: {exc}")

    remaining = _missing_images(image_dir, required)
    if remaining:
        summary["failed"] += len(remaining)
        print(f"still missing images after all sources: {len(remaining)}")
        print(f"first missing: {remaining[:20]}")
    print(
        "images: "
        f"downloaded={summary['downloaded']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    return summary


def _required_image_names(root: Path) -> list[str]:
    split_path = root / "Train_Test_Val_FSC_147.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    names = {file_name for split_names in splits.values() for file_name in split_names}
    return sorted(names)


def _missing_images(image_dir: Path, file_names: list[str]) -> list[str]:
    return [file_name for file_name in file_names if not (image_dir / file_name).exists()]


def _download_and_extract_drive_archive(root: Path, missing: list[str]) -> int:
    archive_path = root / GOOGLE_DRIVE_ARCHIVE
    if not archive_path.exists():
        _download_drive_archive(archive_path)
    extracted = _extract_images_from_archive(archive_path, root / IMAGE_DIR_NAME, set(missing))
    if extracted == 0:
        raise RuntimeError(
            f"archive {archive_path} did not contain any of the {len(missing)} missing images under {IMAGE_DIR_NAME}"
        )
    return extracted


def _download_drive_archive(path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive fallback requires gdown; install it with `uv add gdown`.") from exc

    part_path = path.with_name(f"{path.name}.part")
    result = gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=str(part_path), quiet=False)
    if result is None:
        part_path.unlink(missing_ok=True)
        raise RuntimeError("gdown did not return a downloaded archive path")
    os.replace(part_path, path)


def _extract_images_from_archive(archive_path: Path, image_dir: Path, wanted: set[str]) -> int:
    extracted = 0
    image_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative_path = _zip_image_relative_path(member.filename)
            if relative_path is None or relative_path.name not in wanted:
                continue
            target_path = image_dir / relative_path.name
            if target_path.exists():
                continue
            part_path = target_path.with_name(f"{target_path.name}.part")
            with archive.open(member) as source, part_path.open("wb") as destination:
                while chunk := source.read(CHUNK_SIZE_BYTES):
                    destination.write(chunk)
            os.replace(part_path, target_path)
            extracted += 1
    return extracted


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


def _zip_image_relative_path(member_name: str) -> Path | None:
    parts = Path(member_name).parts
    if IMAGE_DIR_NAME not in parts:
        return None
    image_dir_index = parts.index(IMAGE_DIR_NAME)
    relative = Path(*parts[image_dir_index + 1 :])
    if not relative.name or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    return relative


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
