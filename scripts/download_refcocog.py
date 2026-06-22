"""Download RefCOCOg annotations and only the images needed for one split."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualcue.harness.datasets.refcocog import required_image_files

ANNOTATION_URLS = (
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
)
COCO_TRAIN2014_URL = "https://images.cocodataset.org/train2014"
REQUEST_TIMEOUT_SECONDS = 60
CHUNK_SIZE_BYTES = 1024 * 1024
MAX_RETRIES = 3

DownloadStatus = Literal["downloaded", "skipped", "failed"]


def main() -> None:
    """CLI entry point for selective RefCOCOg split preparation."""

    parser = argparse.ArgumentParser(description="Download RefCOCOg annotations and split images")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--split-by", default="umd")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    args.root.mkdir(parents=True, exist_ok=True)
    ensure_annotations(args.root, args.split_by)
    file_names = required_image_files(args.root, split=args.split, split_by=args.split_by)
    images_dir = args.root / "images"
    images_dir.mkdir(exist_ok=True)

    existing = [file_name for file_name in file_names if (images_dir / file_name).exists()]
    missing = [file_name for file_name in file_names if not (images_dir / file_name).exists()]
    print(f"needed images: {len(file_names)}")
    print(f"already present: {len(existing)}")
    print(f"to download: {len(missing)}")

    summary = download_missing_images(images_dir, missing, args.workers)
    print(
        "summary: "
        f"downloaded={summary['downloaded']} "
        f"skipped={summary['skipped'] + len(existing)} "
        f"failed={summary['failed']}"
    )
    if summary["failed"]:
        raise SystemExit(1)


def ensure_annotations(root: Path, split_by: str) -> None:
    """Ensure refs(split_by).p and instances.json exist under root."""

    refs_path = root / f"refs({split_by}).p"
    instances_path = root / "instances.json"
    if refs_path.exists() and instances_path.exists():
        return

    archive_path = root / "refcocog.zip"
    if not archive_path.exists():
        _download_first_available(ANNOTATION_URLS, archive_path)
    _extract_annotations(archive_path, root, split_by)

    missing = [str(path) for path in (refs_path, instances_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"annotation archive did not provide: {', '.join(missing)}")


def download_missing_images(images_dir: Path, file_names: list[str], workers: int) -> dict[DownloadStatus, int]:
    """Download missing COCO train2014 images in parallel."""

    summary: dict[DownloadStatus, int] = {"downloaded": 0, "skipped": 0, "failed": 0}
    if not file_names:
        return summary

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_image, images_dir, file_name): file_name for file_name in file_names}
        for index, future in enumerate(as_completed(futures), start=1):
            file_name = futures[future]
            try:
                status = future.result()
            except requests.RequestException as exc:
                status = "failed"
                print(f"[{index}/{len(file_names)}] failed {file_name}: {exc}")
            else:
                print(f"[{index}/{len(file_names)}] {status} {file_name}")
            summary[status] += 1
    return summary


def _download_first_available(urls: tuple[str, ...], path: Path) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            print(f"downloading annotations: {url}")
            _download_url(url, path)
            return
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("could not download RefCOCOg annotations\n" + "\n".join(errors))


def _extract_annotations(archive_path: Path, root: Path, split_by: str) -> None:
    targets = {f"refs({split_by}).p": root / f"refs({split_by}).p", "instances.json": root / "instances.json"}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for target_name, target_path in targets.items():
            if target_path.exists():
                continue
            member = _find_archive_member(names, target_name)
            if member is None:
                continue
            part_path = target_path.with_name(f"{target_path.name}.part")
            with archive.open(member) as source, part_path.open("wb") as destination:
                while chunk := source.read(CHUNK_SIZE_BYTES):
                    destination.write(chunk)
            os.replace(part_path, target_path)


def _find_archive_member(names: list[str], target_name: str) -> str | None:
    matches = [name for name in names if Path(name).name == target_name]
    if not matches:
        return None
    refcocog_matches = [name for name in matches if "refcocog" in name.lower()]
    return sorted(refcocog_matches or matches, key=len)[0]


def _download_image(images_dir: Path, file_name: str) -> DownloadStatus:
    path = images_dir / file_name
    if path.exists():
        return "skipped"
    url = f"{COCO_TRAIN2014_URL}/{file_name}"
    _download_url(url, path)
    return "downloaded"


def _download_url(url: str, path: Path) -> None:
    part_path = path.with_name(f"{path.name}.part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if chunk:
                            handle.write(chunk)
            os.replace(part_path, path)
            return
        except requests.RequestException:
            part_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                raise


if __name__ == "__main__":
    main()
