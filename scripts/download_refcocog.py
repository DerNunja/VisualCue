"""Download RefCOCOg annotations and only the images needed for one split."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
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
    "https://huggingface.co/datasets/jxu124/refcocog/archive/main.zip",
    "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
)
COCO_TRAIN2014_URLS = (
    "https://images.cocodataset.org/train2014",
    "http://images.cocodataset.org/train2014",
)
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

    _download_first_available(ANNOTATION_URLS, root, split_by)

    missing = [str(path) for path in (refs_path, instances_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(_missing_annotation_message(missing, root, split_by))


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


def _download_first_available(urls: tuple[str, ...], root: Path, split_by: str) -> None:
    errors: list[str] = []
    for index, url in enumerate(urls, start=1):
        archive_path = root / f"refcocog_source_{index}.part"
        try:
            print(f"downloading annotations: {url}")
            _download_url(url, archive_path)
            with tempfile.TemporaryDirectory(dir=root) as temp_dir:
                temp_root = Path(temp_dir)
                _extract_annotations(archive_path, temp_root, split_by)
                refs_path = temp_root / f"refs({split_by}).p"
                instances_path = temp_root / "instances.json"
                missing = [path.name for path in (refs_path, instances_path) if not path.exists()]
                if missing:
                    errors.append(f"{url}: missing {', '.join(missing)}")
                    continue
                shutil.copy2(refs_path, root / refs_path.name)
                shutil.copy2(instances_path, root / instances_path.name)
                return
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            archive_path.unlink(missing_ok=True)
    raise RuntimeError(_annotation_source_failure_message(root, split_by, errors))


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


def _missing_annotation_message(missing: list[str], root: Path, split_by: str) -> str:
    return (
        f"missing RefCOCOg annotation files: {', '.join(missing)}. "
        f"Expected {root / f'refs({split_by}).p'} and {root / 'instances.json'}. "
        "Download the refer RefCOCOg archive from lichengunc/refer or its Wayback mirror; "
        "if your archive only contains refs, also place the COCO train2014 instances file as instances.json."
    )


def _annotation_source_failure_message(root: Path, split_by: str, errors: list[str]) -> str:
    missing = [str(root / f"refs({split_by}).p"), str(root / "instances.json")]
    return (
        "could not download a RefCOCOg annotation source with the required refer layout. "
        f"Required files: {', '.join(missing)}. "
        "Use the lichengunc/refer RefCOCOg archive or Wayback mirror; if using a refs-only mirror, "
        "copy COCO annotations_trainval2014/annotations/instances_train2014.json to instances.json.\n"
        + "\n".join(errors)
    )


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
    _download_from_urls(_image_urls(file_name), path)
    return "downloaded"


def _download_url(url: str, path: Path) -> None:
    _download_from_urls((url,), path)


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


def _image_urls(file_name: str) -> tuple[str, ...]:
    return tuple(f"{base_url}/{file_name}" for base_url in COCO_TRAIN2014_URLS)


if __name__ == "__main__":
    main()
