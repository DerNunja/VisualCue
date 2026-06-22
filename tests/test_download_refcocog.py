from __future__ import annotations

from scripts.download_refcocog import _image_urls


def test_refcocog_image_urls_include_http_fallback() -> None:
    urls = _image_urls("COCO_train2014_000000000001.jpg")

    assert urls == (
        "https://images.cocodataset.org/train2014/COCO_train2014_000000000001.jpg",
        "http://images.cocodataset.org/train2014/COCO_train2014_000000000001.jpg",
    )
