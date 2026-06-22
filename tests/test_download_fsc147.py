from __future__ import annotations

from scripts.download_fsc147 import _annotation_urls, _is_image_path


def test_fsc147_annotation_urls_include_official_and_hf_sources() -> None:
    urls = _annotation_urls("ImageClasses_FSC147.txt")

    assert urls[0] == (
        "https://raw.githubusercontent.com/cvlab-stonybrook/"
        "LearningToCountEverything/master/data/ImageClasses_FSC147.txt"
    )
    assert urls[1] == "https://huggingface.co/datasets/isentropic/FSC147/resolve/main/ImageClasses_FSC147.txt"


def test_fsc147_image_filter_excludes_density_maps() -> None:
    assert _is_image_path("images_384_VarV2/1.jpg") is True
    assert _is_image_path("images_384_VarV2/nested/1.png") is True
    assert _is_image_path("gt_density_map_adaptive_384_VarV2/1.npy") is False
    assert _is_image_path("annotation_FSC147_384.json") is False
