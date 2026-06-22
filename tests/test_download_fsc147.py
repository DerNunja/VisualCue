from __future__ import annotations

from scripts.download_fsc147 import _annotation_urls, _is_image_path, _zip_image_relative_path


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


def test_fsc147_zip_filter_extracts_only_image_directory() -> None:
    assert _zip_image_relative_path("FSC147/images_384_VarV2/1.jpg").as_posix() == "1.jpg"
    assert _zip_image_relative_path("images_384_VarV2/2.png").as_posix() == "2.png"
    assert _zip_image_relative_path("FSC147/gt_density_map_adaptive_384_VarV2/1.npy") is None
