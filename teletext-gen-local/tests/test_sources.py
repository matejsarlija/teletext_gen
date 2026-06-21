import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teletext.scraper import _build_url
from teletext.sources import SOURCES, ct_subpage_suffix


def test_ct_subpage_suffix() -> None:
    assert ct_subpage_suffix(1) == "A"
    assert ct_subpage_suffix(2) == "B"
    assert ct_subpage_suffix(8) == "H"


def test_ct_image_url() -> None:
    assert (
        _build_url(SOURCES["ct"], 100, 1)
        == "https://api-teletext.ceskatelevize.cz/pages/100A/image.webp"
    )
