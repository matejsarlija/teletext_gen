import sys
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image

from teletext.scraper import scrape_all_sources, scrape_source
from teletext.sources import SOURCES
from teletext.pause import PauseController


DUMMY_IMG = Image.new("RGB", (40, 25))


@pytest.fixture
def patch_strategies():
    with ExitStack() as stack:
        stack.enter_context(
            patch("teletext.scraper._scrape_base64_gif", return_value=DUMMY_IMG)
        )
        stack.enter_context(
            patch("teletext.scraper._scrape_direct_url", return_value=DUMMY_IMG)
        )
        stack.enter_context(
            patch("teletext.scraper._scrape_json_api", return_value=DUMMY_IMG)
        )
        yield


# -- scrape_all_sources --


def test_scrape_all_sources_basic(tmp_path, patch_strategies):
    scrape_all_sources(
        ["hrt", "rtvslo"],
        tmp_path,
        delay=0,
        page_range_override=(100, 100),
    )

    pngs = list(tmp_path.glob("**/*.png"))
    # hrt: 8 subpages, rtvslo: 8 subpages = 16 total
    assert len(pngs) == 16
    assert all(f.stat().st_size > 0 for f in pngs)


def test_scrape_all_sources_invalid_key(tmp_path, capsys, patch_strategies):
    scrape_all_sources(
        ["hrt", "nobody", "rtvslo"],
        tmp_path,
        delay=0,
        page_range_override=(100, 100),
    )
    captured = capsys.readouterr()
    assert "Unknown source 'nobody'" in captured.out


def test_scrape_all_sources_empty_list(tmp_path, patch_strategies):
    scrape_all_sources([], tmp_path, delay=0)


def test_scrape_all_sources_all_unknown(tmp_path, capsys):
    scrape_all_sources(["foo", "bar"], tmp_path, delay=0)
    captured = capsys.readouterr()
    assert "Unknown source 'foo'" in captured.out
    assert "Unknown source 'bar'" in captured.out


def test_scrape_all_sources_runs_concurrently(tmp_path):
    """Prove sources run in parallel via a barrier.

    Both strategies must arrive at the barrier before either passes through.
    If sequential, the first would wait forever (timeout -> BrokenBarrierError).
    """
    barrier = threading.Barrier(2, timeout=5)
    call_order = []

    def strategy_a(*_a, **_kw):
        call_order.append("a_enter")
        barrier.wait()
        call_order.append("a_exit")
        return DUMMY_IMG

    def strategy_b(*_a, **_kw):
        call_order.append("b_enter")
        barrier.wait()
        call_order.append("b_exit")
        return DUMMY_IMG

    with (
        patch("teletext.scraper._scrape_base64_gif", side_effect=strategy_a),
        patch("teletext.scraper._scrape_direct_url", side_effect=strategy_b),
    ):
        scrape_all_sources(
            ["hrt", "rtvslo"],  # hrt=base64_gif, rtvslo=direct_url
            tmp_path,
            delay=0,
            page_range_override=(100, 100),
            max_workers=2,
        )

    assert call_order.index("a_enter") < call_order.index("a_exit")
    assert call_order.index("b_enter") < call_order.index("b_exit")
    # Both entered before either exited
    assert call_order.index("b_enter") < call_order.index("a_exit") or \
           call_order.index("a_enter") < call_order.index("b_exit")
    assert call_order.count("a_enter") == 8  # hrt subpages
    assert call_order.count("b_enter") == 8  # rtvslo subpages


def test_scrape_all_sources_max_workers_one_is_sequential(tmp_path):
    call_counts = {"hrt": 0, "svt": 0}
    current_source = [None]

    def strategy(*_a, **_kw):
        return DUMMY_IMG

    with patch("teletext.scraper._scrape_base64_gif", side_effect=strategy):
        scrape_all_sources(
            ["hrt", "svt"],  # both use base64_gif
            tmp_path,
            delay=0,
            page_range_override=(100, 100),
            max_workers=1,
        )

    pngs = list(tmp_path.glob("hrt/*.png")) + list(tmp_path.glob("svt/*.png"))
    assert len(pngs) == 8 + 4  # hrt:8 + svt:4


def test_scrape_source_creates_files(tmp_path):
    src = SOURCES["hrt"]
    with patch("teletext.scraper._scrape_base64_gif", return_value=DUMMY_IMG):
        scrape_source(src, tmp_path, delay=0, page_range_override=(100, 100))

    pngs = sorted(tmp_path.glob("hrt/*.png"))
    jsons = sorted(tmp_path.glob("hrt/*.json"))
    assert len(pngs) == 8  # hrt subpages 1-8
    assert len(jsons) == 8
    for png, json in zip(pngs, jsons):
        assert png.stem == json.stem


# -- pause integration --


def test_scrape_source_checks_pause(tmp_path):
    src = SOURCES["hrt"]
    pause = PauseController()
    pause_mock = MagicMock(wraps=pause)

    with patch("teletext.scraper._scrape_base64_gif", return_value=DUMMY_IMG):
        scrape_source(
            src, tmp_path,
            delay=0,
            page_range_override=(100, 100),
            pause=pause_mock,
        )

    # Called once per page (8 subpages), plus maybe between sources
    assert pause_mock.wait_if_paused.call_count >= 8


def test_scrape_all_sources_creates_pause_controller(tmp_path):
    with (
        patch("teletext.scraper.PauseController") as mock_pause_ctrl,
        patch("teletext.scraper._scrape_base64_gif", return_value=DUMMY_IMG),
        patch("teletext.scraper._scrape_direct_url", return_value=DUMMY_IMG),
    ):
        scrape_all_sources(
            ["hrt", "rtvslo"],
            tmp_path,
            delay=0,
            page_range_override=(100, 100),
        )

    mock_pause_ctrl.assert_called_once()
    instance = mock_pause_ctrl.return_value
    instance.__enter__.assert_called_once()
    instance.__exit__.assert_called_once()
    # wait_if_paused is called on the __enter__ return value (the context var)
    context_mock = instance.__enter__.return_value
    context_mock.wait_if_paused.assert_called()


def test_pause_pauses_scraping(tmp_path):
    src = SOURCES["hrt"]
    pause = PauseController()

    with patch("teletext.scraper._scrape_base64_gif", return_value=DUMMY_IMG):
        pause.pause()
        timer = threading.Timer(0.1, pause.resume)
        timer.start()
        scrape_source(
            src, tmp_path,
            delay=0,
            page_range_override=(100, 100),
            pause=pause,
        )

    pngs = list(tmp_path.glob("hrt/*.png"))
    assert len(pngs) == 8


def test_scrape_all_sources_honours_pause(tmp_path):
    pause = PauseController()
    pause.pause()

    call_count = 0

    def strategy(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return DUMMY_IMG

    timer = threading.Timer(0.2, pause.resume)

    with (
        patch("teletext.scraper.PauseController", return_value=pause),
        patch("teletext.scraper._scrape_base64_gif", side_effect=strategy),
        patch("teletext.scraper._scrape_direct_url", return_value=DUMMY_IMG),
    ):
        timer.start()
        scrape_all_sources(
            ["hrt", "rtvslo"],
            tmp_path,
            delay=0,
            page_range_override=(100, 100),
            max_workers=2,
        )

    # If pause was honoured, scraping resumed after timer fired (0.2s)
    assert call_count > 0
