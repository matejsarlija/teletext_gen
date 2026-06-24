import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teletext.pause import PauseController


def test_pause_resume_toggle():
    ctl = PauseController()
    assert not ctl.is_paused
    ctl.pause()
    assert ctl.is_paused
    ctl.resume()
    assert not ctl.is_paused


def test_pause_resume_idempotent():
    ctl = PauseController()
    ctl.pause()
    ctl.pause()
    assert ctl.is_paused
    ctl.resume()
    ctl.resume()
    assert not ctl.is_paused


def test_wait_if_paused_blocks_then_unblocks():
    ctl = PauseController()
    ctl.pause()

    start = time.monotonic()
    timer = threading_timer(0.1, ctl.resume)
    timer.start()
    ctl.wait_if_paused()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.08
    assert not ctl.is_paused


def test_wait_if_paused_does_not_block_when_running():
    ctl = PauseController()
    start = time.monotonic()
    ctl.wait_if_paused()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05


def test_stdin_pipe_graceful_degradation(tmp_path):
    pipe_path = tmp_path / "stdin_fifo"
    with open(pipe_path, "w") as f:
        pass
    with open(pipe_path, "r") as f:
        ctl = PauseController(stdin=f)
        ctl.start()
        assert not ctl.is_paused
        ctl.stop()


def test_context_manager_cleanup():
    ctl = PauseController()
    with ctl as p:
        assert p is ctl
    assert not ctl._running
    assert ctl._old_settings is None


def test_trigger_pause_via_stdin_pipe():
    import io
    fake_stdin = io.StringIO("p")
    fake_stdin.isatty = lambda: True
    fake_stdin.fileno = lambda: (_ for _ in ()).throw(io.UnsupportedOperation)

    ctl = PauseController(stdin=fake_stdin)
    ctl.start()
    # start() will bail due to fileno() failure, so is_paused should still be False
    assert not ctl.is_paused
    ctl.stop()


def threading_timer(delay, func):
    import threading
    return threading.Timer(delay, func)
