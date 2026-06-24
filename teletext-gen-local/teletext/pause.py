import sys
import threading


class PauseController:
    def __init__(self, stdin=None):
        self._unpaused = threading.Event()
        self._unpaused.set()
        self._running = True
        self._stdin = stdin or sys.stdin
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._fd = None
        self._old_settings = None
        self._supported = False

    def start(self):
        if not self._stdin.isatty():
            return
        try:
            import termios
            import tty

            self._fd = self._stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._supported = True
        except (ImportError, OSError, AttributeError):
            return
        self._listener.start()

    def stop(self):
        self._running = False
        if self._supported and self._old_settings is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _listen(self):
        import select

        while self._running:
            r, _, _ = select.select([self._stdin], [], [], 0.2)
            if r:
                ch = self._stdin.read(1)
                if ch in ("p", "P", " "):
                    if self._unpaused.is_set():
                        self.pause()
                    else:
                        self.resume()

    def pause(self):
        self._unpaused.clear()
        print("\n⏸  Paused (press p/space to resume)", file=sys.stderr)

    def resume(self):
        self._unpaused.set()
        print("\n▶  Resumed", file=sys.stderr)

    def wait_if_paused(self):
        self._unpaused.wait()

    @property
    def is_paused(self):
        return not self._unpaused.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
