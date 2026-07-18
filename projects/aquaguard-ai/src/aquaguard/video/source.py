import time
from collections.abc import Callable
from typing import Protocol

from aquaguard.video.models import FrameRead, VideoFrame


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, object | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int | str], Capture]


def normalize_source(source: str) -> int | str:
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def _opencv_capture(source: int | str) -> Capture:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' dependencies to read video") from exc
    return cv2.VideoCapture(source)


class OpenCVFrameSource:
    """Lazy video source with timestamps and bounded exponential reconnect backoff."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        reconnect: bool = True,
    ):
        if not camera_id:
            raise ValueError("camera_id is required")
        if initial_delay <= 0 or max_delay < initial_delay:
            raise ValueError("Reconnect delays are invalid")
        self.camera_id = camera_id
        self.source = normalize_source(source)
        self.capture_factory = capture_factory or _opencv_capture
        self.clock = clock
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.reconnect = reconnect
        self._capture: Capture | None = None
        self._next_connect_at = 0.0
        self._delay = initial_delay
        self._sequence = 0
        self._ended = False

    def read(self) -> FrameRead:
        now = self.clock()
        if self._ended:
            return FrameRead(ok=False, error="end_of_stream")
        if not self._connect(now):
            return FrameRead(ok=False, error="source_unavailable", retry_at=self._next_connect_at)

        assert self._capture is not None
        ok, image = self._capture.read()
        if ok and image is not None:
            frame = VideoFrame(self.camera_id, self._sequence, now, image)
            self._sequence += 1
            return FrameRead(ok=True, frame=frame)

        self._release_capture()
        if not self.reconnect:
            self._ended = True
            return FrameRead(ok=False, error="end_of_stream")
        self._schedule_retry(now)
        return FrameRead(ok=False, error="frame_read_failed", retry_at=self._next_connect_at)

    def close(self) -> None:
        self._ended = True
        self._release_capture()

    def _connect(self, now: float) -> bool:
        if self._capture is not None and self._capture.isOpened():
            return True
        if now < self._next_connect_at:
            return False
        self._release_capture()
        capture = self.capture_factory(self.source)
        if capture.isOpened():
            self._capture = capture
            self._delay = self.initial_delay
            return True
        capture.release()
        self._schedule_retry(now)
        return False

    def _schedule_retry(self, now: float) -> None:
        self._next_connect_at = now + self._delay
        self._delay = min(self._delay * 2, self.max_delay)

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
