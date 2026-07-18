from collections import defaultdict, deque

from aquaguard.video.models import VideoFrame


class FrameRingBuffer:
    """Per-camera bounded memory buffer ordered by frame timestamp."""

    def __init__(self, retention_seconds: float = 120.0, max_frames_per_camera: int = 3600):
        if retention_seconds <= 0 or max_frames_per_camera <= 0:
            raise ValueError("Buffer limits must be positive")
        self.retention_seconds = retention_seconds
        self._frames: dict[str, deque[VideoFrame]] = defaultdict(
            lambda: deque(maxlen=max_frames_per_camera)
        )

    def append(self, frame: VideoFrame) -> None:
        frames = self._frames[frame.camera_id]
        if frames and frame.timestamp < frames[-1].timestamp:
            raise ValueError("Frame timestamps must be monotonic per camera")
        frames.append(frame)
        cutoff = frame.timestamp - self.retention_seconds
        while frames and frames[0].timestamp < cutoff:
            frames.popleft()

    def between(self, camera_id: str, starts_at: float, ends_at: float) -> tuple[VideoFrame, ...]:
        if ends_at < starts_at:
            raise ValueError("Evidence range is invalid")
        return tuple(
            frame
            for frame in self._frames.get(camera_id, ())
            if starts_at <= frame.timestamp <= ends_at
        )

    def bounds(self, camera_id: str) -> tuple[float, float] | None:
        frames = self._frames.get(camera_id)
        if not frames:
            return None
        return frames[0].timestamp, frames[-1].timestamp
