from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoFrame:
    camera_id: str
    sequence: int
    timestamp: float
    image: object


@dataclass(frozen=True, slots=True)
class FrameRead:
    ok: bool
    frame: VideoFrame | None = None
    error: str | None = None
    retry_at: float | None = None
