from dataclasses import dataclass

from aquaguard.video.models import VideoFrame


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    event_id: str
    camera_id: str
    trigger_at: float
    starts_at: float
    ends_at: float


@dataclass(frozen=True, slots=True)
class EvidenceClip:
    event_id: str
    camera_id: str
    trigger_at: float
    starts_at: float
    ends_at: float
    frames: tuple[VideoFrame, ...]
    complete: bool
    missing_pre_event: bool
    missing_post_event: bool
