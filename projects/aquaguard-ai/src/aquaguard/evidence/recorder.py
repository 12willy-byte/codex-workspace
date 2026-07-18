from aquaguard.evidence.buffer import FrameRingBuffer
from aquaguard.evidence.models import EvidenceClip, EvidenceWindow
from aquaguard.video.models import VideoFrame


class EvidenceRecorder:
    """Lock pre/post-event windows without performing video encoding."""

    def __init__(self, buffer: FrameRingBuffer | None = None):
        self.buffer = buffer or FrameRingBuffer()
        self.pending: dict[str, EvidenceWindow] = {}
        self.completed: dict[str, EvidenceClip] = {}

    def ingest(self, frame: VideoFrame) -> None:
        self.buffer.append(frame)
        self.finalize_ready(frame.timestamp, camera_id=frame.camera_id)

    def request(
        self,
        event_id: str,
        camera_id: str,
        trigger_at: float,
        *,
        pre_seconds: float = 30.0,
        post_seconds: float = 60.0,
    ) -> EvidenceWindow:
        if not event_id or not camera_id:
            raise ValueError("event_id and camera_id are required")
        if pre_seconds < 0 or post_seconds < 0:
            raise ValueError("Evidence durations must not be negative")
        if event_id in self.pending or event_id in self.completed:
            raise ValueError(f"Evidence already exists for event {event_id}")
        window = EvidenceWindow(
            event_id,
            camera_id,
            trigger_at,
            trigger_at - pre_seconds,
            trigger_at + post_seconds,
        )
        self.pending[event_id] = window
        return window

    def finalize_ready(self, now: float, camera_id: str | None = None) -> list[EvidenceClip]:
        ready = [
            window
            for window in self.pending.values()
            if window.ends_at <= now and (camera_id is None or window.camera_id == camera_id)
        ]
        return [self._finalize(window) for window in ready]

    def force_finalize(self, event_id: str) -> EvidenceClip:
        window = self.pending.get(event_id)
        if window is None:
            raise KeyError(event_id)
        return self._finalize(window)

    def get(self, event_id: str) -> EvidenceWindow | EvidenceClip | None:
        return self.completed.get(event_id) or self.pending.get(event_id)

    def _finalize(self, window: EvidenceWindow) -> EvidenceClip:
        frames = self.buffer.between(window.camera_id, window.starts_at, window.ends_at)
        bounds = self.buffer.bounds(window.camera_id)
        missing_pre = bounds is None or bounds[0] > window.starts_at
        missing_post = bounds is None or bounds[1] < window.ends_at
        clip = EvidenceClip(
            window.event_id,
            window.camera_id,
            window.trigger_at,
            window.starts_at,
            window.ends_at,
            frames,
            complete=bool(frames) and not missing_pre and not missing_post,
            missing_pre_event=missing_pre,
            missing_post_event=missing_post,
        )
        self.completed[window.event_id] = clip
        del self.pending[window.event_id]
        return clip
