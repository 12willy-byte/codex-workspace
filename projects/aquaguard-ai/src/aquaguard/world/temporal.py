from collections import defaultdict, deque

from aquaguard.world.models import FusedTrack


class TemporalTrackBuffer:
    """Bounded time memory for continuous-video feature fusion."""

    def __init__(self, window_seconds: float = 10.0, max_samples: int = 300):
        if window_seconds <= 0 or max_samples <= 0:
            raise ValueError("Temporal buffer limits must be positive")
        self.window_seconds = window_seconds
        self._tracks: dict[str, deque[FusedTrack]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    def append(self, track: FusedTrack) -> tuple[FusedTrack, ...]:
        history = self._tracks[track.track_id]
        history.append(track)
        cutoff = track.timestamp - self.window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        return tuple(history)
