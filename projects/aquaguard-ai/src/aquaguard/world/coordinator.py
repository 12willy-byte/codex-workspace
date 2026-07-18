from copy import deepcopy
from dataclasses import dataclass
from threading import Lock

from aquaguard.world.models import CameraObservation
from aquaguard.world.pipeline import WorldModelPipeline


@dataclass(frozen=True, slots=True)
class CameraWorldFrame:
    timestamp: float
    observations: tuple[CameraObservation, ...]


class TemporalFusionCoordinator:
    """Serialize world-model access and fuse the latest near-time frame from each camera."""

    def __init__(self, pipeline: WorldModelPipeline, *, max_skew_seconds: float = 0.15):
        if max_skew_seconds < 0:
            raise ValueError("max_skew_seconds must not be negative")
        self.pipeline = pipeline
        self.max_skew_seconds = max_skew_seconds
        self._latest: dict[str, CameraWorldFrame] = {}
        self._last_emitted_at = float("-inf")
        self._last_result: dict | None = None
        self._lock = Lock()

    def process_frame(
        self,
        camera_id: str,
        timestamp: float,
        observations: list[CameraObservation],
    ) -> dict:
        if not camera_id:
            raise ValueError("camera_id is required")
        if any(item.camera_id != camera_id for item in observations):
            raise ValueError("Frame observations must belong to the supplied camera")
        with self._lock:
            previous = self._latest.get(camera_id)
            if previous is not None and timestamp < previous.timestamp:
                raise ValueError(f"Out-of-order world frame for camera {camera_id}")
            self._latest[camera_id] = CameraWorldFrame(timestamp, tuple(observations))
            if timestamp < self._last_emitted_at and self._last_result is not None:
                return deepcopy(self._last_result)

            selected = [
                frame
                for frame in self._latest.values()
                if abs(frame.timestamp - timestamp) <= self.max_skew_seconds
            ]
            batch = [item for frame in selected for item in frame.observations]
            result = self.pipeline.process(batch)
            result["sync"] = {
                "timestamp": timestamp,
                "camera_ids": sorted(
                    camera
                    for camera, frame in self._latest.items()
                    if frame in selected
                ),
                "max_skew_seconds": self.max_skew_seconds,
            }
            self._last_emitted_at = timestamp
            self._last_result = deepcopy(result)
            return result
