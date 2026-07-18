from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PixelTrackObservation:
    """Structured output expected from a detector/tracker for one video frame."""

    camera_id: str
    local_track_id: str
    timestamp: float
    anchor_x: float
    anchor_y: float
    global_track_id: str | None = None
    confidence: float = 1.0
    head_submerged: float = 0.0
    body_vertical: float = 0.0
    struggle: float = 0.0
    motion: float = 0.0
    occlusion: float = 0.0
    head_in_water_region: float = 0.0
    water_relation_confidence: float = 0.0
    wrist_motion: float = 0.0
    wrist_motion_confidence: float = 0.0

    def validate(self) -> None:
        if not self.camera_id or not self.local_track_id:
            raise ValueError("camera_id and local_track_id are required")
        for name in (
            "confidence",
            "head_submerged",
            "body_vertical",
            "struggle",
            "motion",
            "occlusion",
            "head_in_water_region",
            "water_relation_confidence",
            "wrist_motion",
            "wrist_motion_confidence",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
