from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PoolGeometry:
    width_m: float = 25.0
    height_m: float = 10.0
    grid_width: int = 25
    grid_height: int = 10

    def validate(self) -> None:
        if min(self.width_m, self.height_m, self.grid_width, self.grid_height) <= 0:
            raise ValueError("Pool and grid dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CameraObservation:
    camera_id: str
    track_id: str
    timestamp: float
    pool_x: float
    pool_y: float
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

    def validate(self, geometry: PoolGeometry) -> None:
        if not self.camera_id or not self.track_id:
            raise ValueError("camera_id and track_id are required")
        if not 0 <= self.pool_x <= geometry.width_m or not 0 <= self.pool_y <= geometry.height_m:
            raise ValueError("Observation is outside the configured pool")
        features = (
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
        )
        for name in features:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FusedTrack:
    track_id: str
    timestamp: float
    pool_x: float
    pool_y: float
    confidence: float
    head_submerged: float
    body_vertical: float
    struggle: float
    motion: float
    occlusion: float
    camera_ids: tuple[str, ...]
    head_in_water_region: float = 0.0
    water_relation_confidence: float = 0.0
    wrist_motion: float = 0.0
    wrist_motion_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskForecast:
    track_id: str
    current_risk: float
    future_risk: float
    time_to_critical_seconds: float | None
    uncertainty: float
    reasons: tuple[str, ...]
    predictor_kind: str = "deterministic_baseline"
    model_version: str | None = None
    assistive_alerting_eligible: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    track_id: str
    level: str
    should_alarm: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)
