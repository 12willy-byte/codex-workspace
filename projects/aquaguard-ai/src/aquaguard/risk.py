from collections import defaultdict

from aquaguard.domain import AlarmLevel, RiskAssessment, RiskFeatures


class RiskEngine:
    """Deterministic baseline; weights require validation before production use."""

    WEIGHTS = {"head_underwater": 35, "vertical_body": 25, "abnormal_motion": 20, "temporal_risk": 20}

    def __init__(self, threshold: float = 80, confirmation_frames: int = 3) -> None:
        self.threshold = threshold
        self.confirmation_frames = confirmation_frames
        self._consecutive: dict[str, int] = defaultdict(int)

    def assess(self, track_id: str, features: RiskFeatures) -> RiskAssessment:
        score = round(sum(getattr(features, key) * weight for key, weight in self.WEIGHTS.items()), 2)
        self._consecutive[track_id] = self._consecutive[track_id] + 1 if score >= self.threshold else 0
        confirmed = self._consecutive[track_id] >= self.confirmation_frames
        return RiskAssessment(score=score, level=self._level(score, confirmed), confirmed=confirmed)

    @staticmethod
    def _level(score: float, confirmed: bool) -> AlarmLevel:
        if confirmed:
            return AlarmLevel.EMERGENCY
        if score >= 70:
            return AlarmLevel.WARNING
        if score >= 40:
            return AlarmLevel.WATCH
        return AlarmLevel.NORMAL
