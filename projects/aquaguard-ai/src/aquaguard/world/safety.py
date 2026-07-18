from collections import defaultdict

from aquaguard.world.models import FusedTrack, RiskForecast, SafetyDecision


class IndependentSafetySupervisor:
    """Deterministic guardrail that does not trust the predictor output alone."""

    def __init__(
        self,
        warning_threshold: float = 0.65,
        alarm_threshold: float = 0.85,
        confirmations: int = 2,
    ):
        self.warning_threshold = warning_threshold
        self.alarm_threshold = alarm_threshold
        self.confirmations = confirmations
        self._high_risk_counts: dict[str, int] = defaultdict(int)

    def evaluate(self, track: FusedTrack, forecast: RiskForecast) -> SafetyDecision:
        severe_signal = track.head_submerged >= 0.8 and (
            track.body_vertical >= 0.65 or track.motion <= 0.1
        )
        reliable = forecast.uncertainty <= 0.55 and track.confidence >= 0.45
        high_risk = max(forecast.current_risk, forecast.future_risk) >= self.alarm_threshold
        self._high_risk_counts[track.track_id] = (
            self._high_risk_counts[track.track_id] + 1 if high_risk and reliable else 0
        )
        confirmed = self._high_risk_counts[track.track_id] >= self.confirmations

        if severe_signal or confirmed:
            guard_reason = "independent_severe_signal" if severe_signal else "temporally_confirmed"
            return SafetyDecision(
                track.track_id, "emergency", True, forecast.reasons + (guard_reason,)
            )
        if not reliable:
            return SafetyDecision(
                track.track_id, "observe", False, forecast.reasons + ("high_uncertainty",)
            )
        if max(forecast.current_risk, forecast.future_risk) >= self.warning_threshold:
            return SafetyDecision(track.track_id, "warning", False, forecast.reasons)
        return SafetyDecision(track.track_id, "normal", False, forecast.reasons)
