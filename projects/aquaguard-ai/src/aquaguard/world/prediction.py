from aquaguard.world.models import FusedTrack, RiskForecast


class FutureRiskPredictor:
    """Explainable baseline behind the interface for a future learned video model."""

    critical_threshold = 0.85

    def predict(self, history: tuple[FusedTrack, ...]) -> RiskForecast:
        if not history:
            raise ValueError("History must not be empty")
        current = history[-1]
        current_risk = self._risk(current)
        trend = self._trend(history)
        future_risk = min(1.0, max(0.0, current_risk + trend * 3.0))
        uncertainty = min(
            1.0,
            0.45 * current.occlusion
            + 0.35 * (1.0 - current.confidence)
            + (0.2 if len(history) < 3 else 0.0),
        )
        time_to_critical = None
        if current_risk >= self.critical_threshold:
            time_to_critical = 0.0
        elif trend > 0:
            time_to_critical = (self.critical_threshold - current_risk) / trend

        reasons = tuple(
            reason
            for active, reason in (
                (current.head_submerged >= 0.6, "head_submerged"),
                (current.body_vertical >= 0.65, "vertical_body"),
                (current.struggle >= 0.6, "struggle"),
                (current.motion <= 0.15, "low_motion"),
                (trend >= 0.03, "risk_increasing"),
            )
            if active
        )
        return RiskForecast(
            current.track_id,
            current_risk,
            future_risk,
            time_to_critical,
            uncertainty,
            reasons,
        )

    @staticmethod
    def _risk(track: FusedTrack) -> float:
        return min(
            1.0,
            0.4 * track.head_submerged
            + 0.25 * track.body_vertical
            + 0.2 * track.struggle
            + 0.15 * (1.0 - track.motion),
        )

    def _trend(self, history: tuple[FusedTrack, ...]) -> float:
        if len(history) < 2:
            return 0.0
        elapsed = history[-1].timestamp - history[0].timestamp
        if elapsed <= 0:
            return 0.0
        return (self._risk(history[-1]) - self._risk(history[0])) / elapsed
