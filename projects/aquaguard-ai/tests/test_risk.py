from aquaguard.domain import AlarmLevel, RiskFeatures
from aquaguard.risk import RiskEngine


def dangerous_features() -> RiskFeatures:
    return RiskFeatures(head_underwater=1, vertical_body=1, abnormal_motion=1, temporal_risk=1)


def test_weighted_score() -> None:
    result = RiskEngine().assess("p1", dangerous_features())
    assert result.score == 100
    assert result.level == AlarmLevel.WARNING
    assert not result.confirmed


def test_confirmation_requires_consecutive_frames() -> None:
    engine = RiskEngine(confirmation_frames=3)
    assert not engine.assess("p1", dangerous_features()).confirmed
    assert not engine.assess("p1", dangerous_features()).confirmed
    result = engine.assess("p1", dangerous_features())
    assert result.confirmed
    assert result.level == AlarmLevel.EMERGENCY


def test_safe_frame_resets_confirmation() -> None:
    engine = RiskEngine(confirmation_frames=2)
    engine.assess("p1", dangerous_features())
    engine.assess("p1", RiskFeatures(head_underwater=0, vertical_body=0, abnormal_motion=0, temporal_risk=0))
    assert not engine.assess("p1", dangerous_features()).confirmed
