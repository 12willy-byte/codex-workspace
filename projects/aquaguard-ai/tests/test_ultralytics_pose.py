import pytest

from aquaguard.video import VideoFrame
from aquaguard.vision import UltralyticsPoseTrackAnalyzer


class FakeTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10, 10, 30, 110]])
    conf = FakeTensor([0.9])
    id = FakeTensor([7])
    cls = FakeTensor([0])


class FakeKeypoints:
    def __init__(self, points: list[list[float]], scores: list[float]) -> None:
        self.xy = FakeTensor([points])
        self.conf = FakeTensor([scores])


class FakeResult:
    boxes = FakeBoxes()

    def __init__(self, keypoints: FakeKeypoints | None) -> None:
        self.keypoints = keypoints


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self.result = result

    def track(self, image: object, **kwargs) -> list[FakeResult]:
        return [self.result]


def points(vertical: bool = True) -> list[list[float]]:
    values = [[20.0, 20.0] for _ in range(17)]
    values[5], values[6] = [15, 30], [25, 30]
    if vertical:
        values[11], values[12] = [15, 80], [25, 80]
    else:
        values[11], values[12] = [65, 30], [75, 30]
    return values


def analyzer(result: FakeResult) -> UltralyticsPoseTrackAnalyzer:
    return UltralyticsPoseTrackAnalyzer(
        "fake-pose.pt", model_loader=lambda _: FakeModel(result)
    )


def test_pose_analyzer_computes_vertical_torso_from_same_tracked_result() -> None:
    result = FakeResult(FakeKeypoints(points(vertical=True), [0.9] * 17))

    observation = analyzer(result).analyze(VideoFrame("cam-a", 0, 1, "pixels"))[0]

    assert observation.local_track_id == "7"
    assert observation.body_vertical == 1
    assert observation.occlusion == pytest.approx(0)
    assert observation.head_submerged == 0
    assert observation.struggle == 0


def test_pose_analyzer_computes_horizontal_torso() -> None:
    result = FakeResult(FakeKeypoints(points(vertical=False), [0.9] * 17))

    observation = analyzer(result).analyze(VideoFrame("cam-a", 0, 1, "pixels"))[0]

    assert observation.body_vertical == 0


def test_low_confidence_torso_does_not_create_vertical_signal() -> None:
    scores = [0.9] * 17
    scores[11] = 0.1
    result = FakeResult(FakeKeypoints(points(vertical=True), scores))

    observation = analyzer(result).analyze(VideoFrame("cam-a", 0, 1, "pixels"))[0]

    assert observation.body_vertical == 0
    assert observation.occlusion == pytest.approx(1 / 17)


def test_pose_analyzer_fails_when_tracked_boxes_have_no_keypoints() -> None:
    with pytest.raises(RuntimeError, match="without pose keypoints"):
        analyzer(FakeResult(None)).analyze(VideoFrame("cam-a", 0, 1, "pixels"))
