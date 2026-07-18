import pytest

from aquaguard.video import VideoFrame
from aquaguard.vision import UltralyticsTrackAnalyzer


class FakeTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


class FakeBoxes:
    def __init__(self, *, track_ids: list | None = None) -> None:
        self.xyxy = FakeTensor([[10, 20, 30, 60], [40, 10, 60, 50]])
        self.conf = FakeTensor([0.9, 0.8])
        self.id = None if track_ids is None else FakeTensor(track_ids)
        self.cls = FakeTensor([0, 2])


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes
        self.calls: list[dict] = []

    def track(self, image: object, **kwargs) -> list[FakeResult]:
        self.calls.append({"image": image, **kwargs})
        return [FakeResult(self.boxes)]


def analyzer(model: FakeModel) -> UltralyticsTrackAnalyzer:
    return UltralyticsTrackAnalyzer("fake.pt", model_loader=lambda _: model)


def test_analyzer_emits_only_tracked_people() -> None:
    model = FakeModel(FakeBoxes(track_ids=[7, 8]))

    observations = analyzer(model).analyze(VideoFrame("cam-a", 0, 1.5, "pixels"))

    assert len(observations) == 1
    assert observations[0].local_track_id == "7"
    assert observations[0].anchor_x == 20
    assert observations[0].anchor_y == 60
    assert observations[0].confidence == 0.9
    assert observations[0].head_submerged == 0
    assert model.calls[0]["persist"] is True
    assert model.calls[0]["classes"] == [0]


def test_analyzer_skips_detections_without_tracking_ids() -> None:
    model = FakeModel(FakeBoxes())

    assert analyzer(model).analyze(VideoFrame("cam-a", 0, 1, "pixels")) == []


def test_analyzer_estimates_normalized_anchor_motion() -> None:
    model = FakeModel(FakeBoxes(track_ids=[7, 8]))
    detector = analyzer(model)
    detector.analyze(VideoFrame("cam-a", 0, 1, "pixels"))
    model.boxes.xyxy = FakeTensor([[20, 20, 40, 60], [40, 10, 60, 50]])

    observation = detector.analyze(VideoFrame("cam-a", 1, 2, "pixels"))[0]

    assert observation.motion == 0.25


def test_analyzer_requires_existing_local_model_without_injected_loader(tmp_path) -> None:
    with pytest.raises(ValueError, match="existing local file"):
        UltralyticsTrackAnalyzer(str(tmp_path / "missing.pt"))
