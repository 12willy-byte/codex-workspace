import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aquaguard.video.models import VideoFrame
from aquaguard.vision.models import PixelTrackObservation


class TrackModel(Protocol):
    def track(self, image: object, **kwargs) -> list[object]: ...


def _load_ultralytics(model_path: str) -> TrackModel:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' dependencies to use Ultralytics") from exc
    return YOLO(model_path)


def _list(value: object) -> list:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


class UltralyticsTrackAnalyzer:
    """Person detector/tracker adapter; it does not infer drowning-specific features."""

    capabilities = ("person_detection", "local_tracking", "anchor_motion")

    def __init__(
        self,
        model_path: str,
        *,
        confidence: float = 0.5,
        device: str | None = None,
        model_loader: Callable[[str], TrackModel] | None = None,
    ):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if model_loader is None and not Path(model_path).is_file():
            raise ValueError("Ultralytics model_path must reference an existing local file")
        self.model = (model_loader or _load_ultralytics)(model_path)
        self.confidence = confidence
        self.device = device
        self._anchors: dict[str, tuple[float, float]] = {}

    def analyze(self, frame: VideoFrame) -> list[PixelTrackObservation]:
        options = {
            "persist": True,
            "verbose": False,
            "conf": self.confidence,
            "classes": [0],
        }
        if self.device is not None:
            options["device"] = self.device
        results = self.model.track(frame.image, **options)
        observations = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "id", None) is None:
                continue
            coordinates = _list(boxes.xyxy)
            confidences = _list(boxes.conf)
            track_ids = _list(boxes.id)
            classes = _list(boxes.cls)
            detections = zip(coordinates, confidences, track_ids, classes, strict=True)
            for detection_index, (xyxy, score, track_id, class_id) in enumerate(detections):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                local_track_id = str(int(track_id))
                anchor = ((x1 + x2) / 2, y2)
                previous = self._anchors.get(local_track_id)
                height = max(y2 - y1, 1.0)
                motion = 0.0
                if previous is not None:
                    motion = min(math.dist(previous, anchor) / height, 1.0)
                self._anchors[local_track_id] = anchor
                features = self._additional_features(
                    result, detection_index, local_track_id, height
                )
                observations.append(
                    PixelTrackObservation(
                        camera_id=frame.camera_id,
                        local_track_id=local_track_id,
                        timestamp=frame.timestamp,
                        anchor_x=anchor[0],
                        anchor_y=anchor[1],
                        confidence=float(score),
                        motion=motion,
                        **features,
                    )
                )
        return observations

    def _additional_features(
        self,
        result: object,
        detection_index: int,
        local_track_id: str,
        body_height: float,
    ) -> dict[str, float]:
        return {}


class UltralyticsPoseTrackAnalyzer(UltralyticsTrackAnalyzer):
    """Pose tracking adapter; head-submersion and struggle remain intentionally unset."""

    capabilities = (
        "person_detection",
        "local_tracking",
        "anchor_motion",
        "body_verticality",
        "pose_visibility",
    )

    def __init__(self, *args, keypoint_confidence: float = 0.3, **kwargs):
        if not 0 <= keypoint_confidence <= 1:
            raise ValueError("keypoint_confidence must be in [0, 1]")
        super().__init__(*args, **kwargs)
        self.keypoint_confidence = keypoint_confidence

    def _additional_features(
        self,
        result: object,
        detection_index: int,
        local_track_id: str,
        body_height: float,
    ) -> dict[str, float]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None:
            raise RuntimeError("Pose analyzer received tracked boxes without pose keypoints")
        coordinates = _list(keypoints.xy)
        confidence_value = getattr(keypoints, "conf", None)
        if confidence_value is None:
            raise RuntimeError("Pose analyzer requires keypoint confidence values")
        confidences = _list(confidence_value)
        if detection_index >= len(coordinates) or detection_index >= len(confidences):
            raise RuntimeError("Pose keypoints are not aligned with tracked boxes")
        points = coordinates[detection_index]
        scores = confidences[detection_index]
        if len(points) < 17 or len(scores) < 17:
            raise RuntimeError("Pose analyzer requires the 17-keypoint COCO layout")
        visible = [float(score) >= self.keypoint_confidence for score in scores[:17]]
        visibility = sum(visible) / 17
        body_vertical = 0.0
        torso_indices = (5, 6, 11, 12)
        if all(visible[index] for index in torso_indices):
            shoulder = self._midpoint(points[5], points[6])
            hip = self._midpoint(points[11], points[12])
            dx = abs(hip[0] - shoulder[0])
            dy = abs(hip[1] - shoulder[1])
            body_vertical = dy / max(dx + dy, 1e-9)
        return {"body_vertical": body_vertical, "occlusion": 1.0 - visibility}

    @staticmethod
    def _midpoint(first: list[float], second: list[float]) -> tuple[float, float]:
        return (
            (float(first[0]) + float(second[0])) / 2,
            (float(first[1]) + float(second[1])) / 2,
        )
