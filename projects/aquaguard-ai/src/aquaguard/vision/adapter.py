from typing import Protocol

from aquaguard.vision.calibration import HomographyProjector
from aquaguard.vision.models import PixelTrackObservation
from aquaguard.world.models import CameraObservation


class TrackAssociator(Protocol):
    def identity(self, observation: PixelTrackObservation) -> str: ...


class ProvidedTrackAssociator:
    """Use an explicit global ID, otherwise keep camera-local identities isolated."""

    def identity(self, observation: PixelTrackObservation) -> str:
        return observation.global_track_id or (
            f"{observation.camera_id}:{observation.local_track_id}"
        )


class CalibratedObservationAdapter:
    def __init__(
        self,
        projectors: dict[str, HomographyProjector],
        associator: TrackAssociator | None = None,
    ):
        self.projectors = projectors
        self.associator = associator or ProvidedTrackAssociator()

    def convert(self, observations: list[PixelTrackObservation]) -> list[CameraObservation]:
        converted = []
        for item in observations:
            item.validate()
            projector = self.projectors.get(item.camera_id)
            if projector is None:
                raise ValueError(f"No calibration configured for camera {item.camera_id}")
            pool_x, pool_y = projector.project(item.anchor_x, item.anchor_y)
            converted.append(
                CameraObservation(
                    camera_id=item.camera_id,
                    track_id=self.associator.identity(item),
                    timestamp=item.timestamp,
                    pool_x=pool_x,
                    pool_y=pool_y,
                    confidence=item.confidence,
                    head_submerged=item.head_submerged,
                    body_vertical=item.body_vertical,
                    struggle=item.struggle,
                    motion=item.motion,
                    occlusion=item.occlusion,
                    head_in_water_region=item.head_in_water_region,
                    water_relation_confidence=item.water_relation_confidence,
                )
            )
        return converted
