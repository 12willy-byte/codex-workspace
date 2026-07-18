from collections import defaultdict

from aquaguard.world.models import CameraObservation, FusedTrack


class MultiCameraFusion:
    """Fuse observations already projected into a shared pool coordinate system."""

    def fuse(self, observations: list[CameraObservation]) -> list[FusedTrack]:
        grouped: dict[str, list[CameraObservation]] = defaultdict(list)
        for observation in observations:
            grouped[observation.track_id].append(observation)

        tracks = []
        for track_id, items in grouped.items():
            weights = [max(item.confidence * (1.0 - item.occlusion), 0.01) for item in items]
            total = sum(weights)

            def mean(field: str) -> float:
                return sum(
                    getattr(item, field) * weight for item, weight in zip(items, weights, strict=True)
                ) / total

            tracks.append(
                FusedTrack(
                    track_id=track_id,
                    timestamp=max(item.timestamp for item in items),
                    pool_x=mean("pool_x"),
                    pool_y=mean("pool_y"),
                    confidence=sum(item.confidence for item in items) / len(items),
                    head_submerged=mean("head_submerged"),
                    body_vertical=mean("body_vertical"),
                    struggle=mean("struggle"),
                    motion=mean("motion"),
                    occlusion=mean("occlusion"),
                    camera_ids=tuple(sorted({item.camera_id for item in items})),
                    head_in_water_region=mean("head_in_water_region"),
                    water_relation_confidence=mean("water_relation_confidence"),
                    wrist_motion=mean("wrist_motion"),
                    wrist_motion_confidence=mean("wrist_motion_confidence"),
                )
            )
        return tracks
