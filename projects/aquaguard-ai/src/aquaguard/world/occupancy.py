from aquaguard.world.models import FusedTrack, PoolGeometry


class PoolOccupancy:
    """Pool BEV with person-occupancy and instantaneous-danger planes."""

    def __init__(self, geometry: PoolGeometry):
        geometry.validate()
        self.geometry = geometry

    def encode(self, tracks: list[FusedTrack]) -> dict[str, list[list[float]]]:
        occupancy = self._plane()
        danger = self._plane()
        for track in tracks:
            column = min(
                int(track.pool_x / self.geometry.width_m * self.geometry.grid_width),
                self.geometry.grid_width - 1,
            )
            row = min(
                int(track.pool_y / self.geometry.height_m * self.geometry.grid_height),
                self.geometry.grid_height - 1,
            )
            occupancy[row][column] = min(1.0, occupancy[row][column] + track.confidence)
            danger[row][column] = max(danger[row][column], self._instant_risk(track))
        return {"occupancy": occupancy, "danger": danger}

    def _plane(self) -> list[list[float]]:
        return [
            [0.0 for _ in range(self.geometry.grid_width)]
            for _ in range(self.geometry.grid_height)
        ]

    @staticmethod
    def _instant_risk(track: FusedTrack) -> float:
        return min(
            1.0,
            0.45 * track.head_submerged
            + 0.25 * track.body_vertical
            + 0.2 * track.struggle
            + 0.1 * (1.0 - track.motion),
        )
