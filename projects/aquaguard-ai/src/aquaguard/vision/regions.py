import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolygonRegion:
    """Calibrated image-space polygon with boundary-inclusive point containment."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("Polygon region requires at least three points")
        if any(not math.isfinite(value) for point in self.points for value in point):
            raise ValueError("Polygon coordinates must be finite")

    def contains(self, x: float, y: float) -> bool:
        inside = False
        previous = self.points[-1]
        for current in self.points:
            if self._on_segment(previous, current, (x, y)):
                return True
            x1, y1 = previous
            x2, y2 = current
            crosses = (y1 > y) != (y2 > y)
            if crosses:
                intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < intersection_x:
                    inside = not inside
            previous = current
        return inside

    @staticmethod
    def _on_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        x1, y1 = start
        x2, y2 = end
        x, y = point
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) > 1e-9:
            return False
        return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(
            y1, y2
        ) - 1e-9 <= y <= max(y1, y2) + 1e-9
