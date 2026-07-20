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
        if len(set(self.points)) != len(self.points):
            raise ValueError("Polygon coordinates must be unique")
        if self._has_self_intersection():
            raise ValueError("Polygon region must not self-intersect")
        if abs(self.signed_area()) < 1e-9:
            raise ValueError("Polygon region must have non-zero area")

    def signed_area(self) -> float:
        return 0.5 * sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                self.points, self.points[1:] + self.points[:1], strict=True
            )
        )

    def area(self) -> float:
        return abs(self.signed_area())

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

    def _has_self_intersection(self) -> bool:
        edges = list(zip(self.points, self.points[1:] + self.points[:1], strict=True))
        for first_index, first in enumerate(edges):
            for second_index in range(first_index + 1, len(edges)):
                if second_index in {
                    first_index,
                    (first_index + 1) % len(edges),
                    (first_index - 1) % len(edges),
                }:
                    continue
                if self._segments_intersect(first, edges[second_index]):
                    return True
        return False

    @staticmethod
    def _segments_intersect(
        first: tuple[tuple[float, float], tuple[float, float]],
        second: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        def orientation(
            a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
        ) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        a, b = first
        c, d = second
        values = (
            orientation(a, b, c),
            orientation(a, b, d),
            orientation(c, d, a),
            orientation(c, d, b),
        )
        if values[0] * values[1] < 0 and values[2] * values[3] < 0:
            return True
        return any(
            abs(value) <= 1e-9 and PolygonRegion._on_segment(start, end, point)
            for value, start, end, point in (
                (values[0], a, b, c),
                (values[1], a, b, d),
                (values[2], c, d, a),
                (values[3], c, d, b),
            )
        )
