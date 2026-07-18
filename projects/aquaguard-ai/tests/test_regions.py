import math

import pytest

from aquaguard.vision import PolygonRegion


def test_polygon_contains_interior_boundary_and_exterior_points() -> None:
    region = PolygonRegion(((0, 0), (10, 0), (10, 10), (0, 10)))

    assert region.contains(5, 5) is True
    assert region.contains(0, 5) is True
    assert region.contains(12, 5) is False


def test_polygon_requires_valid_finite_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        PolygonRegion(((0, 0), (1, 1)))
    with pytest.raises(ValueError, match="finite"):
        PolygonRegion(((0, 0), (1, 0), (math.inf, 1)))
