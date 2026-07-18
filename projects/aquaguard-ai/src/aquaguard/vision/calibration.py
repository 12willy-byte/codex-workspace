from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HomographyProjector:
    """Project image anchors into pool coordinates with a calibrated 3x3 homography."""

    camera_id: str
    matrix: tuple[float, float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id is required")
        if len(self.matrix) != 9:
            raise ValueError("Homography matrix must contain nine values")

    def project(self, image_x: float, image_y: float) -> tuple[float, float]:
        h = self.matrix
        scale = h[6] * image_x + h[7] * image_y + h[8]
        if abs(scale) < 1e-9:
            raise ValueError("Point projects to infinity; calibration is invalid for this point")
        pool_x = (h[0] * image_x + h[1] * image_y + h[2]) / scale
        pool_y = (h[3] * image_x + h[4] * image_y + h[5]) / scale
        return pool_x, pool_y
