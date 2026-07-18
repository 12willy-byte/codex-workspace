from pathlib import Path
from typing import Protocol

from aquaguard.evidence.models import EvidenceClip


class VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, image: object) -> None: ...

    def release(self) -> None: ...


class WriterFactory(Protocol):
    def __call__(
        self, path: str, codec: str, fps: float, size: tuple[int, int]
    ) -> VideoWriter: ...


def _opencv_writer(path: str, codec: str, fps: float, size: tuple[int, int]) -> VideoWriter:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the 'vision' dependencies to encode evidence video") from exc
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)


class OpenCVMp4Encoder:
    """Encode pixel-bearing evidence frames into a constant-frame-rate MP4 artifact."""

    media_type = "video/mp4"
    extension = ".mp4"

    def __init__(
        self,
        fps: float = 10.0,
        codec: str = "mp4v",
        writer_factory: WriterFactory | None = None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(codec) != 4:
            raise ValueError("codec must contain four characters")
        self.fps = fps
        self.codec = codec
        self.writer_factory = writer_factory or _opencv_writer

    def encode(self, clip: EvidenceClip, destination: Path) -> None:
        if not clip.frames:
            raise ValueError("Cannot encode evidence without frames")
        size = self._frame_size(clip.frames[0].image)
        writer = self.writer_factory(str(destination), self.codec, self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Evidence video writer could not be opened")
        try:
            for frame in clip.frames:
                if self._frame_size(frame.image) != size:
                    raise ValueError("Evidence frames must have consistent dimensions")
                writer.write(frame.image)
        finally:
            writer.release()
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("Evidence encoder did not produce a video file")

    @staticmethod
    def _frame_size(image: object) -> tuple[int, int]:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            raise TypeError("Evidence frame image must expose a height/width shape")
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            raise ValueError("Evidence frame dimensions must be positive")
        return width, height
