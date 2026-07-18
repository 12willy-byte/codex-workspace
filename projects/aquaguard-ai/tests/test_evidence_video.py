from pathlib import Path

import pytest

from aquaguard.evidence import EvidenceClip, FileEvidenceRepository, OpenCVMp4Encoder
from aquaguard.video import VideoFrame


class FakeImage:
    def __init__(self, height: int = 480, width: int = 640) -> None:
        self.shape = (height, width, 3)


class FakeWriter:
    def __init__(self, path: str, opened: bool = True) -> None:
        self.path = Path(path)
        self.opened = opened
        self.frames: list[object] = []
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, image: object) -> None:
        self.frames.append(image)

    def release(self) -> None:
        self.released = True
        if self.opened and self.frames:
            self.path.write_bytes(b"fake-mp4-payload")


def clip(images: list[object]) -> EvidenceClip:
    frames = tuple(VideoFrame("cam-a", index, float(index), image) for index, image in enumerate(images))
    return EvidenceClip("event-1", "cam-a", 0, 0, 1, frames, True, False, False)


def test_mp4_encoder_uses_consistent_dimensions_and_repository_integrity(tmp_path) -> None:
    writers: list[FakeWriter] = []

    def factory(path: str, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        assert codec == "mp4v"
        assert fps == 12
        assert size == (640, 480)
        writer = FakeWriter(path)
        writers.append(writer)
        return writer

    repository = FileEvidenceRepository(
        tmp_path, OpenCVMp4Encoder(fps=12, writer_factory=factory)
    )
    artifact = repository.persist(clip([FakeImage(), FakeImage()]))
    assert artifact.media_type == "video/mp4"
    assert artifact.path.suffix == ".mp4"
    assert artifact.frame_count == 2
    assert writers[0].released is True
    assert repository.verify(artifact) is True


def test_mp4_encoder_rejects_mixed_frame_dimensions(tmp_path) -> None:
    writer = FakeWriter(str(tmp_path / "placeholder.mp4"))
    encoder = OpenCVMp4Encoder(writer_factory=lambda *_: writer)
    with pytest.raises(ValueError, match="consistent dimensions"):
        encoder.encode(clip([FakeImage(), FakeImage(width=320)]), tmp_path / "video.mp4")
    assert writer.released is True


def test_mp4_encoder_rejects_non_pixel_frames(tmp_path) -> None:
    encoder = OpenCVMp4Encoder(writer_factory=lambda *_: FakeWriter("unused"))
    with pytest.raises(TypeError, match="shape"):
        encoder.encode(clip(["not-pixels"]), tmp_path / "video.mp4")


def test_mp4_encoder_reports_writer_open_failure(tmp_path) -> None:
    writer = FakeWriter(str(tmp_path / "video.mp4"), opened=False)
    encoder = OpenCVMp4Encoder(writer_factory=lambda *_: writer)
    with pytest.raises(RuntimeError, match="could not be opened"):
        encoder.encode(clip([FakeImage()]), tmp_path / "video.mp4")
    assert writer.released is True
