import pytest

from aquaguard.evidence import EvidenceClip, EvidenceRecorder, FrameRingBuffer
from aquaguard.video import VideoFrame


def frame(camera: str, sequence: int, timestamp: float) -> VideoFrame:
    return VideoFrame(camera, sequence, timestamp, f"pixels-{sequence}")


def test_ring_buffer_evicts_old_frames_and_isolates_cameras() -> None:
    buffer = FrameRingBuffer(retention_seconds=2, max_frames_per_camera=10)
    buffer.append(frame("cam-a", 0, 0))
    buffer.append(frame("cam-b", 0, 1))
    buffer.append(frame("cam-a", 1, 3))
    assert [item.sequence for item in buffer.between("cam-a", 0, 3)] == [1]
    assert [item.sequence for item in buffer.between("cam-b", 0, 3)] == [0]


def test_ring_buffer_rejects_out_of_order_timestamps() -> None:
    buffer = FrameRingBuffer()
    buffer.append(frame("cam-a", 0, 2))
    with pytest.raises(ValueError, match="monotonic"):
        buffer.append(frame("cam-a", 1, 1))


def test_recorder_locks_complete_pre_and_post_event_window() -> None:
    recorder = EvidenceRecorder(FrameRingBuffer(retention_seconds=20))
    for second in range(0, 6):
        recorder.ingest(frame("cam-a", second, second))
    recorder.request("event-1", "cam-a", trigger_at=5, pre_seconds=3, post_seconds=2)
    recorder.ingest(frame("cam-a", 6, 6))
    recorder.ingest(frame("cam-a", 7, 7))
    evidence = recorder.get("event-1")
    assert isinstance(evidence, EvidenceClip)
    assert evidence.complete is True
    assert [item.timestamp for item in evidence.frames] == [2, 3, 4, 5, 6, 7]


def test_force_finalize_marks_missing_post_event_evidence() -> None:
    recorder = EvidenceRecorder()
    recorder.ingest(frame("cam-a", 0, 4))
    recorder.ingest(frame("cam-a", 1, 5))
    recorder.request("event-1", "cam-a", trigger_at=5, pre_seconds=1, post_seconds=3)
    evidence = recorder.force_finalize("event-1")
    assert evidence.complete is False
    assert evidence.missing_post_event is True


def test_duplicate_event_evidence_is_rejected() -> None:
    recorder = EvidenceRecorder()
    recorder.request("event-1", "cam-a", trigger_at=5)
    with pytest.raises(ValueError, match="already exists"):
        recorder.request("event-1", "cam-a", trigger_at=6)
