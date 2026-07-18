import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

from aquaguard.evidence.models import EvidenceClip


class EvidenceEncoder(Protocol):
    media_type: str
    extension: str

    def encode(self, clip: EvidenceClip, destination: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    event_id: str
    path: Path
    checksum_path: Path
    sha256: str
    media_type: str
    frame_count: int
    complete: bool
    stored_at: float
    size_bytes: int


class JsonEvidenceManifestEncoder:
    """Audit manifest baseline; it intentionally does not encode frame pixels or video."""

    media_type = "application/json"
    extension = ".json"

    def encode(self, clip: EvidenceClip, destination: Path) -> None:
        payload = {
            "schema_version": 1,
            "event_id": clip.event_id,
            "camera_id": clip.camera_id,
            "trigger_at": clip.trigger_at,
            "starts_at": clip.starts_at,
            "ends_at": clip.ends_at,
            "complete": clip.complete,
            "missing_pre_event": clip.missing_pre_event,
            "missing_post_event": clip.missing_post_event,
            "frames": [
                {
                    "camera_id": frame.camera_id,
                    "sequence": frame.sequence,
                    "timestamp": frame.timestamp,
                    "image_type": type(frame.image).__name__,
                }
                for frame in clip.frames
            ],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )


class FileEvidenceRepository:
    def __init__(
        self,
        root: Path,
        encoder: EvidenceEncoder | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.root = root
        self.encoder = encoder or JsonEvidenceManifestEncoder()
        self.clock = clock

    def persist(self, clip: EvidenceClip) -> StoredEvidence:
        self.root.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha256(clip.event_id.encode()).hexdigest()[:24]
        destination = self.root / f"{stem}{self.encoder.extension}"
        temporary = self.root / f".{stem}.tmp{self.encoder.extension}"
        checksum_path = destination.with_suffix(destination.suffix + ".sha256")
        temporary_checksum = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
        try:
            self.encoder.encode(clip, temporary)
            digest = self._digest(temporary)
            temporary_checksum.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
            os.replace(temporary, destination)
            os.replace(temporary_checksum, checksum_path)
        finally:
            temporary.unlink(missing_ok=True)
            temporary_checksum.unlink(missing_ok=True)
        return StoredEvidence(
            clip.event_id,
            destination,
            checksum_path,
            digest,
            self.encoder.media_type,
            len(clip.frames),
            clip.complete,
            self.clock(),
            destination.stat().st_size + checksum_path.stat().st_size,
        )

    def verify(self, stored: StoredEvidence) -> bool:
        if not stored.path.is_file() or not stored.checksum_path.is_file():
            return False
        expected = stored.checksum_path.read_text(encoding="ascii").split(maxsplit=1)[0]
        return expected == stored.sha256 == self._digest(stored.path)

    def delete(self, stored: StoredEvidence) -> bool:
        """Delete only the exact artifact and checksum registered in StoredEvidence."""
        if stored.path.parent != self.root or stored.checksum_path.parent != self.root:
            raise ValueError("Evidence artifact is outside the repository root")
        existed = stored.path.exists() or stored.checksum_path.exists()
        stored.path.unlink(missing_ok=True)
        stored.checksum_path.unlink(missing_ok=True)
        return existed

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
