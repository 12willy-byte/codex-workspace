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
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class EvidenceRecovery:
    artifacts: tuple[StoredEvidence, ...]
    rejected_metadata: tuple[str, ...]


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
        metadata_path = self.root / f"{stem}.evidence.json"
        temporary_metadata = self.root / f".{stem}.evidence.json.tmp"
        stored_at = self.clock()
        try:
            self.encoder.encode(clip, temporary)
            digest = self._digest(temporary)
            temporary_checksum.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
            os.replace(temporary, destination)
            os.replace(temporary_checksum, checksum_path)
            metadata = {
                "schema_version": 1,
                "event_id": clip.event_id,
                "artifact_name": destination.name,
                "checksum_name": checksum_path.name,
                "sha256": digest,
                "media_type": self.encoder.media_type,
                "frame_count": len(clip.frames),
                "complete": clip.complete,
                "stored_at": stored_at,
            }
            temporary_metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
            temporary_checksum.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
        return StoredEvidence(
            event_id=clip.event_id,
            path=destination,
            checksum_path=checksum_path,
            sha256=digest,
            media_type=self.encoder.media_type,
            frame_count=len(clip.frames),
            complete=clip.complete,
            stored_at=stored_at,
            size_bytes=self._stored_size(destination, checksum_path, metadata_path),
            metadata_path=metadata_path,
        )

    def recover(self) -> EvidenceRecovery:
        artifacts: list[StoredEvidence] = []
        rejected: list[str] = []
        if not self.root.is_dir():
            return EvidenceRecovery((), ())
        for metadata_path in sorted(self.root.glob("*.evidence.json")):
            try:
                artifact = self._load_metadata(metadata_path)
                if not self.verify(artifact):
                    raise ValueError("integrity verification failed")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                rejected.append(f"{metadata_path.name}: {exc}")
                continue
            artifacts.append(artifact)
        return EvidenceRecovery(tuple(artifacts), tuple(rejected))

    def verify(self, stored: StoredEvidence) -> bool:
        if not stored.path.is_file() or not stored.checksum_path.is_file():
            return False
        expected = stored.checksum_path.read_text(encoding="ascii").split(maxsplit=1)[0]
        return expected == stored.sha256 == self._digest(stored.path)

    def delete(self, stored: StoredEvidence) -> bool:
        """Delete only the exact artifact and checksum registered in StoredEvidence."""
        if any(
            path.parent != self.root
            for path in (stored.path, stored.checksum_path, stored.metadata_path)
        ):
            raise ValueError("Evidence artifact is outside the repository root")
        existed = any(
            path.exists() for path in (stored.path, stored.checksum_path, stored.metadata_path)
        )
        stored.path.unlink(missing_ok=True)
        stored.checksum_path.unlink(missing_ok=True)
        stored.metadata_path.unlink(missing_ok=True)
        return existed

    def _load_metadata(self, metadata_path: Path) -> StoredEvidence:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported schema version")
        event_id = metadata["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("invalid event id")
        expected_stem = hashlib.sha256(event_id.encode()).hexdigest()[:24]
        if metadata_path.name != f"{expected_stem}.evidence.json":
            raise ValueError("metadata filename does not match event id")
        artifact_path = self._child_path(metadata["artifact_name"])
        checksum_path = self._child_path(metadata["checksum_name"])
        if checksum_path.name != f"{artifact_path.name}.sha256":
            raise ValueError("checksum filename does not match artifact")
        frame_count = metadata["frame_count"]
        complete = metadata["complete"]
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 0:
            raise ValueError("invalid frame count")
        if not isinstance(complete, bool):
            raise ValueError("invalid completeness flag")
        return StoredEvidence(
            event_id=event_id,
            path=artifact_path,
            checksum_path=checksum_path,
            sha256=str(metadata["sha256"]),
            media_type=str(metadata["media_type"]),
            frame_count=frame_count,
            complete=complete,
            stored_at=float(metadata["stored_at"]),
            size_bytes=self._stored_size(artifact_path, checksum_path, metadata_path),
            metadata_path=metadata_path,
        )

    def _child_path(self, name: object) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("invalid evidence filename")
        path = self.root / name
        if path.parent != self.root:
            raise ValueError("evidence path is outside repository root")
        return path

    @staticmethod
    def _stored_size(*paths: Path) -> int:
        return sum(path.stat().st_size for path in paths)

    @staticmethod
    def _digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
