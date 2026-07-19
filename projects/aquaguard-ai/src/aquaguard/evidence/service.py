from aquaguard.domain import AlarmEvent
from aquaguard.evidence.models import EvidenceClip, EvidenceWindow
from aquaguard.evidence.recorder import EvidenceRecorder
from aquaguard.evidence.retention import CleanupPlan, EvidenceRetentionPolicy
from aquaguard.evidence.storage import FileEvidenceRepository, StoredEvidence


class EventEvidenceService:
    """Coordinate alarm events, frame windows and durable evidence artifacts."""

    def __init__(self, recorder: EvidenceRecorder, repository: FileEvidenceRepository):
        self.recorder = recorder
        self.repository = repository
        self.artifacts: dict[str, StoredEvidence] = {}
        self.recovery_errors: tuple[str, ...] = ()

    def recover(self) -> tuple[StoredEvidence, ...]:
        recovery = self.repository.recover()
        self.artifacts = {artifact.event_id: artifact for artifact in recovery.artifacts}
        self.recovery_errors = recovery.rejected_metadata
        return recovery.artifacts

    def status(self, event_id: str) -> dict:
        artifact = self.artifacts.get(event_id)
        if artifact is not None:
            return {
                "event_id": event_id,
                "status": "stored",
                "media_type": artifact.media_type,
                "frame_count": artifact.frame_count,
                "complete": artifact.complete,
                "stored_at": artifact.stored_at,
                "size_bytes": artifact.size_bytes,
                "integrity": "verified" if self.repository.verify(artifact) else "failed",
            }
        window = self.recorder.pending.get(event_id)
        if window is not None:
            return {
                "event_id": event_id,
                "status": "pending",
                "camera_id": window.camera_id,
                "starts_at": window.starts_at,
                "ends_at": window.ends_at,
            }
        clip = self.recorder.completed.get(event_id)
        if clip is not None:
            return {
                "event_id": event_id,
                "status": "ready",
                "camera_id": clip.camera_id,
                "frame_count": len(clip.frames),
                "complete": clip.complete,
            }
        return {"event_id": event_id, "status": "missing"}

    def request(
        self,
        event: AlarmEvent,
        trigger_timestamp: float,
        *,
        pre_seconds: float = 30.0,
        post_seconds: float = 60.0,
    ) -> EvidenceWindow:
        return self.recorder.request(
            str(event.id),
            event.camera_id,
            trigger_timestamp,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )

    def persist_ready(self, now: float, camera_id: str | None = None) -> list[StoredEvidence]:
        clips = self.recorder.finalize_ready(now, camera_id)
        return [self._persist(clip) for clip in clips]

    def force_persist(self, event_id: str) -> StoredEvidence:
        clip = self.recorder.completed.get(event_id)
        if clip is None:
            clip = self.recorder.force_finalize(event_id)
        return self._persist(clip)

    def cleanup(self, policy: EvidenceRetentionPolicy, now: float) -> CleanupPlan:
        plan = policy.plan(list(self.artifacts.values()), now)
        for event_id in plan.remove_event_ids:
            artifact = self.artifacts.pop(event_id)
            self.repository.delete(artifact)
        return plan

    def _persist(self, clip: EvidenceClip) -> StoredEvidence:
        artifact = self.repository.persist(clip)
        self.artifacts[clip.event_id] = artifact
        return artifact
