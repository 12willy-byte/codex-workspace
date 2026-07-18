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
        return self._persist(self.recorder.force_finalize(event_id))

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
