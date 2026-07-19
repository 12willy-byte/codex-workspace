from collections import Counter

from pydantic import BaseModel

from aquaguard.events import AlarmEventRepository
from aquaguard.evidence.service import EventEvidenceService


class EventEvidenceLink(BaseModel):
    event_id: str
    evidence_status: str


class EvidenceConsistencyReport(BaseModel):
    events: list[EventEvidenceLink]
    orphaned_evidence_ids: list[str]
    rejected_metadata: list[str]
    counts: dict[str, int]


class EvidenceConsistencyService:
    """Read-only recovery audit across event and evidence repositories."""

    def __init__(
        self,
        events: AlarmEventRepository,
        evidence: EventEvidenceService,
    ) -> None:
        self.events = events
        self.evidence = evidence

    def inspect(self) -> EvidenceConsistencyReport:
        event_ids = {str(event.id) for event in self.events.list()}
        links = []
        for event_id in sorted(event_ids):
            status = self.evidence.status(event_id)
            evidence_status = status["status"]
            if evidence_status == "stored" and status["integrity"] == "failed":
                evidence_status = "integrity_failed"
            links.append(
                EventEvidenceLink(
                    event_id=event_id,
                    evidence_status=evidence_status,
                )
            )
        evidence_ids = (
            set(self.evidence.artifacts)
            | set(self.evidence.recorder.pending)
            | set(self.evidence.recorder.completed)
        )
        orphaned = sorted(evidence_ids - event_ids)
        counts = Counter(link.evidence_status for link in links)
        counts["orphaned"] = len(orphaned)
        counts["rejected_metadata"] = len(self.evidence.recovery_errors)
        return EvidenceConsistencyReport(
            events=links,
            orphaned_evidence_ids=orphaned,
            rejected_metadata=list(self.evidence.recovery_errors),
            counts=dict(counts),
        )
