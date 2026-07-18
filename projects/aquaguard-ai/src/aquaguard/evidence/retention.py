from dataclasses import dataclass

from aquaguard.evidence.storage import StoredEvidence


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    remove_event_ids: tuple[str, ...]
    retained_event_ids: tuple[str, ...]
    retained_bytes: int


class EvidenceRetentionPolicy:
    """Plan expiry and capacity cleanup without deleting files itself."""

    def __init__(self, max_bytes: int, max_age_seconds: float):
        if max_bytes < 0 or max_age_seconds < 0:
            raise ValueError("Retention limits must not be negative")
        self.max_bytes = max_bytes
        self.max_age_seconds = max_age_seconds

    def plan(self, artifacts: list[StoredEvidence], now: float) -> CleanupPlan:
        ordered = sorted(artifacts, key=lambda item: (item.stored_at, item.event_id))
        remove: list[StoredEvidence] = []
        retained: list[StoredEvidence] = []
        for artifact in ordered:
            if now - artifact.stored_at >= self.max_age_seconds:
                remove.append(artifact)
            else:
                retained.append(artifact)

        retained_bytes = sum(item.size_bytes for item in retained)
        while retained and retained_bytes > self.max_bytes:
            oldest = retained.pop(0)
            remove.append(oldest)
            retained_bytes -= oldest.size_bytes

        return CleanupPlan(
            tuple(item.event_id for item in remove),
            tuple(item.event_id for item in retained),
            retained_bytes,
        )
