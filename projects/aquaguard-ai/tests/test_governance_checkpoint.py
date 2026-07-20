from datetime import timedelta

import pytest
from pydantic import ValidationError

from aquaguard.governance_checkpoint import (
    GovernanceAuditCheckpoint,
    GovernanceAuditCheckpointPublisher,
    GovernanceAuditCheckpointVerifier,
)
from aquaguard.governance_checkpoint_cli import main as checkpoint_main

from test_governance_audit import NOW, assemble_runtime, envelope


class InMemoryExternalCheckpointStore:
    def __init__(self) -> None:
        self._latest: dict[str, GovernanceAuditCheckpoint] = {}

    def latest(self, stream_id: str) -> GovernanceAuditCheckpoint | None:
        return self._latest.get(stream_id)

    def publish(
        self,
        checkpoint: GovernanceAuditCheckpoint,
        *,
        expected_previous_sha256: str | None,
    ) -> None:
        current = self.latest(checkpoint.stream_id)
        current_sha256 = None if current is None else current.sha256()
        if current_sha256 != expected_previous_sha256:
            raise RuntimeError("external checkpoint compare-and-swap conflict")
        self._latest[checkpoint.stream_id] = checkpoint


def runtime_in(path):
    path.mkdir()
    return assemble_runtime(path)


def record(runtime, artifact: bytes, hours: int) -> None:
    moment = NOW + timedelta(hours=hours)
    runtime.verifier.verify_and_record(
        artifact,
        envelope(artifact),
        runtime.trust_store,
        runtime.policy,
        checked_at=moment,
        recorded_at=moment,
    )


def test_publish_and_verify_chained_external_checkpoints(tmp_path) -> None:
    runtime, _ = assemble_runtime(tmp_path)
    store = InMemoryExternalCheckpointStore()
    publisher = GovernanceAuditCheckpointPublisher()
    record(runtime, b"first", 1)

    first = publisher.publish(
        runtime.audit_repository,
        store,
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=2),
    )
    assert first.checkpoint_number == 1
    assert first.audit_entries == 1
    assert first.previous_checkpoint_sha256 is None
    assert GovernanceAuditCheckpointVerifier().verify(
        runtime.audit_repository, first
    ).valid

    record(runtime, b"second", 3)
    second = publisher.publish(
        runtime.audit_repository,
        store,
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=4),
    )
    assert second.checkpoint_number == 2
    assert second.audit_entries == 2
    assert second.previous_checkpoint_sha256 == first.sha256()
    assert publisher.publish(
        runtime.audit_repository,
        store,
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=5),
    ) == second


def test_checkpoint_detects_rollback_and_cli_writes_report(tmp_path) -> None:
    complete_runtime, _ = runtime_in(tmp_path / "complete")
    record(complete_runtime, b"first", 1)
    record(complete_runtime, b"second", 2)
    checkpoint = GovernanceAuditCheckpointPublisher().publish(
        complete_runtime.audit_repository,
        InMemoryExternalCheckpointStore(),
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=3),
    )

    rolled_back_runtime, rolled_back_path = runtime_in(tmp_path / "rollback")
    record(rolled_back_runtime, b"first", 1)
    report = GovernanceAuditCheckpointVerifier().verify(
        rolled_back_runtime.audit_repository, checkpoint
    )
    assert report.valid is False
    assert "audit_rollback_or_truncation" in report.failures

    checkpoint_path = tmp_path / "checkpoint.json"
    report_path = tmp_path / "checkpoint-report.json"
    checkpoint_path.write_text(checkpoint.model_dump_json(), encoding="utf-8")
    assert checkpoint_main(
        [str(rolled_back_path), str(checkpoint_path), str(report_path)]
    ) == 2
    assert "audit_rollback_or_truncation" in report_path.read_text(encoding="utf-8")


def test_checkpoint_detects_alternate_valid_history(tmp_path) -> None:
    original, _ = runtime_in(tmp_path / "original")
    replacement, _ = runtime_in(tmp_path / "replacement")
    record(original, b"original", 1)
    record(replacement, b"replacement", 1)
    checkpoint = GovernanceAuditCheckpointPublisher().publish(
        original.audit_repository,
        InMemoryExternalCheckpointStore(),
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=2),
    )

    report = GovernanceAuditCheckpointVerifier().verify(
        replacement.audit_repository, checkpoint
    )
    assert report.valid is False
    assert "audit_history_diverged_from_checkpoint" in report.failures


def test_publisher_refuses_audit_without_external_history(tmp_path) -> None:
    original, _ = runtime_in(tmp_path / "original")
    replacement, _ = runtime_in(tmp_path / "replacement")
    store = InMemoryExternalCheckpointStore()
    record(original, b"original", 1)
    GovernanceAuditCheckpointPublisher().publish(
        original.audit_repository,
        store,
        stream_id="production-governance",
        anchored_at=NOW + timedelta(hours=2),
    )
    record(replacement, b"replacement", 1)

    with pytest.raises(RuntimeError, match="latest external checkpoint"):
        GovernanceAuditCheckpointPublisher().publish(
            replacement.audit_repository,
            store,
            stream_id="production-governance",
            anchored_at=NOW + timedelta(hours=3),
        )


def test_external_store_rejects_stale_compare_and_swap() -> None:
    store = InMemoryExternalCheckpointStore()
    checkpoint = GovernanceAuditCheckpoint(
        stream_id="production-governance",
        checkpoint_number=1,
        audit_entries=1,
        audit_head_sha256="a" * 64,
        anchored_at=NOW,
    )
    store.publish(checkpoint, expected_previous_sha256=None)
    with pytest.raises(RuntimeError, match="compare-and-swap"):
        store.publish(checkpoint, expected_previous_sha256=None)


def test_checkpoint_model_requires_chain_and_time_invariants() -> None:
    with pytest.raises(ValidationError, match="predecessor"):
        GovernanceAuditCheckpoint(
            stream_id="production-governance",
            checkpoint_number=2,
            audit_entries=1,
            audit_head_sha256="a" * 64,
            anchored_at=NOW,
        )
    with pytest.raises(ValidationError, match="timezone"):
        GovernanceAuditCheckpoint(
            stream_id="production-governance",
            checkpoint_number=1,
            audit_entries=1,
            audit_head_sha256="a" * 64,
            anchored_at=NOW.replace(tzinfo=None),
        )
