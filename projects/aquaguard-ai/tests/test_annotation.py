import json
from datetime import date

import pytest

from aquaguard.annotation_cli import main as annotation_main
from aquaguard.vision import (
    AnnotationAdjudication,
    AnnotationProtocol,
    AnnotationReview,
    DatasetRecording,
    DualReviewResolver,
    RiskJudgment,
    VenueGroupedSplitter,
)

PROTOCOL_SHA256 = "a" * 64


def annotation_protocol() -> AnnotationProtocol:
    return AnnotationProtocol(
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        effective_date=date(2026, 7, 19),
        language="zh-CN",
        temporal_context_before_seconds=10,
        temporal_context_after_seconds=5,
        dangerous_observable_criteria=("Approved observable criterion",),
        known_non_dangerous_contexts=("Approved normal-context example",),
        uncertainty_triggers=("Critical body landmarks are occluded",),
        occlusion_policy="Choose uncertain when required evidence is not visible.",
        identity_discontinuity_policy="End the item when track identity changes.",
        reviewer_training_requirements=("Pass the controlled calibration set",),
        privacy_requirements=("Use pseudonymous reviewer and track identifiers",),
        approved_by=("safety-review-board",),
    )


def review(
    reviewer_id: str,
    judgment: RiskJudgment,
    *,
    timestamp: float = 1,
    track_id: str = "person-1",
) -> AnnotationReview:
    return AnnotationReview(
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        recording_id="recording-1",
        venue_id="venue-a",
        session_id="session-1",
        timestamp=timestamp,
        track_id=track_id,
        reviewer_id=reviewer_id,
        judgment=judgment,
    )


def adjudication(dangerous: bool, adjudicator_id: str = "reviewer-c") -> AnnotationAdjudication:
    return AnnotationAdjudication(
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        recording_id="recording-1",
        timestamp=1,
        track_id="person-1",
        adjudicator_id=adjudicator_id,
        dangerous=dangerous,
        reason="Reviewed the full temporal context.",
    )


def test_two_independent_agreeing_reviews_resolve_without_adjudication() -> None:
    resolved = DualReviewResolver().resolve(
        [
            review("reviewer-a", RiskJudgment.DANGEROUS),
            review("reviewer-b", RiskJudgment.DANGEROUS),
        ],
        [],
    )

    assert resolved.reviewed_items == 1
    assert resolved.protocol_sha256 == PROTOCOL_SHA256
    assert resolved.adjudicated_items == 0
    assert resolved.frames[0].tracks[0].dangerous is True


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (RiskJudgment.SAFE, RiskJudgment.DANGEROUS),
        (RiskJudgment.UNCERTAIN, RiskJudgment.UNCERTAIN),
        (RiskJudgment.UNCERTAIN, RiskJudgment.SAFE),
    ],
)
def test_disagreement_or_uncertainty_requires_independent_adjudication(first, second) -> None:
    reviews = [review("reviewer-a", first), review("reviewer-b", second)]

    with pytest.raises(ValueError, match="requires adjudication"):
        DualReviewResolver().resolve(reviews, [])

    resolved = DualReviewResolver().resolve(reviews, [adjudication(False)])

    assert resolved.frames[0].tracks[0].dangerous is False
    assert resolved.adjudicated_items == 1


def test_reviewer_cannot_adjudicate_the_same_item() -> None:
    reviews = [
        review("reviewer-a", RiskJudgment.SAFE),
        review("reviewer-b", RiskJudgment.DANGEROUS),
    ]

    with pytest.raises(ValueError, match="independent of both reviewers"):
        DualReviewResolver().resolve(reviews, [adjudication(True, "reviewer-a")])


def test_exactly_two_unique_reviewers_are_required() -> None:
    with pytest.raises(ValueError, match="exactly two independent"):
        DualReviewResolver().resolve(
            [
                review("reviewer-a", RiskJudgment.SAFE),
                review("reviewer-a", RiskJudgment.SAFE),
            ],
            [],
        )


def test_redundant_or_orphaned_adjudication_is_rejected() -> None:
    agreed = [
        review("reviewer-a", RiskJudgment.SAFE),
        review("reviewer-b", RiskJudgment.SAFE),
    ]
    with pytest.raises(ValueError, match="must not have redundant"):
        DualReviewResolver().resolve(agreed, [adjudication(False)])

    orphan = adjudication(False).model_copy(update={"track_id": "person-2"})
    with pytest.raises(ValueError, match="requires adjudication"):
        DualReviewResolver().resolve(
            [
                review("reviewer-a", RiskJudgment.SAFE),
                review("reviewer-b", RiskJudgment.DANGEROUS),
            ],
            [orphan],
        )


def test_reviews_and_adjudications_must_use_one_protocol() -> None:
    reviews = [
        review("reviewer-a", RiskJudgment.SAFE),
        review("reviewer-b", RiskJudgment.DANGEROUS),
    ]
    mismatched = adjudication(True).model_copy(update={"protocol_sha256": "b" * 64})

    with pytest.raises(ValueError, match="must use one annotation protocol"):
        DualReviewResolver().resolve(reviews, [mismatched])


def test_annotation_protocol_has_stable_digest_and_requires_real_content(tmp_path) -> None:
    protocol = annotation_protocol()
    path = tmp_path / "protocol.json"
    protocol.save(path)

    assert AnnotationProtocol.load(path) == protocol
    assert AnnotationProtocol.load(path).sha256() == protocol.sha256()
    changed = protocol.model_copy(
        update={"occlusion_policy": "A separately approved changed policy."}
    )
    assert changed.sha256() != protocol.sha256()

    with pytest.raises(ValueError, match="non-zero temporal context"):
        AnnotationProtocol.model_validate(
            {
                **protocol.model_dump(),
                "temporal_context_before_seconds": 0,
                "temporal_context_after_seconds": 0,
            }
        )


def test_venue_grouped_split_is_deterministic_and_prevents_cross_split_leakage() -> None:
    recordings = [
        DatasetRecording(recording_id="a-1", venue_id="venue-a", session_id="morning"),
        DatasetRecording(recording_id="a-2", venue_id="venue-a", session_id="evening"),
        DatasetRecording(recording_id="b-1", venue_id="venue-b", session_id="morning"),
        DatasetRecording(recording_id="c-1", venue_id="venue-c", session_id="morning"),
    ]
    splitter = VenueGroupedSplitter("dataset-release-v1")

    first = splitter.assign(recordings)
    second = splitter.assign(list(reversed(recordings)))

    assert first == second
    venue_a_splits = {item.split for item in first if item.venue_id == "venue-a"}
    assert len(venue_a_splits) == 1
    assert {item.split for item in first} == {"train", "validation", "test"}


def test_splitter_rejects_duplicate_recording_ids_and_invalid_fractions() -> None:
    duplicate = DatasetRecording(
        recording_id="recording-1", venue_id="venue-a", session_id="session-1"
    )
    with pytest.raises(ValueError, match="recording ids must be unique"):
        VenueGroupedSplitter("seed").assign([duplicate, duplicate])

    with pytest.raises(ValueError, match="at least three venues"):
        VenueGroupedSplitter("seed").assign([duplicate])

    with pytest.raises(ValueError, match="leave a test partition"):
        VenueGroupedSplitter("seed", train_fraction=0.8, validation_fraction=0.2)


def test_annotation_command_writes_importable_labels_and_resolution_summary(tmp_path) -> None:
    protocol_path = tmp_path / "protocol.json"
    reviews_path = tmp_path / "reviews.jsonl"
    adjudications_path = tmp_path / "adjudications.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    summary_path = tmp_path / "summary.json"
    reviews = [
        review("reviewer-a", RiskJudgment.SAFE),
        review("reviewer-b", RiskJudgment.DANGEROUS),
    ]
    protocol = annotation_protocol()
    protocol.save(protocol_path)
    reviews = [item.model_copy(update={"protocol_sha256": protocol.sha256()}) for item in reviews]
    reviews_path.write_text(
        "\n".join(item.model_dump_json() for item in reviews) + "\n", encoding="utf-8"
    )
    resolved_adjudication = adjudication(True).model_copy(
        update={"protocol_sha256": protocol.sha256()}
    )
    adjudications_path.write_text(resolved_adjudication.model_dump_json() + "\n", encoding="utf-8")

    result = annotation_main(
        [
            str(protocol_path),
            str(reviews_path),
            str(adjudications_path),
            str(labels_path),
            str(summary_path),
        ]
    )

    assert result == 0
    label = json.loads(labels_path.read_text().strip())
    summary = json.loads(summary_path.read_text())
    assert label["tracks"][0] == {"track_id": "person-1", "dangerous": True}
    assert summary["adjudicated_items"] == 1
