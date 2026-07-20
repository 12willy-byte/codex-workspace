# Annotation and Dataset Split Protocol

This protocol controls how binary benchmark labels are produced. It does not define a medical
diagnosis and must not be used to label real people without an approved operational definition,
reviewer training, privacy controls, and lawful data handling.

## Independent review

Each `(recording_id, timestamp, track_id)` item requires exactly two different pseudonymous
reviewers. Each reviewer chooses one judgment:

- `safe`
- `dangerous`
- `uncertain`

Reviewers must work independently and must not see the other review before submission. Reviewer
identifiers are audit identities, not names to embed in released datasets.

Two matching, definite judgments resolve automatically. Any disagreement, or any `uncertain`
judgment, requires an adjudication. The adjudicator must be different from both reviewers and must
record a final binary decision plus a reason. Redundant adjudication of an agreed item and
adjudication without a matching dispute are rejected.

The approved guide is stored as an `AnnotationProtocol` JSON artifact. It contains the temporal
context, observable criteria, known non-dangerous contexts, uncertainty triggers, occlusion and
identity policies, reviewer training requirements, privacy requirements, approver identifiers,
and effective date. Its canonical SHA-256 changes whenever the content changes.

Every review and adjudication must carry `protocol_id`, `annotation_version`, and
`protocol_sha256`. Resolution loads the actual protocol file, recomputes its digest, and rejects
records that do not match it.

```bash
aquaguard-resolve-annotations \
  annotation-protocol.json reviews.jsonl adjudications.jsonl \
  labels.jsonl resolution-summary.json
```

The generated `labels.jsonl` is accepted by `aquaguard-import-evaluation`. The summary preserves
recording, venue, session, reviewed-item count, adjudicated-item count, and resolved frames. Keep
the original review and adjudication files in the controlled audit store.

## Operational label definition

Before collecting real labels, the project owner must publish a versioned guide covering:

- observable criteria and minimum temporal context for `dangerous`;
- handling of normal diving, breath-hold training, floating, lessons, play, and rescue drills;
- visibility/occlusion rules and when reviewers must choose `uncertain`;
- track identity changes, disappearances, and cross-camera hand-off;
- escalation path for ambiguous or safety-critical footage.

Changing this guide requires a new `annotation_version`; datasets produced under incompatible
definitions must not be silently merged.

## Leakage-safe split

`VenueGroupedSplitter` assigns every recording from the same venue to one partition. Venues are
ordered by a stable seeded hash, then allocated to train, validation, and test according to the
configured proportions. At least three venues are required and every partition receives at least
one venue.

This is stricter than splitting individual clips or sessions and prevents pool geometry, camera
position, lighting, background, and recurring patrons from leaking across partitions. Dataset
release metadata must preserve the seed and assignments. Threshold tuning is allowed only on
training and validation data; the test partition stays sealed until final evaluation.

## Reviewer qualification

A `ReviewerCalibrationSet` is bound to one annotation protocol and has its own canonical SHA-256.
The set must contain approved dangerous and safe gold items. A submission must answer every item
exactly once and bind the same protocol and calibration-set digests.

Qualification policy thresholds are explicit project governance inputs, not defaults claimed by
this repository. The report separates:

- overall accuracy;
- dangerous recall;
- safe specificity;
- uncertain-response fraction;
- minimum total, dangerous, and safe item coverage.

```bash
aquaguard-qualify-reviewer \
  calibration-set.json reviewer-submission.json qualification-policy.json report.json \
  --record reviewer-qualification.json \
  --qualified-at 2026-07-19T12:00:00+00:00 \
  --expires-at 2026-10-19T12:00:00+00:00
```

The command exits with status `2` when the reviewer does not pass, while still writing the failure
report and every failed requirement. Only pseudonymous reviewer IDs belong in these artifacts.
When all lifecycle options are supplied and the evaluation passes, the command also issues an
immutable qualification record. The record binds the exact protocol, calibration set, policy,
and recomputed report digests. Both timestamps must include a timezone and the expiry boundary is
exclusive. Project owners choose the validity period; this repository does not claim a universal
retraining interval.

## Agreement monitoring

For batches completed by the same two reviewers, run:

```bash
aquaguard-annotation-agreement reviews.jsonl agreement-report.json
```

The report includes raw agreement, expected chance agreement, Cohen's kappa, disputes, and items
where either reviewer selected `uncertain`. Cohen's kappa is rejected when the batch does not use
the same two reviewers on every item; it must not be misreported for rotating reviewer pools.

## Annotation batch admission

A reviewed batch is not eligible for downstream resolution merely because two files exist. Before
acceptance, supply exactly one qualification record for each of the batch's two reviewers and an
explicit admission policy:

```bash
aquaguard-admit-annotation-batch \
  reviews.jsonl qualifications.jsonl batch-admission-policy.json admission-report.json \
  --checked-at 2026-07-20T12:00:00+00:00
```

The command exits with status `2` and still writes a denial report when either qualification is
missing, failed, not yet valid, expired, bound to another protocol, duplicated, or assigned to a
different reviewer set. It also denies batches below the project's explicit minimum item count,
observed agreement, or Cohen's kappa, and above its maximum uncertainty fraction. An undefined
kappa fails closed. No admission thresholds are embedded as claimed industry standards.

## Continuous quality monitoring

Archive each admission report as an `AnnotationBatchQualitySnapshot` with a unique batch ID and
timezone-aware completion time. Then evaluate a rolling window:

```bash
aquaguard-monitor-annotation-quality \
  batch-quality-snapshots.jsonl continuous-quality-policy.json trend-report.json \
  --evaluated-at 2026-08-01T12:00:00+00:00
```

The report computes item-weighted observed agreement, Cohen's kappa, uncertainty fraction,
changes from the preceding complete window, and the current run of denied batches. Its policy
explicitly controls history/window size, metric limits, denied-batch tolerance, and how many
simultaneous alerts recommend retraining. Missing history produces an
`insufficient_batch_history` alert without pretending a trend exists; unavailable kappa fails its
quality check. Exit status `2` means `retraining_required`, but the command never revokes a
reviewer or launches training automatically. Those remain audited human decisions.

All archived agreement reports are validated for consistent item counts, reviewer identities,
observed agreement, expected agreement, and kappa before monitoring.

## Remaining release gates

- approved real calibration material, qualification/admission thresholds, and renewal schedule;
- documented consent, retention, access, and deletion controls;
- scenario coverage and demographic/operational bias review;
- frozen recording hashes, calibration, model, configuration, and annotation versions;
- independent test-set review and supervised field validation.
