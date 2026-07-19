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
  calibration-set.json reviewer-submission.json qualification-policy.json report.json
```

The command exits with status `2` when the reviewer does not pass, while still writing the failure
report and every failed requirement. Only pseudonymous reviewer IDs belong in these artifacts.

## Agreement monitoring

For batches completed by the same two reviewers, run:

```bash
aquaguard-annotation-agreement reviews.jsonl agreement-report.json
```

The report includes raw agreement, expected chance agreement, Cohen's kappa, disputes, and items
where either reviewer selected `uncertain`. Cohen's kappa is rejected when the batch does not use
the same two reviewers on every item; it must not be misreported for rotating reviewer pools.

## Remaining release gates

- reviewer qualification and agreement statistics;
- documented consent, retention, access, and deletion controls;
- scenario coverage and demographic/operational bias review;
- frozen recording hashes, calibration, model, configuration, and annotation versions;
- independent test-set review and supervised field validation.
