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

## Retraining and qualification renewal

Retraining completion is a separate immutable artifact. It binds a pseudonymous reviewer and
independent assessor to the exact annotation protocol, training-material digest, evidence digest,
and timezone-aware completion time. Completion alone never renews qualification.

Renewal requires the previous qualification, retraining evidence, a complete new calibration
submission, qualification policy, and explicit renewal policy:

```bash
aquaguard-renew-reviewer-qualification \
  previous-qualification.json retraining-completion.json \
  calibration-set.json reviewer-submission.json qualification-policy.json renewal-policy.json \
  renewal-report.json renewed-qualification.json \
  --renewed-at 2026-10-20T12:00:00+00:00 \
  --expires-at 2027-01-20T12:00:00+00:00
```

The new record links the SHA-256 of both its predecessor and retraining completion, while the old
record remains unchanged. Reviewer and protocol identities must match throughout; retraining must
occur after the previous qualification and before renewal. The renewal policy explicitly decides
whether the same calibration set may be reused. A failed recalibration writes its report, returns
status `2`, and does not issue a new qualification.

These digests detect accidental or unauthorized content changes in controlled storage; they are
not digital signatures and do not establish the identity of an external signer.

Verify the complete append-only history before admitting work under its latest qualification:

```bash
aquaguard-verify-qualification-history \
  qualifications.jsonl retraining-completions.jsonl history-report.json \
  --checked-at 2027-01-01T12:00:00+00:00
```

The history must start with one unlinked initial qualification. Every later record must reference
the immediately preceding qualification and exactly one supplied retraining artifact. Reviewer,
protocol, and annotation version remain constant within one chain; qualification times are unique
and ascending; retraining occurs between the preceding qualification and renewal. Renewal validity
periods cannot overlap. Duplicate artifacts, missing links, orphan retraining evidence, identity
changes, invalid qualifications, and multiple active records are reported without mutating source
files. Status `2` means the chain is invalid. An expired but otherwise complete chain is valid with
no active qualification.

## Governance artifact signatures

Hash links detect content changes but do not authenticate a signer. Governance artifacts can be
wrapped in a detached `GovernanceSignatureEnvelope`. The canonical signing payload is domain
separated and binds artifact type and SHA-256, signer identity, exact key ID/version, algorithm,
and timezone-aware signing time. Private keys are never fields in these models.

Verify an envelope with an explicitly distributed trust store and revocation policy:

```bash
pip install -e '.[signing]'

aquaguard-verify-governance-signature \
  qualification.json signature-envelope.json trust-store.json signature-policy.json \
  signature-report.json --audit-database governance-signatures.sqlite3 \
  --checked-at 2027-01-01T12:00:00+00:00
```

The optional implementation verifies Ed25519 signatures. The core issuer accepts an external
signing port so production private keys can remain in an HSM, KMS, or controlled offline signer.
Key rotation uses an exact `(key_id, key_version)` lookup. The policy explicitly chooses whether
signatures made before a later revocation remain acceptable. Signing after revocation, signing
outside key validity, artifact changes, metadata changes, unknown keys, signer mismatch, future
signatures, and invalid signatures all fail closed.

The trust-store file is a root of trust, not self-authenticating input. Production deployment must
distribute and protect it through a separate trusted channel. This repository contains no private
keys, real public keys, certificates, organization identities, or claimed legal signatures.

The verification command requires explicit, existing trust-store and policy files and records
both successful and failed verifications in SQLite. Each audit entry binds the artifact, envelope,
trust-store and policy digests, full result, recording time, and previous entry digest. Writes use
a serialized transaction and revalidate the chain before append. A corrupt chain blocks normal
reads and future appends.

Audit integrity can be checked independently:

```bash
aquaguard-verify-governance-audit \
  governance-signatures.sqlite3 governance-audit-report.json
```

The internal chain detects payload changes, middle-row deletion/reordering, stored digest changes,
and predecessor mismatches. It cannot detect replacing the whole database with an older valid copy
or truncating its valid tail without an external checkpoint. Production must periodically anchor
the reported chain head in a separately controlled immutable service or signed transparency log.

## Remaining release gates

- approved real calibration material, qualification/admission thresholds, and renewal schedule;
- documented consent, retention, access, and deletion controls;
- scenario coverage and demographic/operational bias review;
- frozen recording hashes, calibration, model, configuration, and annotation versions;
- independent test-set review and supervised field validation.
