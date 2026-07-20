# Learned Temporal Risk Model Contract

The learned temporal model is an optional replacement for the deterministic forecasting baseline.
This repository defines its input/output and safety contract; it does not contain a trained or
validated drowning model.

## Fixed feature sequence

`aquaguard.temporal_features.v1` supplies one ordered vector per fused track sample:

1. elapsed seconds from the first selected sample;
2. pool-space X/Y in meters;
3. fused track confidence;
4. head-submerged feature;
5. body verticality;
6. struggle feature;
7. motion;
8. occlusion;
9. head-in-water-ROI relation and its confidence;
10. wrist motion and its confidence.

The feature order is exported as `TEMPORAL_FEATURE_SCHEMA_V1`. A model must not infer a different
order from dictionary iteration or undocumented preprocessing.

## Model package

`TemporalRiskModelManifest` binds the exact model SHA-256, model version, feature schema, minimum
and maximum sequence samples, minimum time span, and maximum allowed sample gap. If the manifest
claims assistive-alerting validation, it must also bind the evaluation bundle, validation report,
and approval artifact digests.

```bash
aquaguard-verify-temporal-model \
  temporal-model-manifest.json temporal-model.bin temporal-model-report.json
```

This command validates package consistency only. A manifest's validation claim is not
self-authenticating. Runtime alert eligibility additionally requires `approval_verified=True`
from trusted composition after the approval artifact and its governance signature have been
verified. A local JSON edit is therefore insufficient to enable learned-model alerting.

## Failure behavior

`ValidatedTemporalRiskPredictor` returns an unavailable forecast with uncertainty `1.0` when:

- samples are insufficient, timestamps do not strictly increase, duration is too short, or a gap
  exceeds the manifest limit;
- one history contains multiple track identities or non-finite timestamps;
- inference raises an exception;
- output risk, time-to-critical, uncertainty, or reasons violate the output schema.

An unavailable or unapproved learned forecast cannot create a prediction-only alarm. The
independent deterministic supervisor still evaluates raw severe signals so model failure does not
remove that separate guardrail. Production alarm publication remains subject to the camera
protection-level gate.

## Remaining work

- collect controlled, consented and independently annotated multi-site sequences;
- train candidate temporal architectures and calibrate uncertainty;
- freeze preprocessing and feature normalization with the model package;
- evaluate scenario-level recall, false alarms, latency, missing tracks and calibration drift;
- independently approve and sign the model approval artifact before supervised field use.

Until those gates pass, the deterministic baseline remains the default and no accuracy claim is
made for learned drowning recognition.
