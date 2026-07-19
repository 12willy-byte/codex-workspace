# Offline Risk Benchmark

The benchmark evaluates recorded replay output against manually reviewed, per-track binary
risk labels. It is an engineering measurement harness, not evidence that the current model is
clinically or operationally validated.

## Manifest

Each manifest records a stable source identifier and strictly increasing frame timestamps.
Every visible track that matters to the evaluation should have a label at each frame.

```json
{
  "name": "pool-a scenario 001",
  "source": "sha256-or-controlled-recording-id",
  "frames": [
    {
      "timestamp": 0.0,
      "tracks": [{"track_id": "person-1", "dangerous": false}]
    },
    {
      "timestamp": 1.0,
      "tracks": [{"track_id": "person-1", "dangerous": true}]
    }
  ]
}
```

`RiskBenchmarkManifest.load()` and `.save()` validate the schema, reject duplicate track IDs,
and reject unordered or duplicate timestamps.

For a reproducible run, use a `ReplayEvaluationBundle`. It embeds the manifest labels alongside
pixel observations, camera homographies, pool geometry, source-recording SHA-256, and explicit
model, configuration, and annotation versions. The generated report records a canonical SHA-256
of the complete bundle.

```bash
aquaguard-benchmark evaluation-bundle.json benchmark-report.json
```

The command validates the entire bundle before replay. Missing camera calibration, timestamp
misalignment, malformed probabilities, and unsupported schema versions fail closed.

## Metrics

`BinaryRiskBenchmark.evaluate(replay_results, manifest)` reports:

- frame-level true/false positives and true/false negatives;
- precision, recall, and F1;
- dangerous episode count, detected/missed episode count, and episode recall;
- mean and maximum time from the first dangerous label to the first alarm in that episode.

An alarm for an unlabelled track is counted as a false positive. A labelled dangerous track
that is absent from replay output is treated as not alarmed. Metrics whose denominator is zero
are `null`, rather than a misleading zero or perfect score.

## Required evaluation discipline

- Keep training, tuning, and final test recordings separated by venue and recording session.
- Include normal diving, floating, coaching, crowding, glare, occlusion, night lighting, and
  stream degradation as hard negative cases.
- Record the model version, configuration, camera calibration, dataset digest, and benchmark
  report together.
- Do not choose thresholds on the final test set.
- Do not claim deployment readiness from synthetic replay tests. Commercial pilot gates require
  representative, independently reviewed pool data and supervised field trials.
