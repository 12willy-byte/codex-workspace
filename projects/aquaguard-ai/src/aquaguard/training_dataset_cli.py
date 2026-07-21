from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from aquaguard.vision.training_dataset import (
    BehaviorEpisode,
    ClipPlanningPolicy,
    TemporalTrainingDatasetBuilder,
    TrainingRecordingSource,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a leakage-safe temporal training clip plan")
    parser.add_argument("recordings", type=Path)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split-seed", required=True)
    parser.add_argument("--created-at", required=True, type=datetime.fromisoformat)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = TemporalTrainingDatasetBuilder().build(
        dataset_name=args.name,
        recordings=_load_jsonl(args.recordings, TrainingRecordingSource),
        episodes=_load_jsonl(args.episodes, BehaviorEpisode),
        policy=ClipPlanningPolicy.model_validate_json(args.policy.read_text(encoding="utf-8")),
        split_seed=args.split_seed,
        created_at=args.created_at,
    )
    manifest.save(args.output)
    print(json.dumps({"sha256": manifest.sha256(), **manifest.summary()}, sort_keys=True))


if __name__ == "__main__":
    main()
