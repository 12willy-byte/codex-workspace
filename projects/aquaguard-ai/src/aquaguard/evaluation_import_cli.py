import argparse
from pathlib import Path

from aquaguard.vision.evaluation_import import ReplayEvaluationImporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aquaguard-import-evaluation",
        description="Build a validated replay evaluation bundle from controlled source files.",
    )
    parser.add_argument("source_recording", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--configuration-version", required=True)
    parser.add_argument("--annotation-version", required=True)
    parser.add_argument("--expected-source-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = ReplayEvaluationImporter().build(
        name=args.name,
        source_recording_path=args.source_recording,
        observations_path=args.observations,
        labels_path=args.labels,
        calibration_path=args.calibration,
        model_version=args.model_version,
        configuration_version=args.configuration_version,
        annotation_version=args.annotation_version,
        expected_source_sha256=args.expected_source_sha256,
    )
    bundle.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
