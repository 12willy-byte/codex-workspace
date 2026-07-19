import argparse
from pathlib import Path

from aquaguard.vision.evaluation import ReplayEvaluationBundle, ReplayEvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aquaguard-benchmark",
        description="Validate and evaluate an AquaGuard replay bundle.",
    )
    parser.add_argument("bundle", type=Path, help="Path to the replay evaluation bundle JSON")
    parser.add_argument("report", type=Path, help="Path for the generated benchmark report JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = ReplayEvaluationBundle.load(args.bundle)
    result = ReplayEvaluationRunner().run(bundle)
    result.save(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
