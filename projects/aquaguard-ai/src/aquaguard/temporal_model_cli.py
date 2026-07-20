import argparse
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.world.learned import (
    TemporalModelPackageVerifier,
    TemporalRiskModelManifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-temporal-model",
        description="Verify a learned temporal-risk model artifact and traceable manifest.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model_artifact", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = TemporalRiskModelManifest.load(args.manifest)
        report = TemporalModelPackageVerifier().verify(manifest, args.model_artifact)
    except (OSError, ValueError) as exc:
        _write_json(args.report, {"valid": False, "failures": [str(exc)]})
        return 2
    _write_json(args.report, report.model_dump(mode="json"))
    return 0
