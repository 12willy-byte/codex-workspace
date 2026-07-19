import argparse
import json
from pathlib import Path

from aquaguard.vision.annotation import (
    AnnotationAdjudication,
    AnnotationProtocol,
    AnnotationReview,
    DualReviewResolver,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aquaguard-resolve-annotations",
        description="Resolve independent reviews into benchmark labels.",
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("reviews", type=Path)
    parser.add_argument("adjudications", type=Path)
    parser.add_argument("labels_output", type=Path)
    parser.add_argument("summary_output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = AnnotationProtocol.load(args.protocol)
    reviews = _load_jsonl(args.reviews, AnnotationReview)
    adjudications = _load_jsonl(args.adjudications, AnnotationAdjudication)
    resolved = DualReviewResolver().resolve(reviews, adjudications)
    if (
        resolved.protocol_id != protocol.protocol_id
        or resolved.annotation_version != protocol.annotation_version
        or resolved.protocol_sha256 != protocol.sha256()
    ):
        raise ValueError("resolved annotations do not match the supplied protocol artifact")
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.write_text(
        "\n".join(frame.model_dump_json() for frame in resolved.frames) + "\n",
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(resolved.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _load_jsonl(path: Path, model: type):
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            records.append(model.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return records


if __name__ == "__main__":
    raise SystemExit(main())
