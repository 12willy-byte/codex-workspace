from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class TrackRiskLabel(BaseModel):
    track_id: str = Field(min_length=1)
    dangerous: bool

    @field_validator("track_id")
    @classmethod
    def normalize_track_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("track_id must not be blank")
        return value


class FrameRiskLabels(BaseModel):
    timestamp: float = Field(ge=0)
    tracks: tuple[TrackRiskLabel, ...]

    @field_validator("tracks")
    @classmethod
    def require_unique_tracks(cls, value: tuple[TrackRiskLabel, ...]) -> tuple[TrackRiskLabel, ...]:
        track_ids = [item.track_id for item in value]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("frame track labels must be unique")
        return value


class RiskBenchmarkManifest(BaseModel):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    frames: tuple[FrameRiskLabels, ...]

    @field_validator("name", "source")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manifest text fields must not be blank")
        return value

    @field_validator("frames")
    @classmethod
    def require_ordered_frames(
        cls, value: tuple[FrameRiskLabels, ...]
    ) -> tuple[FrameRiskLabels, ...]:
        timestamps = [item.timestamp for item in value]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("frame timestamps must be unique and increasing")
        return value

    @classmethod
    def load(cls, path: Path) -> RiskBenchmarkManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class RiskBenchmarkReport:
    frames: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    dangerous_episodes: int
    detected_episodes: int
    missed_episodes: int
    precision: float | None
    recall: float | None
    f1: float | None
    episode_recall: float | None
    mean_detection_latency_seconds: float | None
    max_detection_latency_seconds: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _Episode:
    started_at: float
    detected_at: float | None = None


class BinaryRiskBenchmark:
    """Evaluate safety-supervisor alarm decisions against annotated track risk."""

    def evaluate(
        self, results: Iterable[dict], manifest: RiskBenchmarkManifest
    ) -> RiskBenchmarkReport:
        true_positive = false_positive = true_negative = false_negative = 0
        active: dict[str, _Episode] = {}
        latencies: list[float] = []
        dangerous_episodes = detected_episodes = missed_episodes = 0

        sentinel = object()
        frame_count = 0
        for result, labels in zip_longest(results, manifest.frames, fillvalue=sentinel):
            if result is sentinel or labels is sentinel:
                raise ValueError("replay result count must match manifest frame count")
            frame_count += 1
            predictions = self._alarm_predictions(result)
            expected = {item.track_id: item.dangerous for item in labels.tracks}

            for track_id, dangerous in expected.items():
                alarmed = predictions.pop(track_id, False)
                if dangerous and alarmed:
                    true_positive += 1
                elif dangerous:
                    false_negative += 1
                elif alarmed:
                    false_positive += 1
                else:
                    true_negative += 1
                dangerous_episodes, detected_episodes, missed_episodes = self._update_episode(
                    track_id,
                    dangerous,
                    alarmed,
                    labels.timestamp,
                    active,
                    latencies,
                    dangerous_episodes,
                    detected_episodes,
                    missed_episodes,
                )

            false_positive += sum(predictions.values())
            absent_tracks = set(active).difference(expected)
            for track_id in absent_tracks:
                dangerous_episodes, detected_episodes, missed_episodes = self._close_episode(
                    track_id,
                    active,
                    latencies,
                    dangerous_episodes,
                    detected_episodes,
                    missed_episodes,
                )

        for track_id in list(active):
            dangerous_episodes, detected_episodes, missed_episodes = self._close_episode(
                track_id,
                active,
                latencies,
                dangerous_episodes,
                detected_episodes,
                missed_episodes,
            )

        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return RiskBenchmarkReport(
            frames=frame_count,
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
            dangerous_episodes=dangerous_episodes,
            detected_episodes=detected_episodes,
            missed_episodes=missed_episodes,
            precision=precision,
            recall=recall,
            f1=f1,
            episode_recall=_safe_ratio(detected_episodes, dangerous_episodes),
            mean_detection_latency_seconds=(sum(latencies) / len(latencies) if latencies else None),
            max_detection_latency_seconds=max(latencies) if latencies else None,
        )

    @staticmethod
    def _alarm_predictions(result: dict) -> dict[str, bool]:
        tracks = result.get("tracks")
        if not isinstance(tracks, list):
            raise ValueError("replay result must contain a tracks list")
        predictions: dict[str, bool] = {}
        for track in tracks:
            try:
                track_id = track["track_id"]
                alarmed = track["decision"]["should_alarm"]
            except (KeyError, TypeError) as exc:
                raise ValueError("replay track is missing alarm decision fields") from exc
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("replay track_id must be a non-empty string")
            if not isinstance(alarmed, bool):
                raise ValueError("should_alarm must be boolean")
            if track_id in predictions:
                raise ValueError("replay frame contains duplicate track ids")
            predictions[track_id] = alarmed
        return predictions

    @staticmethod
    def _update_episode(
        track_id: str,
        dangerous: bool,
        alarmed: bool,
        timestamp: float,
        active: dict[str, _Episode],
        latencies: list[float],
        dangerous_episodes: int,
        detected_episodes: int,
        missed_episodes: int,
    ) -> tuple[int, int, int]:
        if dangerous:
            episode = active.setdefault(track_id, _Episode(timestamp))
            if alarmed and episode.detected_at is None:
                episode.detected_at = timestamp
            return dangerous_episodes, detected_episodes, missed_episodes
        if track_id not in active:
            return dangerous_episodes, detected_episodes, missed_episodes
        return BinaryRiskBenchmark._close_episode(
            track_id,
            active,
            latencies,
            dangerous_episodes,
            detected_episodes,
            missed_episodes,
        )

    @staticmethod
    def _close_episode(
        track_id: str,
        active: dict[str, _Episode],
        latencies: list[float],
        dangerous_episodes: int,
        detected_episodes: int,
        missed_episodes: int,
    ) -> tuple[int, int, int]:
        episode = active.pop(track_id)
        dangerous_episodes += 1
        if episode.detected_at is None:
            missed_episodes += 1
        else:
            detected_episodes += 1
            latencies.append(episode.detected_at - episode.started_at)
        return dangerous_episodes, detected_episodes, missed_episodes


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
