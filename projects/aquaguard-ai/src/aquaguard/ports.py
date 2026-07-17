from typing import Protocol

from aquaguard.domain import AlarmEvent


class FrameSource(Protocol):
    def read(self) -> object | None: ...
    def close(self) -> None: ...


class VisionAnalyzer(Protocol):
    def analyze(self, frame: object) -> list[dict]: ...


class AlarmSink(Protocol):
    def publish(self, event: AlarmEvent) -> None: ...
