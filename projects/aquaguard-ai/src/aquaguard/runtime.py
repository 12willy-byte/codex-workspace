from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Protocol

from aquaguard.video.models import FrameRead, VideoFrame
from aquaguard.vision.adapter import CalibratedObservationAdapter
from aquaguard.vision.models import PixelTrackObservation
from aquaguard.world.models import CameraObservation


class VideoSource(Protocol):
    def read(self) -> FrameRead: ...

    def close(self) -> None: ...


class FrameAnalyzer(Protocol):
    def analyze(self, frame: VideoFrame) -> list[PixelTrackObservation]: ...


class FrameObserver(Protocol):
    def ingest(self, frame: VideoFrame) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    ok: bool
    result: dict | None = None
    error: str | None = None
    retry_at: float | None = None


class ManagedVideoRuntime(Protocol):
    def step(self) -> RuntimeStep: ...

    def close(self) -> None: ...


class WorldFrameProcessor(Protocol):
    def process_frame(
        self,
        camera_id: str,
        timestamp: float,
        observations: list[CameraObservation],
    ) -> dict: ...


class VideoWorldRuntime:
    """One-step orchestration from video capture to the pool world model."""

    def __init__(
        self,
        source: VideoSource,
        analyzer: FrameAnalyzer,
        adapter: CalibratedObservationAdapter,
        pipeline: WorldFrameProcessor,
        observers: tuple[FrameObserver, ...] = (),
    ):
        self.source = source
        self.analyzer = analyzer
        self.adapter = adapter
        self.pipeline = pipeline
        self.observers = observers
        self.capabilities = tuple(getattr(analyzer, "capabilities", ()))

    def step(self) -> RuntimeStep:
        read = self.source.read()
        if not read.ok or read.frame is None:
            return RuntimeStep(False, error=read.error, retry_at=read.retry_at)
        for observer in self.observers:
            observer.ingest(read.frame)
        pixels = self.analyzer.analyze(read.frame)
        observations = self.adapter.convert(pixels)
        return RuntimeStep(
            True,
            result=self.pipeline.process_frame(
                read.frame.camera_id, read.frame.timestamp, observations
            ),
        )

    def close(self) -> None:
        self.source.close()


@dataclass(slots=True)
class RuntimeCameraStatus:
    camera_id: str
    capabilities: tuple[str, ...] = ()
    protection_level: str = "unconfigured"
    running: bool = False
    frames_processed: int = 0
    last_error: str | None = None
    last_result: dict | None = None

    def snapshot(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "capabilities": self.capabilities,
            "protection_level": self.protection_level,
            "running": self.running,
            "frames_processed": self.frames_processed,
            "last_error": self.last_error,
            "last_result": self.last_result,
        }


class VideoRuntimeManager:
    """Run camera pipelines independently so one failed stream cannot stop the others."""

    def __init__(self, *, idle_wait_seconds: float = 0.05, join_timeout_seconds: float = 2):
        if idle_wait_seconds < 0 or join_timeout_seconds < 0:
            raise ValueError("Runtime wait durations must not be negative")
        self.idle_wait_seconds = idle_wait_seconds
        self.join_timeout_seconds = join_timeout_seconds
        self._runtimes: dict[str, ManagedVideoRuntime] = {}
        self._statuses: dict[str, RuntimeCameraStatus] = {}
        self._threads: dict[str, Thread] = {}
        self._stop = Event()
        self._lock = Lock()
        self._started = False

    def add(self, camera_id: str, runtime: ManagedVideoRuntime) -> None:
        if not camera_id:
            raise ValueError("camera_id is required")
        with self._lock:
            if self._started:
                raise RuntimeError("Cannot add a runtime after the manager has started")
            if camera_id in self._runtimes:
                raise ValueError(f"Runtime already exists for camera {camera_id}")
            self._runtimes[camera_id] = runtime
            capabilities = tuple(getattr(runtime, "capabilities", ()))
            self._statuses[camera_id] = RuntimeCameraStatus(
                camera_id,
                capabilities=capabilities,
                protection_level=self._protection_level(capabilities),
            )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop.clear()
            for camera_id, runtime in self._runtimes.items():
                status = self._statuses[camera_id]
                status.running = True
                thread = Thread(
                    target=self._run,
                    args=(camera_id, runtime),
                    name=f"aquaguard-video-{camera_id}",
                    daemon=True,
                )
                self._threads[camera_id] = thread
                thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            runtimes = tuple(self._runtimes.values())
            threads = tuple(self._threads.items())
        for runtime in runtimes:
            runtime.close()
        for _, thread in threads:
            thread.join(self.join_timeout_seconds)
        with self._lock:
            alive_threads = {}
            for camera_id, thread in threads:
                if thread.is_alive():
                    alive_threads[camera_id] = thread
                    status = self._statuses[camera_id]
                    status.running = True
                    status.last_error = "shutdown_timeout"
                else:
                    self._statuses[camera_id].running = False
            self._threads = alive_threads
            self._started = bool(alive_threads)

    def statuses(self) -> list[dict]:
        with self._lock:
            return [self._statuses[key].snapshot() for key in sorted(self._statuses)]

    def _run(self, camera_id: str, runtime: ManagedVideoRuntime) -> None:
        try:
            while not self._stop.is_set():
                step = runtime.step()
                with self._lock:
                    status = self._statuses[camera_id]
                    if step.ok:
                        status.frames_processed += 1
                        status.last_result = step.result
                        status.last_error = None
                    else:
                        status.last_error = step.error
                if not step.ok and step.error == "end_of_stream":
                    break
                if not step.ok:
                    self._stop.wait(self.idle_wait_seconds)
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._statuses[camera_id].last_error = f"runtime_failure: {exc}"
        finally:
            runtime.close()
            with self._lock:
                self._statuses[camera_id].running = False

    @staticmethod
    def _protection_level(capabilities: tuple[str, ...]) -> str:
        available = set(capabilities)
        if "drowning_risk_validated" in available:
            return "validated_assistive_alerting"
        if "body_verticality" in available:
            return "pose_baseline"
        if "person_detection" in available:
            return "tracking_only"
        return "unconfigured"
