from aquaguard.world.fusion import MultiCameraFusion
from aquaguard.world.models import CameraObservation, PoolGeometry
from aquaguard.world.occupancy import PoolOccupancy
from aquaguard.world.prediction import FutureRiskPredictor
from aquaguard.world.safety import IndependentSafetySupervisor
from aquaguard.world.temporal import TemporalTrackBuffer


class WorldModelPipeline:
    def __init__(self, geometry: PoolGeometry | None = None):
        self.geometry = geometry or PoolGeometry()
        self.geometry.validate()
        self.fusion = MultiCameraFusion()
        self.temporal = TemporalTrackBuffer()
        self.occupancy = PoolOccupancy(self.geometry)
        self.predictor = FutureRiskPredictor()
        self.supervisor = IndependentSafetySupervisor()

    def process(self, observations: list[CameraObservation]) -> dict:
        for observation in observations:
            observation.validate(self.geometry)
        tracks = self.fusion.fuse(observations)
        results = []
        for track in tracks:
            forecast = self.predictor.predict(self.temporal.append(track))
            decision = self.supervisor.evaluate(track, forecast)
            results.append(
                {
                    "track_id": track.track_id,
                    "camera_ids": track.camera_ids,
                    "position": {"x": track.pool_x, "y": track.pool_y},
                    "forecast": forecast.to_dict(),
                    "decision": decision.to_dict(),
                }
            )
        return {"tracks": results, "bev": self.occupancy.encode(tracks)}
