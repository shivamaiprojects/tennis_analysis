"""
Aggregate per-frame stats into per-player summaries.

Consumes the outputs of SpeedCalculator and produces the numbers shown on
the final stats overlay: max speed, average speed while moving, distance,
shot count.
"""
from typing import Dict, List


# Below this km/h we consider the player "not moving" — reject noise
MOVING_THRESHOLD_KMH = 1.0


class StatsAggregator:
    """Turn per-frame speed lists into per-player summary stats."""

    def aggregate(
        self,
        speeds_by_id: Dict[int, List[float]],
        distances_by_id: Dict[int, float],
        shot_counts_by_id: Dict[int, int],
    ) -> Dict[int, Dict[str, float]]:
        """Produce a summary dict per player.

        Args:
            speeds_by_id: {tid: [kmh_per_frame]} from SpeedCalculator.
            distances_by_id: {tid: total_meters} from SpeedCalculator.
            shot_counts_by_id: {tid: n_shots} from ShotDetector.

        Returns:
            {tid: {'max_speed_kmh', 'avg_speed_kmh', 'total_distance_m', 'shots'}}.
        """
        result: Dict[int, Dict[str, float]] = {}
        for tid, speeds in speeds_by_id.items():
            moving = [s for s in speeds if s >= MOVING_THRESHOLD_KMH]
            result[tid] = {
                "max_speed_kmh": max(speeds) if speeds else 0.0,
                "avg_speed_kmh": sum(moving) / len(moving) if moving else 0.0,
                "total_distance_m": distances_by_id.get(tid, 0.0),
                "shots": float(shot_counts_by_id.get(tid, 0)),
            }
        return result