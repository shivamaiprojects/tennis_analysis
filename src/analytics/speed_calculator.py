"""
Per-frame speed calculation for players.

Speed is computed over a sliding window of frames (default 5) to reject
detection noise. For each frame N, we look at the player's position at
frame N-window and frame N, compute the real-world distance, and divide by
elapsed time.

The output is a per-player, per-frame speed in km/h — one number per
(player, frame) pair.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.utils.bbox_utils import get_foot_position, measure_distance
from src.utils.conversions import mps_to_kmh


# Number of frames to look back when computing speed.
# Larger = smoother but more latency. 5 frames at 30 FPS = 1/6 s window.
SPEED_WINDOW_FRAMES = 5


class SpeedCalculator:
    """Compute per-frame player speeds in km/h using a homography."""

    def __init__(self, mini_court, fps: float, window: int = SPEED_WINDOW_FRAMES):
        """
        Args:
            mini_court: A MiniCourt instance (used for image->meters projection).
            fps: Video frame rate.
            window: How many frames to look back for the speed estimate.
        """
        self.mini_court = mini_court
        self.fps = fps
        self.window = window

    def compute_speeds(
        self,
        player_detections: List[Dict[int, List[float]]],
    ) -> Dict[int, List[float]]:
        """Compute km/h per frame for each player.

        Args:
            player_detections: Per-frame dict of {track_id: bbox}.

        Returns:
            Dict {track_id: [speed_kmh_frame_0, speed_kmh_frame_1, ...]}.
            First `window` frames of each player are 0.0 (no history yet).
        """
        # Discover all player IDs across all frames
        all_ids = set()
        for det in player_detections:
            all_ids.update(det.keys())

        num_frames = len(player_detections)
        speeds: Dict[int, List[float]] = {tid: [0.0] * num_frames for tid in all_ids}
        # Store foot positions in real-world meters per frame (or None if missing)
        real_positions: Dict[int, List[Tuple[float, float] | None]] = {
            tid: [None] * num_frames for tid in all_ids
        }

        # Step 1: project every player's foot to real-world meters
        for i, det in enumerate(player_detections):
            for tid, box in det.items():
                foot = get_foot_position(box)
                real_positions[tid][i] = self.mini_court.project_to_court(foot)

        # Step 2: sliding-window speed
        dt = self.window / self.fps   # elapsed time between window endpoints, seconds
        for tid in all_ids:
            positions = real_positions[tid]
            for i in range(self.window, num_frames):
                p_now = positions[i]
                p_then = positions[i - self.window]
                if p_now is None or p_then is None:
                    continue  # can't compute; leave speed as 0.0
                dist_m = measure_distance(p_now, p_then)
                speed_mps = dist_m / dt
                speeds[tid][i] = mps_to_kmh(speed_mps)

        return speeds

    def compute_distance_covered(
        self,
        player_detections: List[Dict[int, List[float]]],
    ) -> Dict[int, float]:
        """Total distance covered per player, in meters.

        Sums the frame-to-frame Euclidean distance across the whole video.
        Missing frames are skipped (not interpolated) to avoid inventing motion.
        """
        all_ids = set()
        for det in player_detections:
            all_ids.update(det.keys())

        totals: Dict[int, float] = {tid: 0.0 for tid in all_ids}

        # Cache projected positions per player
        real_positions: Dict[int, List[Tuple[float, float] | None]] = {
            tid: [None] * len(player_detections) for tid in all_ids
        }
        for i, det in enumerate(player_detections):
            for tid, box in det.items():
                real_positions[tid][i] = self.mini_court.project_to_court(
                    get_foot_position(box)
                )

        for tid in all_ids:
            positions = real_positions[tid]
            for i in range(1, len(positions)):
                if positions[i] is None or positions[i - 1] is None:
                    continue
                totals[tid] += measure_distance(positions[i], positions[i - 1])

        return totals