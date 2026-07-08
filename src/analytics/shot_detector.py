"""
Shot detection from ball trajectory.

A shot event is a local minimum in the ball's Y position (which grows downward
in image coordinates — a minimum in the array sense is when the ball is at its
LOWEST point on screen, i.e., near court level, when it gets hit).

For each detected shot event, we attribute the shot to whichever player is
closest to the ball at that frame.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.utils.bbox_utils import get_center, get_foot_position, measure_distance


# How many frames of smoothing before shot detection (rejects small jitters)
SMOOTHING_WINDOW = 5
# Minimum frames between two consecutive shots (prevents duplicates)
MIN_FRAMES_BETWEEN_SHOTS = 15


class ShotDetector:
    """Detect shot events and attribute each to a player."""

    def detect_shots(
        self,
        ball_detections: List[Dict[int, List[float]]],
        player_detections: List[Dict[int, List[float]]],
    ) -> Tuple[List[int], Dict[int, int]]:
        """Return (list of shot frame indices, {tid: shot_count}).

        Args:
            ball_detections: Interpolated ball positions per frame.
            player_detections: Filtered player detections per frame.
        """
        # Extract ball Y position per frame using bbox center
        ball_ys = []
        for det in ball_detections:
            if 1 in det:
                _, cy = get_center(det[1])
                ball_ys.append(cy)
            else:
                ball_ys.append(np.nan)

        # Smooth the Y series to reject jitter
        y_series = pd.Series(ball_ys).interpolate().bfill().ffill()
        y_smoothed = y_series.rolling(SMOOTHING_WINDOW, center=True).mean().bfill().ffill()
        y_arr = y_smoothed.to_numpy()

        # A shot is a LOCAL MAXIMUM of Y (Y down = ball at lowest visual point)
        peaks, _ = find_peaks(y_arr, distance=MIN_FRAMES_BETWEEN_SHOTS)
        shot_frames: List[int] = peaks.tolist()

        # Attribute each shot to nearest player
        shot_counts: Dict[int, int] = {}
        for frame_idx in shot_frames:
            if frame_idx >= len(ball_detections):
                continue
            ball_det = ball_detections[frame_idx]
            player_det = player_detections[frame_idx]
            if 1 not in ball_det or not player_det:
                continue
            ball_center = get_center(ball_det[1])
            closest_tid = min(
                player_det.keys(),
                key=lambda tid: measure_distance(
                    ball_center, get_foot_position(player_det[tid])
                ),
            )
            shot_counts[closest_tid] = shot_counts.get(closest_tid, 0) + 1

        return shot_frames, shot_counts