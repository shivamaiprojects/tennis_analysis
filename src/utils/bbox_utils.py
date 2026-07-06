"""
Bounding-box geometry utilities.

A bounding box (bbox) is stored as [x1, y1, x2, y2] where (x1, y1) is the
top-left corner and (x2, y2) is the bottom-right corner, in pixel coordinates
with origin at the top-left of the image.

Called from downstream modules (trackers, mini_court, analytics) whenever we
need a specific point on a bbox (center, foot) or a distance between points.
"""
from typing import Sequence, Tuple

import numpy as np

# A bbox is any sequence of 4 numbers: [x1, y1, x2, y2]
BBox = Sequence[float]


def get_center(bbox: BBox) -> Tuple[int, int]:
    """Return the center point (x, y) of a bbox as integers.

    Used for: computing distance from a detected person to court keypoints
    (to identify which detections are actual players).
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_foot_position(bbox: BBox) -> Tuple[int, int]:
    """Return the bottom-middle of a bbox — where a standing player touches the court.

    Used for: player tracking on the mini court and speed calculation.
    The foot position is what makes physical sense for a player standing on the
    ground — the body center would shift when a player crouches or jumps.
    """
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def get_bbox_width(bbox: BBox) -> float:
    """Width of the bbox in pixels."""
    return bbox[2] - bbox[0]


def get_bbox_height(bbox: BBox) -> float:
    """Height of the bbox in pixels."""
    return bbox[3] - bbox[1]


def measure_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean (straight-line) distance between two 2D points, in the input units.

    Used for: filtering audience from players (distance to nearest keypoint)
    and computing motion between frames (foot at t vs foot at t+1).
    """
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def measure_xy_distance(
    p1: Sequence[float], p2: Sequence[float]
) -> Tuple[float, float]:
    """Signed (dx, dy) between two points — preserves direction of motion.

    Used for: knowing whether a player is moving left/right or up/down (not
    just how far), which is useful for shot-type analysis later.
    """
    return p1[0] - p2[0], p1[1] - p2[1]