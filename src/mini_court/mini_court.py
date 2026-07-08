"""
Mini court projection using homography.

The mini court is a top-down miniature of the tennis court drawn in the corner
of the frame. Players and the ball are projected from image pixels to real-world
court coordinates via a homography matrix, then rescaled to mini-court pixels
for drawing.

The homography H is computed once (court is static within a rally) using:
  - the 14 image-plane keypoints from CourtLineDetector
  - the 14 real-world reference points defined below
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.utils.conversions import (
    DOUBLES_COURT_LENGTH_M,
    DOUBLES_COURT_WIDTH_M,
    DOUBLES_ALLEY_WIDTH_M,
    SERVICE_BOX_LENGTH_M,
)


# =========================================================
# Real-world reference: the 14 court keypoints, in meters.
# Origin at top-left doubles corner. X=width, Y=length.
# =========================================================
# Derived indices for clarity
_SINGLES_LEFT_X = DOUBLES_ALLEY_WIDTH_M                        # 1.37
_SINGLES_RIGHT_X = DOUBLES_COURT_WIDTH_M - DOUBLES_ALLEY_WIDTH_M  # 9.60
_NET_Y = DOUBLES_COURT_LENGTH_M / 2                            # 11.885
_SERVICE_LINE_NEAR_Y = DOUBLES_COURT_LENGTH_M - SERVICE_BOX_LENGTH_M  # 17.37
_CENTER_X = DOUBLES_COURT_WIDTH_M / 2                          # 5.485

REAL_WORLD_KEYPOINTS_M: np.ndarray = np.array([
    # Doubles corners
    [0.0,                    0.0],                            # 0  top-left doubles
    [DOUBLES_COURT_WIDTH_M,  0.0],                            # 1  top-right doubles
    [0.0,                    DOUBLES_COURT_LENGTH_M],         # 2  bot-left doubles
    [DOUBLES_COURT_WIDTH_M,  DOUBLES_COURT_LENGTH_M],         # 3  bot-right doubles
    # Singles corners
    [_SINGLES_LEFT_X,        0.0],                            # 4  top-left singles
    [_SINGLES_LEFT_X,        DOUBLES_COURT_LENGTH_M],         # 5  bot-left singles
    [_SINGLES_RIGHT_X,       0.0],                            # 6  top-right singles
    [_SINGLES_RIGHT_X,       DOUBLES_COURT_LENGTH_M],         # 7  bot-right singles
    # Net line intersections (singles sidelines)
    [_SINGLES_LEFT_X,        _NET_Y],                         # 8  net-left
    [_SINGLES_RIGHT_X,       _NET_Y],                         # 9  net-right
    # Service line intersections (near court, singles sidelines)
    [_SINGLES_LEFT_X,        _SERVICE_LINE_NEAR_Y],           # 10 service-left
    [_SINGLES_RIGHT_X,       _SERVICE_LINE_NEAR_Y],           # 11 service-right
    # Center T intersections
    [_CENTER_X,              _NET_Y],                         # 12 net T
    [_CENTER_X,              _SERVICE_LINE_NEAR_Y],           # 13 service T
], dtype=np.float32)


class MiniCourt:
    """Compute homography and draw a mini top-down court on video frames.

    Attributes:
        H: 3x3 homography from image pixels to real court meters.
        H_inv: 3x3 inverse homography (real meters to image pixels).
        reprojection_error_m: mean reprojection error of the 14 keypoints (meters).
    """

    # Mini court drawing config (pixels on the output frame)
    MINI_COURT_HEIGHT_PX: int = 500  # length dimension
    MINI_COURT_WIDTH_PX: int = 250   # width dimension
    MINI_COURT_MARGIN_PX: int = 30   # distance from frame edges

    def __init__(self, image_keypoints: np.ndarray, frame_shape: Tuple[int, int, int]):
        """Compute homography from a set of detected image keypoints.

        Args:
            image_keypoints: Flat array of 28 values from CourtLineDetector.
            frame_shape: Video frame shape (H, W, 3) — used to position the mini court.
        """
        # Reshape flat [x0,y0,x1,y1,...] into (14, 2)
        image_pts = image_keypoints.astype(np.float32).reshape(-1, 2)
        real_pts = REAL_WORLD_KEYPOINTS_M

        # Compute homography (image -> real world) using RANSAC for robustness
        self.H, mask = cv2.findHomography(image_pts, real_pts, method=cv2.RANSAC)
        # Also compute the inverse (real world -> image), useful for some overlays
        self.H_inv = np.linalg.inv(self.H)

        # Sanity check: mean reprojection error
        reprojected = cv2.perspectiveTransform(
            image_pts.reshape(-1, 1, 2), self.H
        ).reshape(-1, 2)
        errors = np.linalg.norm(reprojected - real_pts, axis=1)
        self.reprojection_error_m = float(errors.mean())
        print(f"[MiniCourt] Homography reprojection error: "
              f"{self.reprojection_error_m * 100:.1f} cm (mean over 14 keypoints)")

        # Cache mini-court placement in output frame coordinates
        frame_h, frame_w = frame_shape[:2]
        self._mini_top_left = (
            frame_w - self.MINI_COURT_WIDTH_PX - self.MINI_COURT_MARGIN_PX,
            self.MINI_COURT_MARGIN_PX,
        )

        # Scale: real meters -> mini-court pixels
        self._m_to_mini_px_x = self.MINI_COURT_WIDTH_PX / DOUBLES_COURT_WIDTH_M
        self._m_to_mini_px_y = self.MINI_COURT_HEIGHT_PX / DOUBLES_COURT_LENGTH_M

    def project_to_court(self, image_point: Tuple[float, float]) -> Tuple[float, float]:
        """Project an image-plane point to real-world court coordinates (meters)."""
        pt = np.array([[[image_point[0], image_point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])

    def real_to_mini_px(self, real_point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert a real-world court (meter) coordinate to mini-court pixel coordinates."""
        origin_x, origin_y = self._mini_top_left
        mini_x = int(origin_x + real_point[0] * self._m_to_mini_px_x)
        mini_y = int(origin_y + real_point[1] * self._m_to_mini_px_y)
        return mini_x, mini_y

    def draw_mini_court(self, frame: np.ndarray) -> np.ndarray:
        """Draw the empty mini-court rectangle on the frame."""
        out = frame.copy()
        ox, oy = self._mini_top_left
        w, h = self.MINI_COURT_WIDTH_PX, self.MINI_COURT_HEIGHT_PX

        # Semi-transparent white background
        overlay = out.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + w, oy + h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

        # Court outline (outer doubles rectangle)
        cv2.rectangle(out, (ox, oy), (ox + w, oy + h), (0, 0, 0), 2)

        # Draw the singles sidelines
        sl = int(ox + DOUBLES_ALLEY_WIDTH_M * self._m_to_mini_px_x)
        sr = int(ox + (DOUBLES_COURT_WIDTH_M - DOUBLES_ALLEY_WIDTH_M) * self._m_to_mini_px_x)
        cv2.line(out, (sl, oy), (sl, oy + h), (0, 0, 0), 1)
        cv2.line(out, (sr, oy), (sr, oy + h), (0, 0, 0), 1)

        # Net line (across the middle)
        net_y = int(oy + (DOUBLES_COURT_LENGTH_M / 2) * self._m_to_mini_px_y)
        cv2.line(out, (ox, net_y), (ox + w, net_y), (0, 0, 0), 2)

        # Service lines
        near_sl_y = int(oy + (DOUBLES_COURT_LENGTH_M - SERVICE_BOX_LENGTH_M) * self._m_to_mini_px_y)
        far_sl_y = int(oy + SERVICE_BOX_LENGTH_M * self._m_to_mini_px_y)
        cv2.line(out, (sl, near_sl_y), (sr, near_sl_y), (0, 0, 0), 1)
        cv2.line(out, (sl, far_sl_y), (sr, far_sl_y), (0, 0, 0), 1)

        # Center service line
        cx = int(ox + (DOUBLES_COURT_WIDTH_M / 2) * self._m_to_mini_px_x)
        cv2.line(out, (cx, far_sl_y), (cx, near_sl_y), (0, 0, 0), 1)

        return out

    def draw_players_on_mini_court(
        self,
        frame: np.ndarray,
        player_foot_positions: Dict[int, Tuple[float, float]],
    ) -> np.ndarray:
        """Draw a dot on the mini court for each player.

        Args:
            frame: Frame that already has the empty mini-court drawn.
            player_foot_positions: {track_id: (image_x, image_y)} — foot in image pixels.
        """
        out = frame.copy()
        for tid, image_foot in player_foot_positions.items():
            real = self.project_to_court(image_foot)
            mini = self.real_to_mini_px(real)
            color = (0, 0, 255) if tid == 1 else (255, 0, 0)  # ID 1 red, ID 2 blue
            cv2.circle(out, mini, radius=6, color=color, thickness=-1)
            cv2.putText(out, f"P{tid}", (mini[0] + 8, mini[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out

    def draw_ball_on_mini_court(
        self, frame: np.ndarray, ball_image_point: Tuple[float, float]
    ) -> np.ndarray:
        """Draw the ball position on the mini court as a yellow dot."""
        out = frame.copy()
        real = self.project_to_court(ball_image_point)
        mini = self.real_to_mini_px(real)
        cv2.circle(out, mini, radius=4, color=(0, 255, 255), thickness=-1)
        return out