"""
Draw a per-player stats box overlay on video frames.

Renders a semi-transparent panel showing: max speed, average speed, total
distance covered, and shots hit — for each player. Called once per frame
after all other overlays.
"""
from typing import Dict, List

import cv2
import numpy as np


class StatsDrawer:
    """Overlay a per-player stats panel on video frames."""

    # Position and size on the output frame (bottom-left corner)
    PANEL_WIDTH = 380
    PANEL_HEIGHT = 140
    MARGIN = 30

    def draw(
        self,
        frames: List[np.ndarray],
        summary_by_id: Dict[int, Dict[str, float]],
    ) -> List[np.ndarray]:
        """Return frames with a stats panel drawn on each.

        Args:
            frames: Input frames.
            summary_by_id: Output of StatsAggregator.aggregate().
        """
        out = []
        for frame in frames:
            out.append(self._draw_one(frame, summary_by_id))
        return out

    def _draw_one(
        self, frame: np.ndarray, summary_by_id: Dict[int, Dict[str, float]]
    ) -> np.ndarray:
        f = frame.copy()
        h, w = f.shape[:2]

        # Panel bounds (bottom-left)
        x0 = self.MARGIN
        y0 = h - self.PANEL_HEIGHT - self.MARGIN
        x1 = x0 + self.PANEL_WIDTH
        y1 = h - self.MARGIN

        # Semi-transparent black background
        overlay = f.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, f, 0.4, 0, f)

        # Header
        cv2.putText(f, "Player Stats", (x0 + 10, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Column headers
        col_y = y0 + 55
        cv2.putText(f, "Player", (x0 + 10, col_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(f, "Max km/h", (x0 + 100, col_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(f, "Avg km/h", (x0 + 190, col_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(f, "Dist m", (x0 + 280, col_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Per-player rows
        for i, (tid, stats) in enumerate(sorted(summary_by_id.items())):
            row_y = col_y + 30 + i * 25
            color = (0, 0, 255) if tid == 1 else (255, 100, 100)
            cv2.putText(f, f"P{tid}", (x0 + 10, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(f, f"{stats['max_speed_kmh']:.1f}", (x0 + 100, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(f, f"{stats['avg_speed_kmh']:.1f}", (x0 + 190, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(f, f"{stats['total_distance_m']:.1f}", (x0 + 280, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return f