"""
Ball detection with interpolation for missed frames.

Uses the fine-tuned YOLO26 model to detect a single tennis ball per frame,
then fills gaps in detection via linear interpolation on the (x, y) position.

Unlike player tracking, we don't use ByteTrack — there is only ever one ball
in play, so persistent IDs aren't needed. We just need the best detection per
frame and clean interpolation over the misses.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# We use a low confidence threshold because the ball is small and often blurry.
# False positives are cheap to handle (interpolation smooths them out);
# false negatives are expensive (gaps we have to fill).
BALL_CONF_THRESHOLD = 0.15


class BallTracker:
    """Detect and interpolate ball positions using the fine-tuned YOLO26."""

    def __init__(self, model_path: str | Path):
        """Load your fine-tuned ball detector.

        Args:
            model_path: Path to your best.pt, e.g.
                'models/yolo26n_fine_tuned_100epoch_best.pt'.
        """
        self.model = YOLO(str(model_path))
        print(f"[BallTracker] Loaded custom ball model: {model_path}")

    def detect_frames(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str | Path] = None,
    ) -> List[Dict[int, List[float]]]:
        """Detect the ball in every frame.

        Returns:
            List of dicts (one per frame). Each dict is either {1: [x1,y1,x2,y2]}
            if the ball was found, or {} if not.
        """
        # Cache hit
        if read_from_stub and stub_path and Path(stub_path).exists():
            print(f"[BallTracker] Loading cached detections from {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print(f"[BallTracker] Running ball detection on {len(frames)} frames...")
        detections = []
        for i, frame in enumerate(frames):
            detections.append(self._detect_frame(frame))
            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(frames)} frames")

        # Save cache
        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(detections, f)
            print(f"[BallTracker] Cached detections to {stub_path}")

        return detections

    def _detect_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        """Return the single highest-confidence ball detection, or {} if none."""
        results = self.model.predict(
            frame, conf=BALL_CONF_THRESHOLD, verbose=False
        )[0]

        if len(results.boxes) == 0:
            return {}

        # Multiple detections possible (false positives) — take the most confident
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        # Ball is a single object, so we use ID 1 as a fixed label
        return {1: boxes[best].tolist()}

    @staticmethod
    def interpolate_ball_positions(
        ball_detections: List[Dict[int, List[float]]],
    ) -> List[Dict[int, List[float]]]:
        """Fill missing frames via linear interpolation on (x1, y1, x2, y2).

        Frames with no detection are represented as an empty dict {}. This
        method turns them into interpolated boxes based on the surrounding
        detected frames.
        """
        # Extract (x1, y1, x2, y2) per frame, using NaN for missing frames
        positions = [d.get(1, [np.nan] * 4) for d in ball_detections]

        # Put into a pandas DataFrame — one row per frame, 4 columns
        df = pd.DataFrame(positions, columns=["x1", "y1", "x2", "y2"])

        # Linear interpolation of NaN values
        df = df.interpolate()
        # If the very first frames are NaN (no detection yet), fill backward
        df = df.bfill()
        # If the very last frames are NaN, fill forward
        df = df.ffill()

        return [{1: row.tolist()} for _, row in df.iterrows()]

    def draw_bboxes(
        self,
        frames: List[np.ndarray],
        detections: List[Dict[int, List[float]]],
    ) -> List[np.ndarray]:
        """Return copies of frames with yellow bboxes labeled 'Ball'."""
        out = []
        for frame, det in zip(frames, detections):
            f = frame.copy()
            for _, box in det.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(
                    f, "Ball", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                )
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 255), 2)
            out.append(f)
        return out