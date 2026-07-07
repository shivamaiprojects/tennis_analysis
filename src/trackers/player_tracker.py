"""
Player detection and tracking using pretrained YOLO plus ByteTrack.

Pipeline:
  1. Ultralytics YOLO detects all persons per frame (COCO class 0).
  2. ByteTrack (built into Ultralytics) assigns persistent IDs across frames.
  3. Court keypoints from Module 4 filter the detections down to the 2 real players.

The stub-cache pattern lets us skip re-running YOLO on unchanged video during
downstream debugging, cutting the debug loop from minutes to seconds.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.bbox_utils import get_center, measure_distance


# COCO class index for "person"
COCO_PERSON_CLASS = 0


class PlayerTracker:
    """Detect and track players in a tennis video.

    Detections are stored as a list (one entry per frame) of dicts mapping
    track_id -> [x1, y1, x2, y2].

    Attributes:
        model: The Ultralytics YOLO model instance.
    """


    def __init__(self, model_path: str = "models/yolo26n.pt"):
        """Load pretrained YOLO. Uses your local YOLO26-nano by default for
        consistency with the ball detector and faster inference.

        Args:
            model_path: Path to a YOLO weights file. Defaults to models/yolo26n.pt
                (fast, small, pretrained on COCO — good enough for the person class).
        """
        self.model = YOLO(model_path)
        print(f"[PlayerTracker] Loaded YOLO model: {model_path}")

    def detect_frames(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str | Path] = None,
    ) -> List[Dict[int, List[float]]]:
        """Detect and track persons across all frames.

        Args:
            frames: List of BGR frames from read_video().
            read_from_stub: If True and stub exists, load cached detections.
            stub_path: Where to save/load the cache. If None, no caching.

        Returns:
            List of dicts, one per frame. Each dict maps track_id -> bbox.
        """
        # Cache hit: return early
        if read_from_stub and stub_path and Path(stub_path).exists():
            print(f"[PlayerTracker] Loading cached detections from {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print(f"[PlayerTracker] Running YOLO tracking on {len(frames)} frames...")
        detections: List[Dict[int, List[float]]] = []
        for i, frame in enumerate(frames):
            detections.append(self._detect_frame(frame))
            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1}/{len(frames)} frames")

        # Save cache
        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(detections, f)
            print(f"[PlayerTracker] Cached detections to {stub_path}")

        return detections

    def _detect_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        """Run YOLO+ByteTrack on one frame. Returns {track_id: bbox}."""
        # persist=True keeps the tracker's state across calls
        # classes=[0] restricts detection to "person" only, skipping other COCO classes
        results = self.model.track(
            frame, persist=True, classes=[COCO_PERSON_CLASS], verbose=False
        )[0]

        out: Dict[int, List[float]] = {}
        # If no detections OR no IDs assigned this frame, return empty
        if results.boxes.id is None:
            return out

        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()
        for tid, box in zip(ids, boxes):
            out[int(tid)] = box.tolist()
        return out

    def choose_and_filter_players(
        self,
        court_keypoints: np.ndarray,
        player_detections: List[Dict[int, List[float]]],
    ) -> List[Dict[int, List[float]]]:
        """Keep only the 2 track IDs closest to the court across the video.

        Uses the first non-empty frame to pick the 2 winners, then applies
        that filter to every frame.
        """
        first_non_empty = next((d for d in player_detections if d), {})

        if len(first_non_empty) <= 2:
            # Nothing to filter — keep whatever we found
            chosen = list(first_non_empty.keys())
        else:
            chosen = self._choose_players(court_keypoints, first_non_empty)

        return [
            {tid: box for tid, box in d.items() if tid in chosen}
            for d in player_detections
        ]

    @staticmethod
    def _choose_players(
        court_keypoints: np.ndarray, detections: Dict[int, List[float]]
    ) -> List[int]:
        """Return the 2 track_ids whose bbox centers are closest to any keypoint."""
        distances = []
        for tid, box in detections.items():
            player_center = get_center(box)
            min_d = float("inf")
            # court_keypoints is a flat [x0,y0,x1,y1,...] array of 28 values
            for i in range(0, len(court_keypoints), 2):
                kp = (court_keypoints[i], court_keypoints[i + 1])
                min_d = min(min_d, measure_distance(player_center, kp))
            distances.append((tid, min_d))

        # Sort by distance ascending, take the 2 closest
        distances.sort(key=lambda pair: pair[1])
        return [distances[0][0], distances[1][0]]

    def draw_bboxes(
        self,
        frames: List[np.ndarray],
        detections: List[Dict[int, List[float]]],
    ) -> List[np.ndarray]:
        """Return copies of frames with red bboxes and 'Player N' labels."""
        out = []
        for frame, det in zip(frames, detections):
            f = frame.copy()
            for tid, box in det.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(
                    f, f"Player {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 2)
            out.append(f)
        return out