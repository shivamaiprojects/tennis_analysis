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

from src.utils.bbox_utils import get_center, get_foot_position, measure_distance


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
        """Keep only the 2 track IDs that are actual players.

        Two-stage filter:
        1. Discard track IDs that appear in fewer than MIN_FRAMES_FRACTION
            of frames (rejects transient ball-kid/audience blips).
        2. Rank the survivors by MEAN signed distance to the court polygon.
            Top 2 win.
        """
        MIN_FRAMES_FRACTION = 0.5   # a real player is on-frame at least half the video

        polygon = np.array([
            [court_keypoints[0], court_keypoints[1]],
            [court_keypoints[2], court_keypoints[3]],
            [court_keypoints[6], court_keypoints[7]],
            [court_keypoints[4], court_keypoints[5]],
        ], dtype=np.float32)

        # Score every track ID across every frame it appears in
        scores: Dict[int, List[float]] = {}
        for det in player_detections:
            for tid, box in det.items():
                foot = get_foot_position(box)
                signed_dist = cv2.pointPolygonTest(
                    polygon, (float(foot[0]), float(foot[1])), True
                )
                scores.setdefault(tid, []).append(signed_dist)

        # Filter to track IDs present in enough frames to be a real player
        min_frames = int(MIN_FRAMES_FRACTION * len(player_detections))
        persistent = {tid: vals for tid, vals in scores.items() if len(vals) >= min_frames}

        print(f"[PLAYER_FILTER] {len(scores)} total tracks, "
            f"{len(persistent)} present in >={min_frames} frames")

        if len(persistent) < 2:
            # Fallback: use all tracks if we don't have 2 persistent ones
            persistent = scores

        # Rank the survivors by mean signed distance (most-inside first)
        mean_scores = [(tid, sum(vals) / len(vals)) for tid, vals in persistent.items()]
        mean_scores.sort(key=lambda pair: pair[1], reverse=True)

        print("[PLAYER_FILTER] Ranked persistent candidates:")
        for tid, score in mean_scores:
            print(f"  tid={tid}: mean_signed_dist={score:.1f} px "
                f"(n_frames={len(persistent[tid])})")

        chosen = [mean_scores[0][0], mean_scores[1][0]] if len(mean_scores) >= 2 \
                else list(mean_scores[0:1])
        print(f"[PLAYER_FILTER] Chosen: {chosen}")

        return [
            {tid: box for tid, box in d.items() if tid in chosen}
            for d in player_detections
        ]
    


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