"""
Tennis Analysis — main entry point.

Runs the full pipeline:
    input video  →  court keypoints
                 →  player tracking (YOLO26 + ByteTrack + polygon filter)
                 →  ball tracking (fine-tuned YOLO26 + interpolation)
                 →  homography (image pixels ↔ real court meters)
                 →  speed / distance / shot analytics
                 →  annotated output video with mini court + stats overlay

Usage:
    python main.py
    python main.py --config custom_config.yaml
    python main.py --no-stubs                # ignore cached detections
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils import (
    read_video, save_video, get_video_metadata,
    get_foot_position, get_center,
)
from src.court_detection import CourtLineDetector
from src.trackers import PlayerTracker, BallTracker
from src.mini_court import MiniCourt
from src.analytics import (
    SpeedCalculator, StatsAggregator, ShotDetector, StatsDrawer,
)


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("runs/last_run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config file and return it as a nested dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a dict, got {type(config)}")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--no-stubs", action="store_true",
        help="Ignore cached detections and re-run all inference.",
    )
    return parser.parse_args()


def run_pipeline(config: Dict[str, Any], use_stubs: bool) -> None:
    """Run the full tennis analysis pipeline.

    Args:
        config: Loaded config dict.
        use_stubs: If True, load cached detections when available.
    """
    paths = config["paths"]

    # --- Ensure output directories exist ---
    Path(paths["output_video"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["stubs"]["player_detections"]).parent.mkdir(parents=True, exist_ok=True)

    # === 1. Read video ===
    log.info("Reading input video: %s", paths["input_video"])
    frames = read_video(paths["input_video"])
    meta = get_video_metadata(paths["input_video"])
    log.info("Loaded %d frames at %.1f FPS (%dx%d)",
             len(frames), meta["fps"], meta["width"], meta["height"])

    # === 2. Court keypoints (one prediction — court is static) ===
    log.info("Detecting court keypoints...")
    court_detector = CourtLineDetector(paths["models"]["court_keypoints"])
    court_keypoints = court_detector.predict(frames[0])
    log.info("Detected 14 court keypoints")

    # === 3. Player detection + tracking + court-polygon filter ===
    log.info("Tracking players...")
    player_tracker = PlayerTracker(paths["models"]["player_detector"])
    raw_players = player_tracker.detect_frames(
        frames,
        read_from_stub=use_stubs,
        stub_path=paths["stubs"]["player_detections"],
    )
    player_dets = player_tracker.choose_and_filter_players(
        court_keypoints, raw_players
    )

    # === 4. Ball detection + interpolation ===
    log.info("Tracking ball...")
    ball_tracker = BallTracker(paths["models"]["ball_detector"])
    raw_ball = ball_tracker.detect_frames(
        frames,
        read_from_stub=use_stubs,
        stub_path=paths["stubs"]["ball_detections"],
    )
    ball_dets = ball_tracker.interpolate_ball_positions(raw_ball)

    # === 5. Homography ===
    log.info("Computing homography...")
    mini_court = MiniCourt(court_keypoints, frames[0].shape)
    log.info("Reprojection error: %.1f cm",
             mini_court.reprojection_error_m * 100)

    # === 6. Analytics ===
    log.info("Computing analytics...")
    speed_calc = SpeedCalculator(mini_court, fps=meta["fps"])
    speeds = speed_calc.compute_speeds(player_dets)
    distances = speed_calc.compute_distance_covered(player_dets)

    shot_detector = ShotDetector()
    _, shot_counts = shot_detector.detect_shots(ball_dets, player_dets)

    summary = StatsAggregator().aggregate(speeds, distances, shot_counts)
    for tid, s in sorted(summary.items()):
        log.info("Player %d: max=%.1f km/h  avg=%.1f km/h  "
                 "distance=%.1f m  shots=%d",
                 tid, s["max_speed_kmh"], s["avg_speed_kmh"],
                 s["total_distance_m"], int(s["shots"]))

    # === 7. Render annotated frames ===
    log.info("Rendering %d frames...", len(frames))
    out_frames = []
    for i, frame in enumerate(frames):
        f = mini_court.draw_mini_court(frame)

        # Project player feet to mini court
        foots = {tid: get_foot_position(box) for tid, box in player_dets[i].items()}
        f = mini_court.draw_players_on_mini_court(f, foots)

        # Project ball to mini court
        if ball_dets[i]:
            f = mini_court.draw_ball_on_mini_court(f, get_center(ball_dets[i][1]))

        # Draw image-plane boxes and keypoints
        f = player_tracker.draw_bboxes([f], [player_dets[i]])[0]
        f = ball_tracker.draw_bboxes([f], [ball_dets[i]])[0]
        f = court_detector.draw_keypoints(f, court_keypoints)

        out_frames.append(f)

    # Composite the stats box on all frames
    out_frames = StatsDrawer().draw(out_frames, summary)

    # === 8. Save ===
    log.info("Writing output video: %s", paths["output_video"])
    save_video(out_frames, paths["output_video"], fps=meta["fps"])
    log.info("Pipeline complete.")


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        log.error("Failed to load config: %s", e)
        sys.exit(1)

    use_stubs = config.get("use_stubs", True) and not args.no_stubs

    try:
        run_pipeline(config, use_stubs=use_stubs)
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()