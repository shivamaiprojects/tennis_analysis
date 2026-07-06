"""
Video I/O utilities for the tennis analysis pipeline.

Two reading strategies are provided:
- read_video(): loads all frames into memory. Fast to iterate, uses RAM.
- read_video_lazy(): yields frames one at a time. Slower per-frame, low RAM.

Use read_video() for clips under ~2 minutes. Use read_video_lazy() for longer
videos or when running on a memory-constrained machine.
"""
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np


def read_video(video_path: str | Path) -> List[np.ndarray]:
    """Read all frames of a video into memory as a list of BGR arrays.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of numpy arrays, each of shape (H, W, 3) with dtype uint8, in BGR order.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def read_video_lazy(video_path: str | Path) -> Generator[np.ndarray, None, None]:
    """Yield frames one at a time. Use for long videos to avoid running out of RAM.

    Args:
        video_path: Path to the input video file.

    Yields:
        A single frame at a time as a numpy array in BGR order.

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_metadata(video_path: str | Path) -> dict:
    """Return video metadata: FPS, frame count, width, height.

    Metadata is essential for downstream analytics — we need FPS to convert
    per-frame motion into real-world speed (meters per second, km/h).

    Args:
        video_path: Path to the input video file.

    Returns:
        Dictionary with keys: fps (float), frame_count (int), width (int), height (int).

    Raises:
        FileNotFoundError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    cap.release()
    return metadata


def save_video(
    frames: List[np.ndarray],
    output_path: str | Path,
    fps: float = 24.0,
    codec: str = "mp4v",
) -> None:
    """Write a list of frames to a video file.

    Args:
        frames: List of numpy arrays, each of shape (H, W, 3) in BGR order.
        output_path: Where to save the video. Parent folder is created if missing.
        fps: Frames per second for the output. Match the input video's FPS.
        codec: Four-character codec identifier. 'mp4v' works for .mp4 files.

    Raises:
        ValueError: If frames is empty.
        RuntimeError: If the VideoWriter fails to open (usually a codec problem).
    """
    if not frames:
        raise ValueError("Cannot save an empty frame list.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()