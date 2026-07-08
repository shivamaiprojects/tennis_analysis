"""
Tennis Analysis — main entry point.

Loads config, runs the full pipeline, saves the annotated video.
Currently a scaffold; later modules will fill in the pipeline body.
"""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config file and return it as a nested dict.

    Args:
        config_path: Path to the YAML config, relative to CWD.

    Returns:
        Nested dict representation of the YAML.

    Raises:
        FileNotFoundError: If the config doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a dict, got {type(config)}")

    return config


def main() -> None:
    """Entry point for the tennis analysis pipeline."""
    print("=" * 60)
    print("Tennis Analysis Pipeline")
    print("=" * 60)

    config = load_config()

    print(f"\nInput video:      {config['paths']['input_video']}")
    print(f"Output video:     {config['paths']['output_video']}")
    print(f"Ball model:       {config['paths']['models']['ball_detector']}")
    print(f"Court model:      {config['paths']['models']['court_keypoints']}")
    print(f"Player model:     {config['paths']['models']['player_detector']}")
    print(f"Use stubs:        {config['use_stubs']}")

    print("\n[Skeleton is working. Full pipeline body arrives in later modules.]")


if __name__ == "__main__":
    main()