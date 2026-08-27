#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leader_follower_mmatd3.config import load_config
from leader_follower_mmatd3.env_factory import make_gazebo_env
from leader_follower_mmatd3.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leader-follower DRL experiment.")
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--initialize-from",
        default=None,
        help="Optional checkpoint used to initialize a new scenario (e.g., complex from simple).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_gazebo_env(config)
    train(config, env, args.output, initialize_from=args.initialize_from)


if __name__ == "__main__":
    main()
