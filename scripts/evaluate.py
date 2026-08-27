#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leader_follower_mmatd3.action_utils import policy_to_robot_commands
from leader_follower_mmatd3.algorithm import MultiAgentContinuousControl
from leader_follower_mmatd3.config import load_config
from leader_follower_mmatd3.env_factory import make_gazebo_env
from leader_follower_mmatd3.metrics import average_normalized_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved actor/critic checkpoint without exploration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="evaluation.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_gazebo_env(config)
    agent = MultiAgentContinuousControl(config)
    metadata = agent.load(args.checkpoint, load_optimizers=False)
    print("Checkpoint metadata:", metadata)

    rows = []
    collisions_total = 0
    successes_total = 0
    for episode in range(1, args.episodes + 1):
        state = np.asarray(env.reset(), dtype=np.float32)
        returns = np.zeros(2, dtype=np.float64)
        collided = False
        success = False
        steps = 0
        for step in range(config.training.max_steps_per_episode):
            policy_action = agent.act(state, explore=False)
            next_state, reward, done, targets, collisions = env.step(policy_to_robot_commands(policy_action))
            returns += np.asarray(reward)
            state = np.asarray(next_state, dtype=np.float32)
            collided = collided or bool(np.any(collisions))
            success = success or bool(np.asarray(targets)[0])
            steps = step + 1
            if done:
                break
        collisions_total += int(collided)
        successes_total += int(success)
        rows.append(
            {
                "episode": episode,
                "steps": steps,
                "leader_return": returns[0],
                "follower_return": returns[1],
                "anrf": average_normalized_reward(returns[0], returns[1], steps),
                "collision": int(collided),
                "success": int(success),
            }
        )

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collision percentage: {100*collisions_total/args.episodes:.2f}%")
    print(f"Success rate: {100*successes_total/args.episodes:.2f}%")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
