from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .action_utils import policy_to_robot_commands
from .algorithm import MultiAgentContinuousControl
from .config import ExperimentConfig
from .metrics import average_normalized_reward
from .replay_buffer import ReplayBuffer


@dataclass
class RunningOutcomeStats:
    episodes: int = 0
    collisions: int = 0
    successes: int = 0

    def update(self, collision: bool, success: bool) -> None:
        self.episodes += 1
        self.collisions += int(collision)
        self.successes += int(success)

    @property
    def collision_percentage(self) -> float:
        return 100.0 * self.collisions / max(1, self.episodes)

    @property
    def success_percentage(self) -> float:
        return 100.0 * self.successes / max(1, self.episodes)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    config: ExperimentConfig,
    env,
    output_dir: str | Path,
    initialize_from: Optional[str | Path] = None,
) -> None:
    """Train one scenario and save metadata-rich checkpoints plus episode metrics.

    Optimization begins after the configured warm-up period and then performs replay
    updates using transitions collected from the environment.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.training.seed)
    agent = MultiAgentContinuousControl(config)
    if initialize_from:
        metadata = agent.load(initialize_from, load_optimizers=False)
        print(f"Initialized from {initialize_from} ({metadata.get('algorithm', 'unknown')}).")

    buffer = ReplayBuffer(
        config.training.replay_capacity,
        config.network.joint_observation_dim,
        config.network.joint_action_dim,
        config.network.n_agents,
        config.training.seed,
    )
    stats = RunningOutcomeStats()
    csv_path = output_dir / "episode_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "episode",
                "steps",
                "leader_return",
                "follower_return",
                "anrf",
                "collision_percentage",
                "success_percentage",
                "gaussian_std",
                "epsilon",
            ],
        )
        writer.writeheader()

        for episode in range(1, config.training.episodes + 1):
            state = np.asarray(env.reset(), dtype=np.float32)
            returns = np.zeros(config.network.n_agents, dtype=np.float64)
            episode_steps = 0
            episode_collision = False
            leader_success = False

            for step in range(config.training.max_steps_per_episode):
                policy_action = agent.act(state, explore=True)
                robot_command = policy_to_robot_commands(policy_action)
                next_state, reward, done, targets, collisions = env.step(robot_command)
                next_state = np.asarray(next_state, dtype=np.float32)
                reward = np.asarray(reward, dtype=np.float32)

                timed_out = step + 1 >= config.training.max_steps_per_episode
                terminal = bool(done or timed_out)
                # The replay buffer stores the normalized actor-domain action, not the mapped velocity command.
                buffer.add(state, policy_action, reward, terminal, next_state)
                returns += reward
                state = next_state
                episode_steps = step + 1
                episode_collision = episode_collision or bool(np.any(collisions))
                leader_success = leader_success or bool(np.asarray(targets)[0])
                if terminal:
                    break

            if episode > config.training.learning_starts_episodes and len(buffer) >= config.training.batch_size:
                for _ in range(episode_steps):
                    batch = buffer.sample(config.training.batch_size, agent.device)
                    agent.update(batch)

            stats.update(collision=episode_collision, success=leader_success)
            sigma, epsilon = agent.exploration_values()
            anrf = average_normalized_reward(returns[0], returns[1], episode_steps)
            writer.writerow(
                {
                    "episode": episode,
                    "steps": episode_steps,
                    "leader_return": float(returns[0]),
                    "follower_return": float(returns[1]),
                    "anrf": anrf,
                    "collision_percentage": stats.collision_percentage,
                    "success_percentage": stats.success_percentage,
                    "gaussian_std": sigma,
                    "epsilon": epsilon,
                }
            )
            stream.flush()

            if episode % 100 == 0 or episode == 1:
                print(
                    f"episode={episode}/{config.training.episodes} "
                    f"ANRF={anrf:.3f} CP={stats.collision_percentage:.2f}% "
                    f"success={stats.success_percentage:.2f}%"
                )

            if episode % config.training.checkpoint_interval_episodes == 0:
                agent.save(
                    checkpoint_dir / f"{config.algorithm}_{config.scenario.name}_ep{episode:06d}.pt",
                    episode=episode,
                    scenario=config.scenario.name,
                )

    agent.save(
        checkpoint_dir / f"{config.algorithm}_{config.scenario.name}_final.pt",
        episode=config.training.episodes,
        scenario=config.scenario.name,
    )
