from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class NetworkConfig:
    observation_dim_per_agent: int = 64
    action_dim_per_agent: int = 2
    hidden_sizes: Tuple[int, int] = (650, 500)
    n_agents: int = 2

    @property
    def joint_observation_dim(self) -> int:
        return self.observation_dim_per_agent * self.n_agents

    @property
    def joint_action_dim(self) -> int:
        return self.action_dim_per_agent * self.n_agents


@dataclass(frozen=True)
class ExplorationConfig:
    gaussian_std_start: float = 0.5
    gaussian_std_end: float = 0.1
    gaussian_decay_steps: int = 500_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.005
    epsilon_decay: float = 0.9999


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 50_000
    max_steps_per_episode: int = 500
    batch_size: int = 128
    replay_capacity: int = 1_000_000
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.9999
    tau: float = 0.005
    policy_update_frequency: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    learning_starts_episodes: int = 50
    checkpoint_interval_episodes: int = 500
    seed: int = 0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "simple"
    launch_file: str = "simple.launch"
    world_file: str = "simple.world"
    random_boxes: int = 4
    arena_half_width: float = 8.0
    follower_spawn_min_m: float = 1.0
    follower_spawn_max_m: float = 3.0
    goal_min_robot_distance_m: float = 1.5
    obstacle_min_robot_or_goal_distance_m: float = 2.0
    goal_reached_distance_m: float = 0.4
    collision_distance_m: float = 0.35


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = "mmatd3"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(
            algorithm=str(data.get("algorithm", "mmatd3")).lower(),
            network=NetworkConfig(**data.get("network", {})),
            exploration=ExplorationConfig(**data.get("exploration", {})),
            training=TrainingConfig(**data.get("training", {})),
            scenario=ScenarioConfig(**data.get("scenario", {})),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(json.load(handle))
