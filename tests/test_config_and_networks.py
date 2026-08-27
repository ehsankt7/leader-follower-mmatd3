from pathlib import Path

import torch

from leader_follower_mmatd3.config import load_config
from leader_follower_mmatd3.networks import Actor, TwinCritic


def test_paper_config_values():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "simple_mmatd3.json")
    assert cfg.network.observation_dim_per_agent == 64
    assert cfg.training.batch_size == 128
    assert cfg.training.actor_learning_rate == 1e-4
    assert cfg.training.critic_learning_rate == 1e-3
    assert cfg.training.gamma == 0.9999
    assert cfg.training.tau == 0.005
    assert cfg.training.policy_update_frequency == 2
    assert cfg.training.max_steps_per_episode == 500
    assert cfg.training.episodes == 50_000
    assert cfg.scenario.random_boxes == 4


def test_network_shapes():
    actor = Actor()
    critic = TwinCritic()
    states_agent = torch.zeros(3, 64)
    states_joint = torch.zeros(3, 128)
    actions_joint = torch.zeros(3, 4)
    assert actor(states_agent).shape == (3, 2)
    q1, q2 = critic(states_joint, actions_joint)
    assert q1.shape == (3, 1)
    assert q2.shape == (3, 1)
