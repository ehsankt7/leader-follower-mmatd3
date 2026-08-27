from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ExperimentConfig
from .networks import Actor, SingleCritic, TwinCritic
from .replay_buffer import ReplayBatch


SUPPORTED_ALGORITHMS = {"maddpg", "matd3", "mmatd3"}


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


class MultiAgentContinuousControl:
    """CTDE implementation of MADDPG, MATD3, and the proposed M-MATD3.

    For M-MATD3, each actor is optimized using the average of its two centralized
    critics. MATD3 uses Q1 for the actor update. Target values for both TD3 variants
    use the minimum of two target critics.
    """

    def __init__(self, config: ExperimentConfig, device: Optional[str] = None) -> None:
        if config.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{config.algorithm}'.")
        self.config = config
        self.algorithm = config.algorithm
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        n = config.network.n_agents
        obs_dim = config.network.observation_dim_per_agent
        act_dim = config.network.action_dim_per_agent
        joint_obs = config.network.joint_observation_dim
        joint_act = config.network.joint_action_dim

        self.actors = nn.ModuleList([Actor(obs_dim, act_dim, config.network.hidden_sizes) for _ in range(n)]).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)

        critic_cls = SingleCritic if self.algorithm == "maddpg" else TwinCritic
        self.critics = nn.ModuleList([critic_cls(joint_obs, joint_act, config.network.hidden_sizes) for _ in range(n)]).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)

        tr = config.training
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=tr.actor_learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=tr.critic_learning_rate)
            for critic in self.critics
        ]
        self.update_count = 0
        self.environment_step = 0
        self.rng = np.random.default_rng(tr.seed)

    def _split_states(self, joint_states: torch.Tensor) -> List[torch.Tensor]:
        d = self.config.network.observation_dim_per_agent
        return [joint_states[:, i * d : (i + 1) * d] for i in range(self.config.network.n_agents)]

    def exploration_values(self, step: Optional[int] = None) -> tuple[float, float]:
        step = self.environment_step if step is None else int(step)
        ex = self.config.exploration
        fraction = min(1.0, step / max(1, ex.gaussian_decay_steps))
        sigma = ex.gaussian_std_start + fraction * (ex.gaussian_std_end - ex.gaussian_std_start)
        epsilon = max(ex.epsilon_end, ex.epsilon_start * (ex.epsilon_decay ** step))
        return float(sigma), float(epsilon)

    @torch.no_grad()
    def act(self, joint_state: np.ndarray, explore: bool = False) -> np.ndarray:
        joint_state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        obs_dim = self.config.network.observation_dim_per_agent
        actions: List[np.ndarray] = []
        sigma, epsilon = self.exploration_values()

        for i, actor in enumerate(self.actors):
            if explore and self.rng.random() < epsilon:
                action = self.rng.uniform(-1.0, 1.0, size=self.config.network.action_dim_per_agent)
            else:
                state_i = torch.as_tensor(
                    joint_state[i * obs_dim : (i + 1) * obs_dim],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                action = actor(state_i).squeeze(0).cpu().numpy()
                if explore:
                    action = action + self.rng.normal(0.0, sigma, size=action.shape)
            actions.append(np.clip(action, -1.0, 1.0))

        self.environment_step += int(explore)
        return np.concatenate(actions).astype(np.float32)

    def update(self, batch: ReplayBatch) -> Dict[str, float]:
        tr = self.config.training
        split_next = self._split_states(batch.next_states)

        with torch.no_grad():
            next_actions = torch.cat(
                [actor(state_i) for actor, state_i in zip(self.target_actors, split_next)],
                dim=1,
            )
            if self.algorithm in {"matd3", "mmatd3"}:
                noise = torch.randn_like(next_actions) * tr.target_policy_noise
                noise = noise.clamp(-tr.target_noise_clip, tr.target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)

        critic_losses: List[float] = []
        actor_losses: List[float] = []
        update_actor = self.update_count % tr.policy_update_frequency == 0

        for agent_idx in range(self.config.network.n_agents):
            critic = self.critics[agent_idx]
            target_critic = self.target_critics[agent_idx]
            with torch.no_grad():
                if self.algorithm == "maddpg":
                    next_q = target_critic(batch.next_states, next_actions)
                else:
                    q1_t, q2_t = target_critic(batch.next_states, next_actions)
                    next_q = torch.minimum(q1_t, q2_t)
                target_q = batch.rewards[:, agent_idx : agent_idx + 1] + (
                    (1.0 - batch.dones) * tr.gamma * next_q
                )

            if self.algorithm == "maddpg":
                current_q = critic(batch.states, batch.actions)
                critic_loss = F.mse_loss(current_q, target_q)
            else:
                current_q1, current_q2 = critic(batch.states, batch.actions)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizers[agent_idx].zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            if update_actor:
                d = self.config.network.action_dim_per_agent
                policy_actions = batch.actions.detach().clone()
                own_state = self._split_states(batch.states)[agent_idx]
                own_action = self.actors[agent_idx](own_state)
                policy_actions[:, agent_idx * d : (agent_idx + 1) * d] = own_action

                if self.algorithm == "maddpg":
                    policy_q = critic(batch.states, policy_actions)
                else:
                    q1_pi, q2_pi = critic(batch.states, policy_actions)
                    # Proposed modification: both critics contribute equally.
                    policy_q = (q1_pi + q2_pi) / 2.0 if self.algorithm == "mmatd3" else q1_pi

                actor_loss = -policy_q.mean()
                self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizers[agent_idx].step()
                actor_losses.append(float(actor_loss.detach().cpu()))

                _soft_update(self.actors[agent_idx], self.target_actors[agent_idx], tr.tau)
                _soft_update(self.critics[agent_idx], self.target_critics[agent_idx], tr.tau)

        self.update_count += 1
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
        }

    def checkpoint_metadata(self) -> Dict[str, object]:
        config_dict = asdict(self.config)
        config_json = json.dumps(config_dict, sort_keys=True)
        return {
            "algorithm": self.algorithm,
            "config": config_dict,
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "update_count": self.update_count,
            "environment_step": self.environment_step,
            "torch_version": torch.__version__,
        }

    def save(self, path: str | Path, **extra_metadata) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {**self.checkpoint_metadata(), **extra_metadata},
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
        }
        torch.save(payload, path)

    def load(self, path: str | Path, load_optimizers: bool = True) -> Dict[str, object]:
        payload = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(payload["actors"])
        self.target_actors.load_state_dict(payload.get("target_actors", payload["actors"]))
        self.critics.load_state_dict(payload["critics"])
        self.target_critics.load_state_dict(payload.get("target_critics", payload["critics"]))
        if load_optimizers:
            for optimizer, state in zip(self.actor_optimizers, payload.get("actor_optimizers", [])):
                optimizer.load_state_dict(state)
            for optimizer, state in zip(self.critic_optimizers, payload.get("critic_optimizers", [])):
                optimizer.load_state_dict(state)
        metadata = payload.get("metadata", {})
        self.update_count = int(metadata.get("update_count", 0))
        self.environment_step = int(metadata.get("environment_step", 0))
        return metadata
