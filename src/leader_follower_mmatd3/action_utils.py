from __future__ import annotations

import numpy as np


def policy_to_robot_commands(policy_action) -> np.ndarray:
    """Map tanh actor outputs to [v, omega] for two robots.

    Each actor outputs values in [-1, 1]^2. Linear velocity is mapped to [0, 1] m/s;
    angular velocity remains in [-1, 1] rad/s, as stated in the paper.
    """
    a = np.asarray(policy_action, dtype=np.float32).reshape(2, 2).copy()
    a[:, 0] = (a[:, 0] + 1.0) / 2.0
    return a.reshape(-1)
