from __future__ import annotations


def average_normalized_reward(leader_return: float, follower_return: float, steps: int) -> float:
    """Paper Eq. (13): (leader cumulative reward + follower cumulative reward) / episode steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return (float(leader_return) + float(follower_return)) / float(steps)
