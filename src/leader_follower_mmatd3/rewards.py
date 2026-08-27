from __future__ import annotations

import math
from typing import Sequence


def obstacle_penalty(min_obstacle_distance_m: float) -> float:
    if min_obstacle_distance_m < 1.0:
        return -2.0 * math.sqrt(max(0.0, 1.0 - min_obstacle_distance_m))
    return 0.0


def turning_reward(action: Sequence[float]) -> float:
    linear_velocity, angular_velocity = float(action[0]), float(action[1])
    return linear_velocity - abs(angular_velocity)


def leader_reward(
    *,
    reached_goal: bool,
    collision: bool,
    action: Sequence[float],
    min_obstacle_distance_m: float,
    distance_to_goal_m: float,
    previous_distance_to_goal_m: float,
) -> float:
    """Leader reward used for goal seeking, collision avoidance, and smooth motion."""
    if reached_goal:
        return 800.0
    if collision:
        return -500.0

    distance_reward = -200.0 * (distance_to_goal_m - previous_distance_to_goal_m)
    timestep_reward = -1.0
    return (
        distance_reward
        + timestep_reward
        + turning_reward(action)
        + obstacle_penalty(min_obstacle_distance_m)
    )


def follower_distance_reward(distance_to_leader_m: float) -> float:
    """Follower spacing reward for maintaining a 1-3 m leader distance."""
    if distance_to_leader_m < 1.0:
        return -0.2 * math.sqrt(max(0.0, 1.0 - distance_to_leader_m))
    if distance_to_leader_m > 3.0:
        return -0.1 * distance_to_leader_m
    return 10.0


def follower_heading_reward(heading_difference_deg: float) -> float:
    """Follower heading reward based on the absolute leader-relative heading error."""
    heading_difference_deg = abs(float(heading_difference_deg))
    if heading_difference_deg > 10.0:
        return -0.02 * heading_difference_deg
    return 4.0


def follower_reward(
    *,
    collision: bool,
    action: Sequence[float],
    min_obstacle_distance_m: float,
    distance_to_leader_m: float,
    heading_difference_deg: float,
) -> float:
    """Follower reward for tracking, obstacle avoidance, and smooth motion."""
    if collision:
        return -500.0

    return (
        follower_distance_reward(distance_to_leader_m)
        + follower_heading_reward(heading_difference_deg)
        + turning_reward(action)
        + obstacle_penalty(min_obstacle_distance_m)
    )
