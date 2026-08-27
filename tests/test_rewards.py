import math

from leader_follower_mmatd3.rewards import (
    follower_distance_reward,
    follower_heading_reward,
    leader_reward,
)


def test_leader_goal_and_collision_terminal_rewards():
    assert leader_reward(
        reached_goal=True,
        collision=False,
        action=[0.0, 0.0],
        min_obstacle_distance_m=2.0,
        distance_to_goal_m=1.0,
        previous_distance_to_goal_m=1.0,
    ) == 800.0
    assert leader_reward(
        reached_goal=False,
        collision=True,
        action=[0.0, 0.0],
        min_obstacle_distance_m=2.0,
        distance_to_goal_m=1.0,
        previous_distance_to_goal_m=1.0,
    ) == -500.0


def test_leader_distance_reward_rewards_progress():
    closer = leader_reward(
        reached_goal=False,
        collision=False,
        action=[0.0, 0.0],
        min_obstacle_distance_m=2.0,
        distance_to_goal_m=4.0,
        previous_distance_to_goal_m=5.0,
    )
    farther = leader_reward(
        reached_goal=False,
        collision=False,
        action=[0.0, 0.0],
        min_obstacle_distance_m=2.0,
        distance_to_goal_m=6.0,
        previous_distance_to_goal_m=5.0,
    )
    assert closer > 0
    assert farther < 0
    assert math.isclose(closer, 199.0)
    assert math.isclose(farther, -201.0)


def test_follower_distance_reward_regions():
    assert math.isclose(follower_distance_reward(0.75), -0.2 * math.sqrt(0.25))
    assert math.isclose(follower_distance_reward(2.0), 10.0)
    assert math.isclose(follower_distance_reward(4.0), -0.4)


def test_follower_heading_reward_threshold():
    assert follower_heading_reward(5.0) == 4.0
    assert math.isclose(follower_heading_reward(20.0), -0.4)
