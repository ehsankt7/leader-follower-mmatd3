from leader_follower_mmatd3.metrics import average_normalized_reward


def test_anrf_matches_equation_13_without_extra_factor_two():
    assert average_normalized_reward(100.0, 50.0, 10) == 15.0
