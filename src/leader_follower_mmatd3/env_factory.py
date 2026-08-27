from __future__ import annotations

from .config import ExperimentConfig


def make_gazebo_env(config: ExperimentConfig):
    """Create the ROS/Gazebo environment for the selected scenario.

    ROS imports are intentionally delayed so the neural-network/reward unit tests can run
    on machines that do not have ROS installed.
    """
    env_dim = 60 * config.network.n_agents
    if config.scenario.name == "simple":
        from .ros_env_simple import GazeboEnv

        return GazeboEnv(config.scenario.launch_file, env_dim)
    if config.scenario.name == "complex":
        from .ros_env_complex import GazeboEnv

        return GazeboEnv(config.scenario.launch_file, env_dim)
    raise ValueError(f"Unsupported scenario: {config.scenario.name}")
