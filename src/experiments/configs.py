"""Environment configuration presets for experiments.

Three configurations for Phase 5 experiments:
1. standard_config: Default settings, balanced for training and evaluation
2. hidden_resources_config: Requires coordination to reveal hidden food
3. food_scarcity_config: Scarce resources (num_food=5) for competition

Each function returns a Config object that can be further customized.
"""

from dataclasses import replace

from src.configs import Config


def standard_config(base_config: Config | None = None) -> Config:
    """Create the standard environment configuration.

    This is the default balanced configuration suitable for most experiments.
    Good balance between exploration and exploitation challenges.

    Settings:
        - grid_size: 20
        - num_agents: 8
        - num_food: 10
        - max_steps: 500
        - Evolution enabled with default parameters

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config with standard settings.
    """
    if base_config is None:
        base_config = Config()

    # Standard environment settings
    env_config = replace(
        base_config.env,
        grid_size=20,
        num_agents=8,
        num_food=10,
        max_steps=500,
        observation_radius=5,
    )

    # Standard field settings
    field_config = replace(
        base_config.field,
        num_channels=4,
        diffusion_rate=0.1,
        decay_rate=0.05,
    )

    # Enable evolution with balanced parameters
    evolution_config = replace(
        base_config.evolution,
        enabled=True,
        starting_energy=100,
        energy_per_step=1,
        food_energy=50,
        max_energy=200,
        reproduce_threshold=150,
        reproduce_cost=80,
        mutation_std=0.01,
        max_agents=32,
        min_agents=2,
    )

    return replace(
        base_config,
        env=env_config,
        field=field_config,
        evolution=evolution_config,
    )


def hidden_resources_config(base_config: Config | None = None) -> Config:
    """Create the hidden resources environment configuration.

    This configuration requires coordination between agents to reveal
    hidden high-value food. Tests emergence of collective behavior.

    Hidden food parameters (when US-010 is implemented):
        - K=3 agents required within D=3 distance to reveal
        - 5x value multiplier for hidden food
        - 10 step reveal duration

    Current settings (before US-010):
        - Standard config with more agents for coordination
        - Higher food count for sustainability
        - Longer episodes for coordination to emerge

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config optimized for coordination tasks.
    """
    if base_config is None:
        base_config = Config()

    # Start from standard config
    config = standard_config(base_config)

    # Adjust for coordination tasks
    env_config = replace(
        config.env,
        num_agents=12,  # More agents for coordination
        num_food=15,    # More regular food for sustainability
        max_steps=1000,  # Longer episodes for coordination to emerge
    )

    # Higher max agents for population dynamics
    evolution_config = replace(
        config.evolution,
        max_agents=48,  # Allow larger populations
    )

    # Note: Hidden food config will be added by US-010
    # When that's implemented, this will also set:
    # hidden_food = HiddenFoodConfig(
    #     enabled=True,
    #     num_hidden=3,
    #     required_agents=3,
    #     reveal_distance=3,
    #     reveal_duration=10,
    #     hidden_food_value_multiplier=5.0,
    # )

    return replace(
        config,
        env=env_config,
        evolution=evolution_config,
    )


def food_scarcity_config(base_config: Config | None = None) -> Config:
    """Create the food scarcity environment configuration.

    This configuration creates resource scarcity (num_food=5) to test
    competition dynamics and efficiency under pressure.

    Settings:
        - num_food: 5 (half of standard)
        - Slightly smaller grid for increased density
        - Faster food respawn to maintain some sustainability

    Args:
        base_config: Optional base config to modify. If None, uses default Config.

    Returns:
        Config with scarce resources.
    """
    if base_config is None:
        base_config = Config()

    # Start from standard config
    config = standard_config(base_config)

    # Scarce resources
    env_config = replace(
        config.env,
        num_food=5,            # Half the standard amount
        grid_size=16,          # Smaller grid for higher density
        food_respawn_prob=0.15,  # Faster respawn to maintain viability
        max_steps=500,
    )

    # Adjust evolution for scarce environment
    evolution_config = replace(
        config.evolution,
        food_energy=75,          # Higher reward per food (compensate for scarcity)
        reproduce_threshold=180,  # Harder to reproduce
        max_agents=24,           # Lower population cap
    )

    return replace(
        config,
        env=env_config,
        evolution=evolution_config,
    )


def get_env_config(config_name: str, base_config: Config | None = None) -> Config:
    """Get environment config by name.

    Args:
        config_name: One of "standard", "hidden_resources", or "food_scarcity".
        base_config: Optional base config to modify.

    Returns:
        Config for the specified environment.

    Raises:
        ValueError: If config_name is not recognized.
    """
    config_map = {
        "standard": standard_config,
        "hidden_resources": hidden_resources_config,
        "food_scarcity": food_scarcity_config,
    }

    if config_name not in config_map:
        valid_names = list(config_map.keys())
        raise ValueError(
            f"Unknown config name '{config_name}'. Valid names: {valid_names}"
        )

    return config_map[config_name](base_config)


def list_env_configs() -> list[str]:
    """List available environment configuration names.

    Returns:
        List of valid config names.
    """
    return ["standard", "hidden_resources", "food_scarcity"]
