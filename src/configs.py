"""Configuration dataclasses for Emergence Lab."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Literal

import yaml


@dataclass
class ArchiveConfig:
    """MAP-Elites behavioral archive configuration."""
    grid_size: int = 100
    """Number of cells along each axis of the 2D behavioral descriptor
    grid. Total capacity = grid_size ** 2. Default 100."""
    enabled: bool = False
    """Whether to maintain a behavioral archive during training.
    When enabled, agents are added to the archive based on their
    behavioral descriptors (movement_entropy, field_write_frequency)."""


class TrainingMode(Enum):
    """Training mode for the training loop.

    - GRADIENT: Standard PPO gradient training (default).
    - EVOLVE: Pure evolution — no gradient updates, only reproduction + mutation.
    - FREEZE_EVOLVE: Alternate between GRADIENT and EVOLVE phases.
    """
    GRADIENT = "gradient"
    EVOLVE = "evolve"
    FREEZE_EVOLVE = "freeze_evolve"


@dataclass
class HiddenFoodConfig:
    """Configuration for hidden food that requires coordination to reveal.

    Hidden food items are invisible until K agents are within distance D.
    Once revealed, they stay visible for a duration then re-hide at a new position.
    This creates tasks that REQUIRE coordination — individuals cannot solve alone.

    Based on Level-Based Foraging (LBF) benchmark (Papoudakis et al. 2021).
    """
    enabled: bool = False
    """Whether hidden food is active. Disabled by default for backward compatibility."""
    num_hidden: int = 3
    """Number of hidden food items on the grid."""
    required_agents: int = 3
    """Minimum number of alive agents within reveal_distance to reveal hidden food."""
    reveal_distance: int = 3
    """Chebyshev distance for agents to be considered 'near' hidden food."""
    reveal_duration: int = 10
    """Number of steps hidden food stays revealed before re-hiding."""
    hidden_food_value_multiplier: float = 5.0
    """Multiplier on food_energy for hidden food (5x default = high-value coordination reward)."""


@dataclass
class EnvConfig:
    """Environment configuration."""
    grid_size: int = 20
    num_agents: int = 8
    num_food: int = 10
    max_steps: int = 500
    observation_radius: int = 5
    food_respawn_prob: float = 0.1
    hidden_food: HiddenFoodConfig = dataclass_field(default_factory=HiddenFoodConfig)
    """Configuration for hidden food requiring coordination to reveal."""


@dataclass
class NestConfig:
    """Nest mechanics for pheromone-based foraging.

    Agents carry food back to a central nest for delivery. Reproduction
    only happens inside the nest area. The compass provides noisy path
    integration back to the nest.
    """
    radius: int = 2
    """Half-width of nest area. radius=2 gives a 5x5 nest."""
    food_sip_fraction: float = 0.05
    """Deprecated: no longer used. Crop refuel fills energy to max on pickup."""
    food_delivery_fraction: float = 0.95
    """Fraction of food_energy given on nest delivery."""
    pickup_reward_fraction: float = 0.1
    """PPO reward fraction on food pickup (split signal)."""
    delivery_reward_fraction: float = 0.9
    """PPO reward fraction on nest delivery (split signal)."""
    compass_noise_rate: float = 0.10
    """Path integration error rate. Noise std = rate * distance / grid_size."""
    # Patch scaling parameters
    patch_radius: int = 2
    """Radius for counting agents at food source (Chebyshev distance)."""
    patch_n_cap: int = 6
    """Max agents for scaling benefit (cap on effective count)."""
    patch_scaling_enabled: bool = False
    """Toggle patch throughput scaling for A/B experiments."""


@dataclass
class FieldConfig:
    """Shared field configuration."""
    num_channels: int = 4
    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    field_obs_radius: int | None = None
    """Radius of the local field patch agents observe. When None, defaults to
    env.observation_radius (backward compatible). Set to a smaller value (e.g., 2)
    to reduce the field observation dimension while keeping food observation range
    unchanged. radius=2 gives 5x5x4=100 field dims instead of 11x11x4=484."""
    channel_diffusion_rates: tuple[float, ...] | None = None
    """Per-channel diffusion rates. When set, overrides diffusion_rate with
    per-channel values. Length must equal num_channels.
    Example: (0.5, 0.01, 0.0, 0.0) for recruitment/territory/reserved/reserved."""
    channel_decay_rates: tuple[float, ...] | None = None
    """Per-channel decay rates. When set, overrides decay_rate with per-channel
    values. Length must equal num_channels.
    Example: (0.05, 0.0001, 0.0, 0.0) for fast recruitment decay, near-permanent territory."""
    field_value_cap: float = 1.0
    """Maximum field value per cell per channel. Values are clipped after writes."""
    territory_write_strength: float = 0.01
    """Passive territory channel (ch1) write strength per step per agent."""
    adaptive_gate: bool = False
    """Enable learnable per-channel gate that modulates field observation influence.
    When True, agents learn WHEN to use the field via a sigmoid gate (0-1 per channel).
    Gate weights are shared (PPO-trained), gate bias is per-agent (evolved)."""
    gate_sparsity_penalty: float = 0.0
    """L1 penalty on gate values to encourage sparse field usage. Added to PPO loss.
    Higher values push gates toward 0 (ignore field). 0.0 = disabled."""
    gate_bias_mutation_std: float = 0.02
    """Mutation standard deviation for per-agent gate bias during reproduction.
    Higher values allow faster evolution of field usage preferences."""


@dataclass
class AgentConfig:
    """Agent neural network configuration."""
    hidden_dims: tuple[int, ...] = (64, 64)
    num_actions: int = 5  # 0=stay,1=up,2=down,3=left,4=right
    activation: Literal["relu", "tanh", "gelu"] = "tanh"
    layer_norm: bool = True
    agent_architecture: Literal["shared", "agent_heads"] = "shared"
    """Network architecture type:
    - "shared": All agents share the same weights (ActorCritic)
    - "agent_heads": Shared encoder + per-agent output heads (AgentSpecificActorCritic)
    """
    agent_embed_dim: int = 0
    """Dimension of learnable agent identity embedding. Each agent gets a unique
    embedding vector that is concatenated to its observation before encoding.
    0 = disabled (no embedding), >0 = embedding dimension. Default 8 when enabled."""


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    seed: int = 42
    total_steps: int = 10_000_000
    num_envs: int = 32
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 256

    # Training mode
    training_mode: TrainingMode = TrainingMode.GRADIENT
    """Training mode: GRADIENT (default PPO), EVOLVE (pure evolution),
    or FREEZE_EVOLVE (alternate between gradient and evolve phases)."""

    # Checkpoint resume
    resume_from: str | None = None
    """Path to checkpoint file to resume training from. Loads shared params
    (and per-agent params if available) before starting the training loop."""

    # PPO specific
    learning_rate: float = 3e-4
    lr_schedule: Literal["constant", "linear"] = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class LogConfig:
    """Logging configuration."""
    wandb: bool = True
    project: str = "emergence-lab"
    log_interval: int = 1000
    video_interval: int = 50000
    save_interval: int = 100000
    checkpoint_dir: str = "checkpoints"
    server: bool = False
    """Start WebSocket server for live dashboard visualization.
    When True, the training loop publishes frames to a local server
    at localhost:8765 that the Svelte dashboard can connect to."""
    server_port: int = 8765
    """Port for the WebSocket dashboard server."""


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    ablation_interval: int = 100000
    emergence_check_interval: int = 10000
    specialization_check_interval: int = 20000


@dataclass
class SpecializationConfig:
    """Configuration for encouraging specialization."""
    diversity_bonus: float = 0.0
    """Reward bonus for weight diversity. Agents with more unique weights
    get a positive reward bonus proportional to their cosine distance from
    the population mean. 0.0 = disabled."""
    niche_pressure: float = 0.0
    """Penalty for identical strategies. Agents whose weights are very
    similar to their nearest neighbor get a negative reward. 0.0 = disabled."""
    layer_mutation_rates: dict[str, float] | None = None
    """Optional per-layer mutation rates. Maps layer name substrings to
    mutation std overrides, e.g. {"Dense_0": 0.02, "Dense_1": 0.005}.
    When None, all layers use the global evolution.mutation_std."""


@dataclass
class FreezeEvolveConfig:
    """Configuration for freeze-evolve training cycles.

    During FREEZE_EVOLVE mode, training alternates between:
    - GRADIENT phase: Normal PPO training for `gradient_steps` steps.
    - EVOLVE phase: No gradient updates for `evolve_steps` steps;
      agents still act (using frozen policy), but only reproduction
      and mutation drive weight changes. Mutation is amplified by
      `evolve_mutation_boost` during this phase.
    """
    gradient_steps: int = 500000
    """Number of agent-steps in each gradient training phase.
    Same units as train.total_steps (num_envs * num_steps * max_agents per iteration)."""
    evolve_steps: int = 100000
    """Number of agent-steps in each pure evolution phase.
    Same units as train.total_steps."""
    evolve_mutation_boost: float = 5.0
    """Multiplier applied to evolution.mutation_std during evolve phases.
    Higher = more aggressive mutation when gradients are frozen.
    Default 5.0 means if mutation_std=0.01, evolve phase uses 0.05."""


@dataclass
class EvolutionConfig:
    """Evolution and reproduction configuration."""
    enabled: bool = True
    starting_energy: int = 100
    energy_per_step: int = 1
    food_energy: int = 50
    max_energy: int = 200
    reproduce_threshold: int = 150
    reproduce_cost: int = 80
    mutation_std: float = 0.01
    max_agents: int = 32
    min_agents: int = 2


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    env: EnvConfig = dataclass_field(default_factory=EnvConfig)
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    agent: AgentConfig = dataclass_field(default_factory=AgentConfig)
    train: TrainConfig = dataclass_field(default_factory=TrainConfig)
    log: LogConfig = dataclass_field(default_factory=LogConfig)
    analysis: AnalysisConfig = dataclass_field(default_factory=AnalysisConfig)
    evolution: EvolutionConfig = dataclass_field(default_factory=EvolutionConfig)
    specialization: SpecializationConfig = dataclass_field(default_factory=SpecializationConfig)
    freeze_evolve: FreezeEvolveConfig = dataclass_field(default_factory=FreezeEvolveConfig)
    archive: ArchiveConfig = dataclass_field(default_factory=ArchiveConfig)
    nest: NestConfig = dataclass_field(default_factory=NestConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        agent_data = data.get("agent", {})
        if "hidden_dims" in agent_data:
            agent_data["hidden_dims"] = tuple(agent_data["hidden_dims"])

        field_data = data.get("field", {})
        if "channel_diffusion_rates" in field_data and field_data["channel_diffusion_rates"] is not None:
            field_data["channel_diffusion_rates"] = tuple(field_data["channel_diffusion_rates"])
        if "channel_decay_rates" in field_data and field_data["channel_decay_rates"] is not None:
            field_data["channel_decay_rates"] = tuple(field_data["channel_decay_rates"])

        return cls(
            env=EnvConfig(**data.get("env", {})),
            field=FieldConfig(**field_data),
            agent=AgentConfig(**agent_data),
            train=TrainConfig(**data.get("train", {})),
            log=LogConfig(**data.get("log", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            evolution=EvolutionConfig(**data.get("evolution", {})),
            specialization=SpecializationConfig(**data.get("specialization", {})),
            freeze_evolve=FreezeEvolveConfig(**data.get("freeze_evolve", {})),
            archive=ArchiveConfig(**data.get("archive", {})),
            nest=NestConfig(**data.get("nest", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import dataclasses

        def _make_yaml_safe(obj: object) -> object:
            """Recursively convert to YAML-safe types."""
            if isinstance(obj, dict):
                return {k: _make_yaml_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_yaml_safe(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        raw = dataclasses.asdict(self)
        with open(path, 'w') as f:
            yaml.dump(_make_yaml_safe(raw), f, default_flow_style=False)


if __name__ == "__main__":
    # Test config creation
    config = Config()
    print(config)
