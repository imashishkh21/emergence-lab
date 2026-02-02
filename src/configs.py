"""Configuration dataclasses for Emergence Lab."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Literal

import yaml


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
class EnvConfig:
    """Environment configuration."""
    grid_size: int = 20
    num_agents: int = 8
    num_food: int = 10
    max_steps: int = 500
    observation_radius: int = 5
    food_respawn_prob: float = 0.1


@dataclass
class FieldConfig:
    """Shared field configuration."""
    num_channels: int = 4
    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    write_strength: float = 1.0


@dataclass
class AgentConfig:
    """Agent neural network configuration."""
    hidden_dims: tuple[int, ...] = (64, 64)
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
    gradient_steps: int = 10000
    """Number of environment steps in each gradient training phase."""
    evolve_steps: int = 1000
    """Number of environment steps in each pure evolution phase."""
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

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        agent_data = data.get("agent", {})
        if "hidden_dims" in agent_data:
            agent_data["hidden_dims"] = tuple(agent_data["hidden_dims"])

        return cls(
            env=EnvConfig(**data.get("env", {})),
            field=FieldConfig(**data.get("field", {})),
            agent=AgentConfig(**agent_data),
            train=TrainConfig(**data.get("train", {})),
            log=LogConfig(**data.get("log", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            evolution=EvolutionConfig(**data.get("evolution", {})),
            specialization=SpecializationConfig(**data.get("specialization", {})),
            freeze_evolve=FreezeEvolveConfig(**data.get("freeze_evolve", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import dataclasses

        def to_dict(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)


if __name__ == "__main__":
    # Test config creation
    config = Config()
    print(config)
