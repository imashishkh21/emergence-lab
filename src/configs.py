"""Configuration dataclasses for Emergence Lab."""

from dataclasses import dataclass, field
from typing import Literal, Optional
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    grid_size: int = 20
    num_agents: int = 8
    num_food: int = 10
    max_steps: int = 500
    observation_radius: int = 5


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


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    seed: int = 42
    total_steps: int = 10_000_000
    num_envs: int = 32
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 256
    
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


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    env: EnvConfig = field(default_factory=EnvConfig)
    field: FieldConfig = field(default_factory=FieldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            env=EnvConfig(**data.get("env", {})),
            field=FieldConfig(**data.get("field", {})),
            agent=AgentConfig(**{
                k: tuple(v) if k == "hidden_dims" else v 
                for k, v in data.get("agent", {}).items()
            }),
            train=TrainConfig(**data.get("train", {})),
            log=LogConfig(**data.get("log", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)


if __name__ == "__main__":
    # Test config creation
    config = Config()
    print(config)
