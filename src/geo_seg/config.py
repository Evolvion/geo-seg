from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    root: Optional[str] = None
    image_size: int = 64
    num_classes: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-3
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "unet"


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    seed: int = 42
    out_dir: str = "runs/debug1"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


default_cfg = Config()
