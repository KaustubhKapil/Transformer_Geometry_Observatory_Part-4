
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    train_dir: str = ""
    val_dir: str = ""
    class_subset_file: Optional[str] = None
    image_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True
    analysis_set_size: int = 1000
    analysis_seed: int = 2025
    trajectory_image_index: int = 0
    topology_image_index: int = 0


@dataclass
class ModelConfig:
    name: str = "vit_small_patch16_224"
    pretrained: bool = True
    num_classes: int = 100
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    opt: str = "adamw"
    momentum: float = 0.9
    label_smoothing: float = 0.1
    clip_grad: float = 1.0
    amp: bool = True


@dataclass
class AnalysisConfig:
    snapshot_epochs: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    topology_maxdim: int = 2
    jacobian_ridge: float = 1e-6
    jacobian_heterogeneity_size: int = 32
    topology_eps_grid: int = 64
    plot_layers: list[int] = field(default_factory=lambda: [0, 3, 6, 9, 12])
    save_raw_diagrams: bool = True
    save_raw_jacobians: bool = False


@dataclass
class CheckpointConfig:
    save_best: bool = True
    save_last: bool = True


@dataclass
class Config:
    seed: int = 1337
    device: str = "cuda"
    output_dir: str = "results_tgo_iv"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Config.from_dict(raw or {})

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        cfg = Config()
        for k, v in raw.items():
            if k in {"data", "model", "train", "analysis", "checkpoint"}:
                section = getattr(cfg, k)
                for sk, sv in (v or {}).items():
                    setattr(section, sk, sv)
            else:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "output_dir": self.output_dir,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "train": self.train.__dict__,
            "analysis": self.analysis.__dict__,
            "checkpoint": self.checkpoint.__dict__,
        }
