
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(output_dir: str | Path) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger("tgo_iv")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(Path(output_dir) / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)


def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def get_world_info() -> tuple[int, int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return (
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    return 0, 1, 0


def is_main_process() -> bool:
    rank, _, _ = get_world_info()
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_mean(values) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0
