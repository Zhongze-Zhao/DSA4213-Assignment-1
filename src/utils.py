"""Shared helpers: seeding, config loading, device selection, result saving."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    """Fix Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path) -> dict:
    """Load a yaml config; a `base:` key is loaded first and overridden by this file."""
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        merged = load_config(cfg.pop("base"))
        merged.update(cfg)
        cfg = merged
    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_gpu_memory(fraction: float = 0.7) -> None:
    """Under WSL2 the Windows driver can silently spill CUDA allocations into shared system RAM,
    which froze the whole VM once. Capping PyTorch's share of VRAM makes it raise a normal
    CUDA OOM error instead."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def hardware_info() -> dict:
    info = {"torch": torch.__version__, "device": str(get_device())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    info["cpu_threads"] = os.cpu_count()
    return info


def save_json(obj, path) -> None:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
