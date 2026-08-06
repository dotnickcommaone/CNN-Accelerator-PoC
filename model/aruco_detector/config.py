from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("model", "dataset", "training", "export")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    input_size = int(config["model"].get("input_size", 0))
    if input_size <= 0 or input_size % 32:
        raise ValueError("model.input_size must be a positive multiple of 32")
    width_mult = float(config["model"].get("width_mult", 0))
    if width_mult <= 0:
        raise ValueError("model.width_mult must be positive")
    for key in ("score_threshold", "nms_iou_threshold"):
        value = float(config["model"].get(key, -1))
        if not 0 <= value <= 1:
            raise ValueError(f"model.{key} must be in [0,1]")
    if int(config["training"].get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(config["training"].get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
