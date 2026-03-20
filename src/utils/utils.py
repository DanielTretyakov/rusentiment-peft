import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости результатов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str) -> dict:
    """Загружает YAML-конфиг и возвращает его как словарь."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Возвращает наилучшее доступное устройство (GPU или CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU не найден, используется CPU")
    return device


def count_parameters(model) -> dict:
    """Подсчитывает общее и обучаемое количество параметров модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
    }


def ensure_dir(path: str) -> Path:
    """Создаёт директорию, если она не существует."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
