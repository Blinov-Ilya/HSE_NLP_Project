import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_artifact_dirs(artifacts_dir: Path) -> Dict[str, Path]:
    paths = {
        "processed": artifacts_dir / "processed",
        "metrics": artifacts_dir / "metrics",
        "predictions": artifacts_dir / "predictions",
        "thresholds": artifacts_dir / "thresholds",
        "plots": artifacts_dir / "plots",
        "models": artifacts_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def get_torch_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    print("Warning: GPU was not detected. Neural and transformer baselines will run on CPU.")
    return torch.device("cpu")


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return cleaned or "label"


def labels_to_pipe(labels) -> str:
    return "|".join(str(label) for label in labels)


@dataclass
class ModelResult:
    model_name: str
    val_scores: np.ndarray
    test_scores: np.ndarray
    raw_scores: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
