from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"

ALL_MODELS = [
    "most_frequent",
    "tfidf_logreg",
    "tfidf_linearsvm",
    "textcnn",
    "bilstm_attention",
    "distilbert",
    "roberta",
]

QUICK_MODELS = ["most_frequent", "tfidf_logreg", "textcnn", "distilbert"]

TRANSFORMER_CHECKPOINTS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}


@dataclass
class BaselineConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTEFACTS_DIR
    data_path: Optional[Path] = None

    seed: int = 42
    quick: bool = False
    max_samples: Optional[int] = None
    top_genres: int = 15
    min_overview_chars: int = 30

    tfidf_max_features: int = 100_000
    tfidf_min_df: int = 2
    sklearn_max_iter: int = 1_000

    max_vocab_size: int = 50_000
    min_token_freq: int = 2
    max_seq_len: int = 256
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3

    batch_size: int = 16
    epochs: int = 3
    patience: int = 2
    learning_rate: float = 1e-3
    transformer_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    device: str = "auto"

    def path(self, *parts: str) -> Path:
        return self.artifacts_dir.joinpath(*parts)

