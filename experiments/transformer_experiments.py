from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from baselines.config import BaselineConfig
from baselines.data_utils import DataBundle
from baselines.thresholding import apply_thresholds, tune_thresholds
from baselines.utils import ModelResult, get_torch_device, save_json


TRANSFORMER_EXPERIMENT_CHECKPOINTS = {
    "distilbert": "distilbert-base-uncased",
}
DEFAULT_TRANSFORMER_LEARNING_RATE = 2e-5
DEFAULT_TRANSFORMER_WEIGHT_DECAY = 0.01


class TransformerMovieDataset(Dataset):
    def __init__(self, texts, labels: np.ndarray, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


def _model_dir(config: BaselineConfig, model_name: str) -> Path:
    path = config.path("models", model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_probabilities(model, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels", None)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            scores.append(torch.sigmoid(logits).cpu().numpy())
            if labels is not None:
                batch["labels"] = labels
    return np.vstack(scores)


def train_transformer_experiment(model_name: str, bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    if model_name not in TRANSFORMER_EXPERIMENT_CHECKPOINTS:
        raise ValueError(f"Unknown transformer experiment: {model_name}")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to run transformer experiments.") from exc

    checkpoint = TRANSFORMER_EXPERIMENT_CHECKPOINTS[model_name]
    model_dir = _model_dir(config, model_name)
    device = get_torch_device(config.device)
    if device.type == "cpu":
        print(f"Warning: running {model_name} on CPU may be slow. Use a small sample for debugging.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(bundle.label_names),
            problem_type="multi_label_classification",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Hugging Face checkpoint '{checkpoint}'. "
            "Install transformers and ensure the model is cached or network access is available."
        ) from exc

    model.to(device)
    train_dataset = TransformerMovieDataset(
        bundle.train_df["text"], bundle.y_train, tokenizer, config.max_seq_len
    )
    val_dataset = TransformerMovieDataset(bundle.val_df["text"], bundle.y_val, tokenizer, config.max_seq_len)
    test_dataset = TransformerMovieDataset(
        bundle.test_df["text"], bundle.y_test, tokenizer, config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_TRANSFORMER_LEARNING_RATE,
        weight_decay=DEFAULT_TRANSFORMER_WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_saved = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}", leave=False):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)

        val_scores = _collect_probabilities(model, val_loader, device)
        thresholds = tune_thresholds(bundle.y_val, val_scores, raw_scores=False)
        val_pred = apply_thresholds(val_scores, thresholds["per_label_thresholds"])
        macro_f1 = float(f1_score(bundle.y_val, val_pred, average="macro", zero_division=0))
        mean_loss = total_loss / max(1, len(train_dataset))
        print(f"{model_name} epoch {epoch}: loss={mean_loss:.4f}, val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            best_saved = True
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"{model_name}: early stopping after epoch {epoch}")
                break

    if not best_saved:
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    val_scores = _collect_probabilities(model, val_loader, device)
    test_scores = _collect_probabilities(model, test_loader, device)

    save_json(
        {
            "checkpoint": checkpoint,
            "labels": bundle.label_names,
            "best_val_macro_f1": best_macro_f1,
        },
        model_dir / "metadata.json",
    )
    return ModelResult(
        model_name=model_name,
        val_scores=val_scores,
        test_scores=test_scores,
        raw_scores=False,
        metadata={"checkpoint": checkpoint, "best_val_macro_f1": best_macro_f1},
    )


def train_distilbert_experiment(bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    return train_transformer_experiment("distilbert", bundle, config)
