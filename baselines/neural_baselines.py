from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from baselines.config import BaselineConfig
from baselines.data_utils import DataBundle
from baselines.thresholding import apply_thresholds, tune_thresholds
from baselines.torch_datasets import TextSequenceDataset, build_vocab
from baselines.utils import ModelResult, get_torch_device, save_json


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes=(3, 4, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=size) for size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled.append(torch.max(features, dim=2).values)
        return self.output(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids):
        mask = input_ids.ne(0)
        embedded = self.embedding(input_ids)
        states, _ = self.lstm(embedded)
        attention_scores = self.attention(states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(states * weights, dim=1)
        return self.output(self.dropout(pooled))


def _model_dir(config: BaselineConfig, model_name: str) -> Path:
    path = config.path("models", model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_model(model_name: str, vocab_size: int, num_labels: int, config: BaselineConfig) -> nn.Module:
    if model_name == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            num_labels=num_labels,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )
    if model_name == "bilstm_attention":
        return BiLSTMAttention(
            vocab_size=vocab_size,
            num_labels=num_labels,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown neural baseline: {model_name}")


def _loader(dataset, config: BaselineConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )


def _collect_probabilities(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(scores)


def train_neural_baseline(model_name: str, bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    device = get_torch_device(config.device)
    model_dir = _model_dir(config, model_name)

    vocab = build_vocab(
        bundle.train_df["text"],
        max_size=config.max_vocab_size,
        min_freq=config.min_token_freq,
    )
    train_dataset = TextSequenceDataset(bundle.train_df["text"], bundle.y_train, vocab, config.max_seq_len)
    val_dataset = TextSequenceDataset(bundle.val_df["text"], bundle.y_val, vocab, config.max_seq_len)
    test_dataset = TextSequenceDataset(bundle.test_df["text"], bundle.y_test, vocab, config.max_seq_len)

    train_loader = _loader(train_dataset, config, shuffle=True)
    val_loader = _loader(val_dataset, config, shuffle=False)
    test_loader = _loader(test_dataset, config, shuffle=False)

    model = _build_model(model_name, len(vocab), len(bundle.label_names), config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_macro_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * input_ids.size(0)

        val_scores = _collect_probabilities(model, val_loader, device)
        thresholds = tune_thresholds(bundle.y_val, val_scores, raw_scores=False)
        val_pred = apply_thresholds(val_scores, thresholds["per_label_thresholds"])
        macro_f1 = float(f1_score(bundle.y_val, val_pred, average="macro", zero_division=0))
        mean_loss = total_loss / max(1, len(train_dataset))
        print(f"{model_name} epoch {epoch}: loss={mean_loss:.4f}, val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"{model_name}: early stopping after epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "labels": bundle.label_names,
        "config": {
            "max_seq_len": config.max_seq_len,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
        },
    }
    torch.save(checkpoint, model_dir / "checkpoint.pt")
    save_json({"labels": bundle.label_names, "vocab_size": len(vocab)}, model_dir / "metadata.json")

    val_scores = _collect_probabilities(model, val_loader, device)
    test_scores = _collect_probabilities(model, test_loader, device)
    return ModelResult(
        model_name=model_name,
        val_scores=val_scores,
        test_scores=test_scores,
        raw_scores=False,
        metadata={"best_val_macro_f1": best_macro_f1, "vocab_size": len(vocab)},
    )

