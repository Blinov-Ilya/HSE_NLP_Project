from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from baselines.utils import labels_to_pipe, sanitize_label


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, y_true.shape[1])
    top_idx = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_idx, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, y_true.shape[1])
    top_idx = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_idx, axis=1).sum(axis=1)
    denom = y_true.sum(axis=1)
    values = np.divide(hits, denom, out=np.zeros_like(hits, dtype=float), where=denom > 0)
    return float(np.mean(values))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_at_1": precision_at_k(y_true, scores, 1),
        "precision_at_3": precision_at_k(y_true, scores, 3),
        "recall_at_3": recall_at_k(y_true, scores, 3),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    try:
        metrics["average_precision_micro"] = float(
            average_precision_score(y_true, scores, average="micro")
        )
    except ValueError:
        metrics["average_precision_micro"] = float("nan")
    return metrics


def prediction_frame(
    df: pd.DataFrame,
    label_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    true_labels = [
        labels_to_pipe([label_names[i] for i, value in enumerate(row) if value])
        for row in y_true
    ]
    pred_labels = [
        labels_to_pipe([label_names[i] for i, value in enumerate(row) if value])
        for row in y_pred
    ]
    output = pd.DataFrame(
        {
            "sample_id": df["sample_id"].to_numpy(),
            "text": df["text"].to_numpy(),
            "true_labels": true_labels,
            "predicted_labels": pred_labels,
        }
    )
    used_columns = set(output.columns)
    for idx, label in enumerate(label_names):
        column = f"score_{sanitize_label(label)}"
        while column in used_columns:
            column = f"{column}_{idx}"
        output[column] = scores[:, idx]
        used_columns.add(column)
    return output


def save_leaderboard(rows: List[Dict], metrics_dir: Path) -> pd.DataFrame:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["macro_f1", "micro_f1"], ascending=False
        ).reset_index(drop=True)
    leaderboard.to_csv(metrics_dir / "leaderboard.csv", index=False)
    leaderboard.to_json(metrics_dir / "leaderboard.json", orient="records", indent=2)
    return leaderboard


def plot_leaderboards(leaderboard: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if leaderboard.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, filename, title in [
        ("micro_f1", "leaderboard_micro_f1.png", "Micro-F1 by model"),
        ("macro_f1", "leaderboard_macro_f1.png", "Macro-F1 by model"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_df = leaderboard.sort_values(metric, ascending=True)
        ax.barh(plot_df["model"], plot_df[metric], color="#2f6f73")
        ax.set_xlabel(metric)
        ax.set_xlim(0, max(1.0, float(plot_df[metric].max()) * 1.1))
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)

