from typing import Dict

import numpy as np
from sklearn.metrics import f1_score


def probability_threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 0.951, 0.05), 2)


def threshold_grid_for_scores(scores: np.ndarray, raw_scores: bool = False) -> np.ndarray:
    base = probability_threshold_grid()
    if not raw_scores:
        return base

    finite = np.asarray(scores)[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0])

    quantiles = np.quantile(finite, np.linspace(0.05, 0.95, 19))
    return np.unique(np.round(np.concatenate([base, quantiles, np.array([0.0])]), 6))


def apply_thresholds(scores: np.ndarray, thresholds) -> np.ndarray:
    scores = np.asarray(scores)
    thresholds = np.asarray(thresholds)
    return (scores >= thresholds).astype(int)


def tune_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    raw_scores: bool = False,
    average: str = "macro",
) -> Dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    global_grid = threshold_grid_for_scores(scores, raw_scores=raw_scores)
    best_global_threshold = float(global_grid[0])
    best_global_f1 = -1.0
    for threshold in global_grid:
        y_pred = apply_thresholds(scores, threshold)
        value = f1_score(y_true, y_pred, average=average, zero_division=0)
        if value > best_global_f1:
            best_global_f1 = float(value)
            best_global_threshold = float(threshold)

    per_label_thresholds = []
    per_label_f1 = []
    for label_idx in range(y_true.shape[1]):
        label_scores = scores[:, label_idx]
        label_true = y_true[:, label_idx]
        label_grid = threshold_grid_for_scores(label_scores, raw_scores=raw_scores)
        if label_true.sum() == 0:
            threshold = float(label_grid.max())
            per_label_thresholds.append(threshold)
            per_label_f1.append(0.0)
            continue

        best_threshold = float(label_grid[0])
        best_f1 = -1.0
        for threshold in label_grid:
            label_pred = (label_scores >= threshold).astype(int)
            value = f1_score(label_true, label_pred, zero_division=0)
            if value > best_f1:
                best_f1 = float(value)
                best_threshold = float(threshold)
        per_label_thresholds.append(best_threshold)
        per_label_f1.append(best_f1)

    return {
        "score_type": "raw_decision" if raw_scores else "probability",
        "global_threshold": best_global_threshold,
        "global_f1": best_global_f1,
        "per_label_thresholds": per_label_thresholds,
        "per_label_f1": per_label_f1,
        "probability_grid": probability_threshold_grid().tolist(),
        "raw_score_grid_note": (
            "Raw decision scores also use validation-score quantiles."
            if raw_scores
            else None
        ),
    }

