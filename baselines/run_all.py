import argparse
import sys
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from baselines.config import ALL_MODELS, QUICK_MODELS, BaselineConfig
from baselines.data_utils import DataBundle, prepare_data
from baselines.metrics import (
    compute_metrics,
    plot_leaderboards,
    prediction_frame,
    save_leaderboard,
)
from baselines.thresholding import apply_thresholds, tune_thresholds
from baselines.utils import ModelResult, ensure_artifact_dirs, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-label movie genre baseline benchmarks."
    )
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a small debug setup.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-genres", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tfidf-max-features", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    max_samples = args.max_samples
    if args.quick and max_samples is None:
        max_samples = 3_000

    epochs = args.epochs
    if epochs is None:
        epochs = 1 if args.quick else 3

    tfidf_max_features = args.tfidf_max_features
    if args.quick:
        tfidf_max_features = min(tfidf_max_features, 20_000)

    return BaselineConfig(
        data_path=args.data_path,
        seed=args.seed,
        quick=args.quick,
        max_samples=max_samples,
        top_genres=args.top_genres,
        epochs=epochs,
        batch_size=args.batch_size,
        tfidf_max_features=tfidf_max_features,
        device=args.device,
    )


def selected_models(args: argparse.Namespace) -> List[str]:
    if args.models is not None:
        return args.models
    return QUICK_MODELS if args.quick else ALL_MODELS


def run_model(model_name: str, bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    if model_name == "most_frequent":
        from baselines.sklearn_baselines import run_most_frequent

        return run_most_frequent(bundle, config)
    if model_name == "tfidf_logreg":
        from baselines.sklearn_baselines import run_tfidf_logreg

        return run_tfidf_logreg(bundle, config)
    if model_name == "tfidf_linearsvm":
        from baselines.sklearn_baselines import run_tfidf_linearsvm

        return run_tfidf_linearsvm(bundle, config)
    if model_name in {"textcnn", "bilstm_attention"}:
        from baselines.neural_baselines import train_neural_baseline

        return train_neural_baseline(model_name, bundle, config)
    raise ValueError(f"Unknown model: {model_name}")


def finalize_model_result(
    result: ModelResult,
    bundle: DataBundle,
    config: BaselineConfig,
) -> dict:
    thresholds = tune_thresholds(bundle.y_val, result.val_scores, raw_scores=result.raw_scores)
    y_pred = apply_thresholds(result.test_scores, thresholds["per_label_thresholds"])
    metrics = compute_metrics(bundle.y_test, y_pred, result.test_scores)

    threshold_payload = {
        "model": result.model_name,
        "score_type": thresholds["score_type"],
        "global_threshold": thresholds["global_threshold"],
        "global_f1": thresholds["global_f1"],
        "per_label_thresholds": {
            label: float(thresholds["per_label_thresholds"][idx])
            for idx, label in enumerate(bundle.label_names)
        },
        "per_label_f1": {
            label: float(thresholds["per_label_f1"][idx])
            for idx, label in enumerate(bundle.label_names)
        },
        "probability_grid": thresholds["probability_grid"],
        "raw_score_grid_note": thresholds["raw_score_grid_note"],
        "metadata": result.metadata,
    }
    save_json(
        threshold_payload,
        config.path("thresholds", f"{result.model_name}_thresholds.json"),
    )

    metrics_payload = {
        "model": result.model_name,
        "primary_thresholding": "per_label",
        "split_method": bundle.split_method,
        "labels": bundle.label_names,
        "metadata": result.metadata,
        **metrics,
    }
    save_json(metrics_payload, config.path("metrics", f"{result.model_name}_metrics.json"))

    predictions = prediction_frame(
        bundle.test_df,
        bundle.label_names,
        bundle.y_test,
        y_pred,
        result.test_scores,
    )
    predictions.to_csv(
        config.path("predictions", f"{result.model_name}_test_predictions.csv"),
        index=False,
    )

    return {"model": result.model_name, **metrics}


def main() -> pd.DataFrame:
    args = parse_args()
    config = config_from_args(args)
    models = selected_models(args)

    ensure_artifact_dirs(config.artifacts_dir)
    seed_everything(config.seed)

    print(f"Running models: {', '.join(models)}")
    print(
        "Config: "
        f"quick={config.quick}, max_samples={config.max_samples}, "
        f"top_genres={config.top_genres}, epochs={config.epochs}, "
        f"batch_size={config.batch_size}"
    )

    bundle = prepare_data(config)
    rows = []
    for model_name in models:
        print(f"\n=== {model_name} ===")
        result = run_model(model_name, bundle, config)
        row = finalize_model_result(result, bundle, config)
        rows.append(row)
        print(
            f"{model_name}: micro_f1={row['micro_f1']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}"
        )

    leaderboard = save_leaderboard(rows, config.path("metrics"))
    plot_leaderboards(leaderboard, config.path("plots"))
    print(f"\nSaved leaderboard to {config.path('metrics', 'leaderboard.csv')}")
    return leaderboard


if __name__ == "__main__":
    main()
