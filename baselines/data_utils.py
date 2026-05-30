import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baselines.config import BaselineConfig
from baselines.utils import ensure_artifact_dirs, labels_to_pipe, save_json


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_names: List[str]
    stats: Dict
    split_method: str
    source_path: Path


def _candidate_data_files(data_dir: Path) -> List[Path]:
    files = []
    for pattern in ("*.parquet", "*.csv", "*.json", "*.jsonl"):
        files.extend(
            path for path in data_dir.rglob(pattern) if "metadata" not in path.name.lower()
        )
    return sorted(files, key=lambda path: path.stat().st_size, reverse=True)


def find_data_file(config: BaselineConfig) -> Path:
    if config.data_path is not None:
        path = Path(config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured data path does not exist: {path}")
        return path

    candidates = _candidate_data_files(config.data_dir)
    if not candidates:
        raise FileNotFoundError(f"No CSV, Parquet, JSON, or JSONL files found in {config.data_dir}")

    if len(candidates) > 1:
        print("Found candidate tabular files:")
        for path in candidates:
            print(f"  {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"Choosing largest file by default: {candidates[0]}")
    return candidates[0]


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def _find_column(columns, candidates: List[str], required: bool = True) -> Optional[str]:
    exact = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    for candidate in candidates:
        needle = candidate.lower()
        for column in columns:
            if needle in column.lower():
                return column
    if required:
        raise ValueError(
            f"Could not find a required column. Tried: {', '.join(candidates)}. "
            f"Available columns: {', '.join(map(str, columns))}"
        )
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "title": _find_column(df.columns, ["title", "movie_title", "name"], required=False),
        "overview": _find_column(
            df.columns, ["overview", "description", "plot", "summary", "synopsis"], required=True
        ),
        "genres": _find_column(df.columns, ["genres", "genre", "labels"], required=True),
        "release_date": _find_column(
            df.columns, ["release_date", "released", "date", "year"], required=False
        ),
        "id": _find_column(df.columns, ["tmdb_id", "id", "movie_id", "imdb_id"], required=False),
    }


def _clean_label(label: str) -> str:
    cleaned = " ".join(str(label).strip().strip("'\"").split())
    return cleaned


def parse_genres(value) -> List[str]:
    labels: List[str] = []

    def add(item) -> None:
        if item is None:
            return
        if isinstance(item, float) and np.isnan(item):
            return
        if isinstance(item, dict):
            for key in ("name", "genre", "label", "title"):
                if key in item:
                    add(item[key])
                    return
            for nested in item.values():
                add(nested)
            return
        if isinstance(item, (list, tuple, set, np.ndarray)):
            for nested in item:
                add(nested)
            return

        text = str(item).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return

        if text[0] in "[{":
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(text)
                    if parsed != text:
                        add(parsed)
                        return
                except Exception:
                    pass

        separator = "|" if "|" in text else "," if "," in text else None
        if separator is not None:
            for part in text.split(separator):
                add(part)
            return

        cleaned = _clean_label(text)
        if cleaned:
            labels.append(cleaned)

    add(value)
    deduped = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def _make_text(title, overview) -> str:
    overview_text = "" if pd.isna(overview) else str(overview).strip()
    title_text = "" if pd.isna(title) else str(title).strip()
    if not title_text:
        return overview_text
    return f"{title_text} [SEP] {overview_text}"


def _extract_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9 and numeric.dropna().between(1800, 2200).mean() > 0.9:
        return numeric.astype("Int64")
    return pd.to_datetime(series, errors="coerce").dt.year.astype("Int64")


def _temporal_split(df: pd.DataFrame) -> Optional[Tuple[Dict[str, pd.DataFrame], str]]:
    if "year" not in df.columns or df["year"].isna().all():
        return None
    train = df[df["year"] <= 2023].copy()
    val = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()
    if min(len(train), len(val), len(test)) == 0:
        return None
    return {"train": train, "val": val, "test": test}, "temporal_train_le_2023_val_2024_test_2025"


def _random_split(df: pd.DataFrame, seed: int) -> Tuple[Dict[str, pd.DataFrame], str]:
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, shuffle=True)
    return (
        {
            "train": df.iloc[train_idx].copy(),
            "val": df.iloc[val_idx].copy(),
            "test": df.iloc[test_idx].copy(),
        },
        "random_70_15_15",
    )


def _make_multihot(label_lists: pd.Series, label_names: List[str]) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(label_names)}
    matrix = np.zeros((len(label_lists), len(label_names)), dtype=np.int64)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in index:
                matrix[row_idx, index[label]] = 1
    return matrix


def _filter_to_top_labels(
    splits: Dict[str, pd.DataFrame], top_genres: int
) -> Optional[Tuple[Dict[str, pd.DataFrame], List[str]]]:
    train_counter = Counter(label for labels in splits["train"]["genres_list"] for label in labels)
    label_names = [label for label, _ in train_counter.most_common(top_genres)]
    if not label_names:
        return None

    label_set = set(label_names)
    filtered = {}
    for split_name, split_df in splits.items():
        current = split_df.copy()
        current["labels_list"] = current["genres_list"].apply(
            lambda labels: [label for label in labels if label in label_set]
        )
        current = current[current["labels_list"].map(len) > 0].copy()
        if current.empty:
            return None
        filtered[split_name] = current.reset_index(drop=True)
    return filtered, label_names


def _save_processed_split(df: pd.DataFrame, path: Path) -> None:
    output = df.copy()
    output["labels"] = output["labels_list"].apply(labels_to_pipe)
    output["all_genres"] = output["genres_list"].apply(labels_to_pipe)
    columns = ["sample_id", "text", "labels", "all_genres"]
    for optional in ("release_date", "year"):
        if optional in output.columns:
            columns.append(optional)
    output[columns].to_csv(path, index=False)


def _plot_genre_distribution(per_label_frequency: Dict[str, int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(per_label_frequency.keys())
    values = list(per_label_frequency.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#8a5a44")
    ax.set_ylabel("Samples")
    ax.set_title("Top genre distribution")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_data(config: BaselineConfig) -> DataBundle:
    ensure_artifact_dirs(config.artifacts_dir)
    source_path = find_data_file(config)
    df = load_table(source_path)
    original_rows = len(df)
    columns = detect_columns(df)

    overview = df[columns["overview"]]
    overview_mask = overview.fillna("").astype(str).str.strip().str.len() >= config.min_overview_chars
    work = df.loc[overview_mask].copy()
    after_overview_rows = len(work)

    title_col = columns["title"]
    id_col = columns["id"]
    release_col = columns["release_date"]

    if title_col is None:
        work["text"] = work[columns["overview"]].apply(lambda value: _make_text("", value))
    else:
        work["text"] = [
            _make_text(title, overview)
            for title, overview in zip(work[title_col], work[columns["overview"]])
        ]

    work["genres_list"] = work[columns["genres"]].apply(parse_genres)
    work = work[work["genres_list"].map(len) > 0].copy()
    after_parse_rows = len(work)

    if id_col is not None:
        work["sample_id"] = work[id_col].astype(str)
    else:
        work["sample_id"] = work.index.astype(str)

    if release_col is not None:
        work["release_date"] = work[release_col].astype(str)
        work["year"] = _extract_year(work[release_col])

    keep_columns = ["sample_id", "text", "genres_list"]
    for optional in ("release_date", "year"):
        if optional in work.columns:
            keep_columns.append(optional)
    work = work[keep_columns].reset_index(drop=True)

    if config.max_samples is not None and len(work) > config.max_samples:
        work = work.sample(n=config.max_samples, random_state=config.seed).reset_index(drop=True)

    split_attempts: List[Tuple[Dict[str, pd.DataFrame], str]] = []
    temporal = _temporal_split(work)
    if temporal is not None:
        split_attempts.append(temporal)
    split_attempts.append(_random_split(work, config.seed))

    filtered_splits = None
    label_names = None
    split_method = None
    for splits, method in split_attempts:
        result = _filter_to_top_labels(splits, config.top_genres)
        if result is not None:
            filtered_splits, label_names = result
            split_method = method
            break

    if filtered_splits is None or label_names is None or split_method is None:
        raise ValueError(
            "Could not create non-empty train/validation/test splits after top-genre filtering. "
            "Try increasing --max-samples or --top-genres."
        )

    y_train = _make_multihot(filtered_splits["train"]["labels_list"], label_names)
    y_val = _make_multihot(filtered_splits["val"]["labels_list"], label_names)
    y_test = _make_multihot(filtered_splits["test"]["labels_list"], label_names)

    all_y = np.vstack([y_train, y_val, y_test])
    per_label_frequency = {
        label: int(all_y[:, idx].sum()) for idx, label in enumerate(label_names)
    }
    train_label_frequency = {
        label: int(y_train[:, idx].sum()) for idx, label in enumerate(label_names)
    }
    stats = {
        "source_path": str(source_path),
        "detected_columns": columns,
        "rows_before_filtering": int(original_rows),
        "rows_after_overview_filter": int(after_overview_rows),
        "rows_after_genre_parse": int(after_parse_rows),
        "rows_after_top_genre_filter": int(all_y.shape[0]),
        "selected_genre_labels": label_names,
        "per_label_frequency": per_label_frequency,
        "train_per_label_frequency": train_label_frequency,
        "average_labels_per_sample": float(all_y.sum(axis=1).mean()),
        "train_size": int(len(filtered_splits["train"])),
        "val_size": int(len(filtered_splits["val"])),
        "test_size": int(len(filtered_splits["test"])),
        "split_method": split_method,
        "top_genres": int(config.top_genres),
        "max_samples": config.max_samples,
        "seed": int(config.seed),
    }

    processed_dir = config.path("processed")
    _save_processed_split(filtered_splits["train"], processed_dir / "train.csv")
    _save_processed_split(filtered_splits["val"], processed_dir / "val.csv")
    _save_processed_split(filtered_splits["test"], processed_dir / "test.csv")
    save_json(stats, processed_dir / "dataset_stats.json")
    save_json({"label_names": label_names}, processed_dir / "labels.json")
    _plot_genre_distribution(per_label_frequency, config.path("plots") / "genre_distribution.png")

    print(
        "Prepared data: "
        f"train={len(filtered_splits['train'])}, "
        f"val={len(filtered_splits['val'])}, "
        f"test={len(filtered_splits['test'])}, "
        f"labels={len(label_names)}, split={split_method}"
    )

    return DataBundle(
        train_df=filtered_splits["train"],
        val_df=filtered_splits["val"],
        test_df=filtered_splits["test"],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_names=label_names,
        stats=stats,
        split_method=split_method,
        source_path=source_path,
    )
