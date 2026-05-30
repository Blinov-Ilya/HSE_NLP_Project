from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from baselines.config import BaselineConfig
from baselines.data_utils import DataBundle
from baselines.utils import ModelResult, save_json


def _model_dir(config: BaselineConfig, model_name: str) -> Path:
    path = config.path("models", model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_most_frequent(bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    model_name = "most_frequent"
    label_counts = bundle.y_train.sum(axis=0)
    avg_labels = int(round(float(bundle.y_train.sum(axis=1).mean())))
    k = max(1, min(len(bundle.label_names), avg_labels))
    top_indices = np.argsort(-label_counts)[:k]

    score_template = np.zeros(len(bundle.label_names), dtype=float)
    score_template[top_indices] = 1.0
    val_scores = np.tile(score_template, (len(bundle.val_df), 1))
    test_scores = np.tile(score_template, (len(bundle.test_df), 1))

    metadata = {
        "k": k,
        "top_labels": [bundle.label_names[idx] for idx in top_indices],
        "train_label_counts": {
            bundle.label_names[idx]: int(count) for idx, count in enumerate(label_counts)
        },
    }
    save_json(metadata, _model_dir(config, model_name) / "metadata.json")
    return ModelResult(model_name, val_scores, test_scores, raw_scores=False, metadata=metadata)


def _fit_tfidf(config: BaselineConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )


def run_tfidf_logreg(bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    model_name = "tfidf_logreg"
    model_dir = _model_dir(config, model_name)

    vectorizer = _fit_tfidf(config)
    x_train = vectorizer.fit_transform(bundle.train_df["text"])
    x_val = vectorizer.transform(bundle.val_df["text"])
    x_test = vectorizer.transform(bundle.test_df["text"])

    base_model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=config.sklearn_max_iter,
        random_state=config.seed,
    )
    classifier = OneVsRestClassifier(base_model, n_jobs=-1)
    classifier.fit(x_train, bundle.y_train)

    val_scores = classifier.predict_proba(x_val)
    test_scores = classifier.predict_proba(x_test)

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(classifier, model_dir / "model.joblib")
    save_json({"labels": bundle.label_names}, model_dir / "metadata.json")
    return ModelResult(model_name, val_scores, test_scores, raw_scores=False)


def run_tfidf_linearsvm(bundle: DataBundle, config: BaselineConfig) -> ModelResult:
    model_name = "tfidf_linearsvm"
    model_dir = _model_dir(config, model_name)

    vectorizer = _fit_tfidf(config)
    x_train = vectorizer.fit_transform(bundle.train_df["text"])
    x_val = vectorizer.transform(bundle.val_df["text"])
    x_test = vectorizer.transform(bundle.test_df["text"])

    base_model = LinearSVC(
        class_weight="balanced",
        max_iter=config.sklearn_max_iter,
        random_state=config.seed,
    )
    classifier = OneVsRestClassifier(base_model, n_jobs=-1)
    classifier.fit(x_train, bundle.y_train)

    val_scores = classifier.decision_function(x_val)
    test_scores = classifier.decision_function(x_test)

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(classifier, model_dir / "model.joblib")
    save_json({"labels": bundle.label_names, "score_type": "raw_decision"}, model_dir / "metadata.json")
    return ModelResult(model_name, val_scores, test_scores, raw_scores=True)

