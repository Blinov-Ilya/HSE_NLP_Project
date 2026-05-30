# Movie Genre Classification

Multi-label movie genre classification from `title + overview`.

## Dataset

Dataset source:
Kaggle `mechatronixs/tmdb-movies-dataset-20212025`, stored locally in
`data/tmdb_movies_2021_2025.csv` and `data/tmdb_movies_2021_2025.parquet`.

Dataset Statistics
Statistic	Value
Original entries	232,586
Final entries after preprocessing	155,031
Number of genres	15
Train samples	87,575
Validation samples	34,470
Test samples	32,986

## Repository
Notebooks with baselines are in baselines directory.

Main experiment is DistilBERT in `experiments/01_distilbert.ipynb`.

## Results
Metrics and plots are summarized in
`artefacts/analysis/metrics_analysis.ipynb`.

| Model                        |  Macro-F1 |  Micro-F1 | Weighted-F1 | Precision (Micro) | Recall (Micro) | Hamming Loss | Subset Accuracy |
| --- | --- | --- | --- | --- | --- | --- | ---- |
| Most Frequent                |     0.036 |     0.292 |       0.130 |             0.373 |          0.239 |        0.121 |           0.170 |
| TF-IDF + Logistic Regression |     0.471 |     0.577 |       0.580 |             0.521 |          0.648 |        0.099 |           0.279 |
| TF-IDF + Linear SVM          |     0.442 |     0.551 |       0.554 |             0.500 |          0.614 |        0.104 |           0.250 |
| BiLSTM + Attention           |     0.448 |     0.561 |       0.566 |             0.506 |          0.628 |        0.102 |           0.294 |
| TextCNN                      |     0.402 |     0.525 |       0.525 |             0.465 |          0.603 |        0.113 |           0.248 |
| DistilBERT                   | **0.532** | **0.632** |   **0.634** |         **0.591** |      **0.679** |    **0.082** |       **0.369** |
