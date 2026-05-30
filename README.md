# Movie Genre Classification

Multi-label movie genre classification from `title + overview`.

Data: Kaggle `mechatronixs/tmdb-movies-dataset-20212025`, stored locally in
`data/tmdb_movies_2021_2025.csv` and `data/tmdb_movies_2021_2025.parquet`.

Main experiment entry point:

```bash
python baselines/run_all.py --models most_frequent tfidf_logreg tfidf_linearsvm textcnn bilstm_attention
```

DistilBERT is kept as a separate experiment in `experiments/01_distilbert.ipynb`,
not as a baseline. Metrics and plots are summarized in
`artefacts/analysis/metrics_analysis.ipynb`.
