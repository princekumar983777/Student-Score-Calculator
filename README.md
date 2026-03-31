# 🎓 Student Exam Score Prediction (Production ML System)

This repository contains a **production-grade ML system** that predicts `exam_score` from tabular student data, trains **multiple models**, and **automatically selects the best** based on validation \(highest R²\).

It also includes a legacy Flask demo app (`app.py`) that can continue to be used independently.

## 📦 Project structure

```
.
├── artifacts/
│   ├── logs/
│   ├── metrics.json
│   ├── models/
│   │   ├── best_model.pkl
│   │   └── best_model.pt
│   └── scaler/
│       └── preprocessor.pkl
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── students.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
├── tests/
├── dvc.yaml
├── params.yaml
├── main.py
└── requirements.txt
```

## ✅ Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run (CLI)

- **Ingest raw data** (copies from `research/` if needed):

```bash
python main.py ingest
```

- **Preprocess** (split train/val/test, fit & save preprocessor):

```bash
python main.py preprocess
```

- **Train + auto-select best model** (logs runs to MLflow, saves best model locally):

```bash
python main.py train
```

- **Evaluate** (prints `artifacts/metrics.json`):

```bash
python main.py evaluate
```

- **Predict** from a CSV:

```bash
python main.py predict --input data/raw/students.csv --output artifacts/predictions.csv
```

## 🧪 Tests

```bash
pytest -q
```

## 🔬 MLflow UI

Runs are logged under the default local `./mlruns` unless `mlflow.tracking_uri` is set in `configs/config.yaml`.

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## ♻️ DVC pipeline (reproducible)

```bash
dvc init
dvc repro
```

To track the raw dataset:

```bash
dvc add data/raw/students.csv
git add data/raw/students.csv.dvc .gitignore
```

## 🧾 Model registry

If enabled in `configs/config.yaml`, the best model is registered to MLflow Model Registry and optionally promoted to `Staging` or `Production`.

## 🗂️ Dataset

Based on the Kaggle dataset “Student Habits vs Academic Performance”.
