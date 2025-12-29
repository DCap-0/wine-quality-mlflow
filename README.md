# wine-quality-mlflow

A machine learning project demonstrating **model training, hyperparameter tuning, and experiment tracking using MLflow** on the Wine Classification dataset.

---

## Project Overview
- Dataset: `sklearn.datasets.load_wine`
- Model: Random Forest Classifier
- Hyperparameter tuning: GridSearchCV
- Experiment tracking: MLflow (local & DagsHub)
- Artifacts: confusion matrix, CV results, trained models

---

## Project Structure
  ```
  wine-quality-mlflow
  ├── experiments/ # Training & MLflow scripts
  ├── mlruns/ # MLflow local tracking data
  ├── gridsearch_results.csv # GridSearchCV results
  ├── requirements.txt
  ├── README.md
  └── LICENSE
  ```

---

## Setup Instructions

1. Create virtual environment
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

2.Install dependencies
  ```bash
  pip install -r requirements.txt
  ```


3. Run an experiment
  ```bash
  python3 experiments/2_mlflow_local.py
  ```

4. Launch MLflow UI
  ```bash
  mlflow ui
  ```
Open: [localhost:500](http://localhost:500)

---

## Experiments

| Script | Description |
|------|------------|
| `1_pipeline.py` | Basic training pipeline |
| `2_mlflow_local.py` | Local MLflow tracking |
| `3_mlflow_dagshub.py` | MLflow + DagsHub integration |
| `4_mlflow_autolog_dagshub.py` | MLflow autologging |
| `5_mlflow_hypertune_dagshub.py` | Hyperparameter tuning with GridSearchCV and MLflow |

---

## Key Learnings
- Proper MLflow experiment structuring
- Logging metrics, models, datasets, and artifacts
- Handling sklearn + typing edge cases
- Hyperparameter tuning with reproducibility