"""
modelling_tuning.py
===================

Pipeline lengkap:
- Setup MLflow (Local / DagsHub)
- Load & split data
- TF-IDF + LinearSVC hyperparameter tuning
- Logging artefak lengkap (MLmodel + conda.yaml + python_env.yaml,
  + requirements.txt + estimator.html + metric_info.json +
  confusion_matrix.png)
- Simpan final model (.pkl)

Pastikan .env berisi:

MLFLOW_MODE=remote
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<token>
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import json
import shutil
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =====================================================================
# PATH GLOBAL
# =====================================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_preprocessing" / "videos_with_genre.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "Track_Model_Eksperimen_Aldy-Naufal"


# =====================================================================
# MLFLOW SETUP
# =====================================================================

def load_env():
    env_path = ROOT / ".env"
    print(f"[MLFLOW] Loading .env → {env_path}")

    if env_path.exists():
        load_dotenv(env_path, override=True)
        print("[MLFLOW] .env loaded")
    else:
        print("[MLFLOW] WARNING: .env not found!")


def setup_mlflow():
    load_env()

    mode = os.getenv("MLFLOW_MODE", "remote").lower()
    print(f"[MLFLOW] Mode: {mode}")

    if mode == "local":
        tracking_uri = f"file:{ROOT / 'mlruns'}"
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI tidak ditemukan di .env")

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[MLFLOW] Tracking URI:", mlflow.get_tracking_uri())
    print("[MLFLOW] Experiment :", EXPERIMENT_NAME)


# =====================================================================
# DATA LOADING & SPLIT
# =====================================================================

def load_preprocessed_data():
    df = pd.read_csv(DATA_PATH)

    if "primary_genre" not in df:
        raise ValueError("'primary_genre' tidak ada di dataset!")

    df = df.dropna(subset=["primary_genre"]).copy()

    if "text" not in df.columns:
        df["text"] = (
            df["title"].fillna("") + " " +
            df["description"].fillna("") + " " +
            df["tags"].fillna("")
        )

    return df["text"], df["primary_genre"]


def split_data(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# MODEL PIPELINE
# =====================================================================

def build_pipeline(params):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            stop_words="english"
        )),
        ("clf", LinearSVC(C=params["C"]))
    ])


# =====================================================================
# LOGGING ARTEFAK TAMBAHAN
# =====================================================================

def log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")


def log_metric_info(metrics):
    with open("metric_info.json", "w") as f:
        json.dump(metrics, f, indent=4)
    mlflow.log_artifact("metric_info.json")


def log_estimator_description(model):
    html = f"""
    <html>
        <body>
            <h2>Pipeline Description</h2>
            <pre>{model}</pre>
        </body>
    </html>
    """

    with open("estimator.html", "w") as f:
        f.write(html)

    mlflow.log_artifact("estimator.html")


# =====================================================================
# TUNING LOOP
# =====================================================================

def run_tuning(X_train, y_train, X_val, y_val):

    param_grid = [
        {"max_features": 20000, "ngram_range": (1, 1), "C": 0.5},
        {"max_features": 30000, "ngram_range": (1, 2), "C": 1.0},
        {"max_features": 50000, "ngram_range": (1, 2), "C": 2.0},
    ]

    best_model = None
    best_f1 = -1
    best_params = None

    for params in param_grid:
        with mlflow.start_run(run_name="tuning_run"):
            mlflow.log_params(params)

            model = build_pipeline(params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            f1_macro = f1_score(y_val, y_pred, average="macro")

            mlflow.log_metric("val_f1_macro", f1_macro)

            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model = model
                best_params = params

    return best_model, best_params, best_f1


# =====================================================================
# MAIN
# =====================================================================

def main():
    setup_mlflow()

    X, y = load_preprocessed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\n[INFO] Starting tuning...")
    best_model, best_params, best_f1 = run_tuning(X_train, y_train, X_val, y_val)

    print("\n[INFO] Best params:", best_params)
    print("[INFO] Best F1 val:", best_f1)

    print("\n[INFO] Evaluating test set...")
    y_pred_test = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    with mlflow.start_run(run_name="best_model_test"):

        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)

        log_confusion_matrix(y_test, y_pred_test)
        log_metric_info({
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1)
        })
        log_estimator_description(best_model)

        # =====================
        #   MLflow Model Save
        # =====================

        local_model_dir = "model"

        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)

        mlflow.sklearn.save_model(best_model, local_model_dir)
        mlflow.log_artifacts(local_model_dir, artifact_path="model")

    # simpan model lokal
    joblib.dump(best_model, MODEL_DIR / "best_tfidf_svc.pkl")

    print("\n[SAVED] best_tfidf_svc.pkl stored at:", MODEL_DIR)
    print("[DONE] Training complete ✓")


# =====================================================================
if __name__ == "__main__":
    main()
