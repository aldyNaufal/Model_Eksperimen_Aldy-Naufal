"""
modelling_tuning.py
===================

Hyperparameter tuning TF-IDF + LinearSVC dengan MLflow.
- Run 1: tracking lokal (file:./mlruns) -> untuk Basic/Skilled (UI 127.0.0.1)
- Run 2: tracking ke DagsHub (MLFLOW_TRACKING_URI) -> untuk Advanced
- Manual logging + extra metrics (F1 per-genre)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

import mlflow
import mlflow.sklearn

# ============================================================
# 1. Path & loader
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PREP_DATA_PATH = PROJECT_ROOT / "data_preprocessing" / "videos_preprocessed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocessed_data() -> tuple[pd.Series, pd.Series]:
    """
    Load dataset preprocessed.
    Pastikan punya:
      - primary_genre
      - text (kalau belum, akan dibuat dari title+description+tags)
    """
    print(f"[INFO] Load data dari: {PREP_DATA_PATH}")
    df = pd.read_csv(PREP_DATA_PATH)

    if "primary_genre" not in df.columns:
        raise ValueError("Kolom 'primary_genre' tidak ditemukan di videos_preprocessed.csv")

    df = df.dropna(subset=["primary_genre"]).copy()
    print("[INFO] Jumlah data setelah buang NaN genre:", len(df))

    if "text" not in df.columns:
        print("[INFO] Kolom 'text' tidak ada. Membuat dari title+description+tags...")
        def combine_text(row):
            parts = []
            for col in ["title", "description", "tags"]:
                val = row.get(col, "")
                if isinstance(val, str):
                    parts.append(val)
            return " ".join(parts)
        df["text"] = df.apply(combine_text, axis=1)

    X = df["text"]
    y = df["primary_genre"]

    return X, y


def split_train_val_test(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.2,
    val_size_of_temp: float = 0.25,
    random_state: int = 42,
):
    """
    Split data menjadi train / val / test:
    - test_size total default 0.2
    - val_size_of_temp adalah proporsi dari (train+val) untuk val
      (0.25 → train 60%, val 20%, test 20%)
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    print("[INFO] Train size:", len(X_train))
    print("[INFO] Val size  :", len(X_val))
    print("[INFO] Test size :", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 2. Setup MLflow (LOCAL & REMOTE)
# ============================================================

def setup_mlflow_local(experiment_name: str):
    """
    Mode lokal: simpan ke ./mlruns
    Untuk Basic/Skilled, jalankan:
      mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
    """
    tracking_uri = "file:" + str(PROJECT_ROOT / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print("[MLFLOW] MODE LOCAL")
    print("[MLFLOW] Tracking URI :", tracking_uri)
    print("[MLFLOW] Experiment   :", experiment_name)


def setup_mlflow_remote(experiment_name: str):
    """
    Mode remote: simpan ke DagsHub.
    Butuh .env berisi:
      MLFLOW_TRACKING_URI
      MLFLOW_TRACKING_USERNAME
      MLFLOW_TRACKING_PASSWORD
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI tidak diset. "
            "Isi di .env / environment dengan URL MLflow DagsHub."
        )

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("[MLFLOW] MODE REMOTE (DagsHub)")
    print("[MLFLOW] Tracking URI :", tracking_uri)
    print("[MLFLOW] Experiment   :", experiment_name)


# ============================================================
# 3. Model, training, tuning
# ============================================================

def build_pipeline(params: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=params["max_features"],
                    ngram_range=params["ngram_range"],
                    stop_words="english",
                ),
            ),
            ("clf", LinearSVC(C=params["C"])),
        ]
    )


def train_eval_single_config(
    params: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    run_name: str,
) -> tuple[Dict[str, Any], Pipeline, str]:
    """
    Latih & evaluasi satu kombinasi hyperparameter.
    Manual logging ke MLflow + extra metrics (F1 per-genre).
    """
    pipeline = build_pipeline(params)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] run_id: {run_id}")

        # log params
        mlflow.log_param("max_features", params["max_features"])
        mlflow.log_param("ngram_range", str(params["ngram_range"]))
        mlflow.log_param("C", params["C"])

        # training
        print("[INFO] Training model...")
        pipeline.fit(X_train, y_train)

        # validation metrics
        print("[INFO] Evaluasi validation set...")
        y_val_pred = pipeline.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)
        f1_macro_val = f1_score(y_val, y_val_pred, average="macro")
        f1_weighted_val = f1_score(y_val, y_val_pred, average="weighted")

        print(f"[VAL] accuracy   : {acc_val:.4f}")
        print(f"[VAL] f1_macro   : {f1_macro_val:.4f}")
        print(f"[VAL] f1_weighted: {f1_weighted_val:.4f}")

        # log metrics utama
        mlflow.log_metric("val_accuracy", acc_val)
        mlflow.log_metric("val_f1_macro", f1_macro_val)
        mlflow.log_metric("val_f1_weighted", f1_weighted_val)

        # extra metrics (beyond autolog): F1 per-genre (ADVANCED)
        report = classification_report(y_val, y_val_pred, output_dict=True)
        for label, metrics in report.items():
            if label in y_val.unique():
                mlflow.log_metric(f"val_f1_{label}", metrics["f1-score"])

        # (opsional) simpan classification report sebagai artefak tambahan
        report_text = classification_report(y_val, y_val_pred)
        with open("val_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        mlflow.log_artifact("val_classification_report.txt")

        # simpan model sebagai artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    result = {
        "params": params,
        "val_accuracy": acc_val,
        "val_f1_macro": f1_macro_val,
        "val_f1_weighted": f1_weighted_val,
    }
    return result, pipeline, run_id


def tuning_loop(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid: Dict[str, List[Any]],
    run_prefix: str,
):
    """
    Loop manual tuning.
    Return:
      best_result, best_pipeline, best_run_id, all_results
    """
    param_list: List[Dict[str, Any]] = []
    for mf in param_grid["max_features"]:
        for ng in param_grid["ngram_range"]:
            for C in param_grid["C"]:
                param_list.append(
                    {"max_features": mf, "ngram_range": ng, "C": C}
                )

    print(f"[INFO] Total kombinasi hyperparameter: {len(param_list)}")

    best_result = None
    best_pipeline = None
    best_run_id = None
    all_results: List[Dict[str, Any]] = []

    for i, params in enumerate(param_list, start=1):
        print("\n" + "=" * 60)
        print(f"[TUNING] Run {i}/{len(param_list)}")
        print("[TUNING] Params:", params)

        run_name = f"{run_prefix}_tuning_run_{i}"
        result, pipeline, run_id = train_eval_single_config(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name=run_name,
        )

        all_results.append({**result, "run_id": run_id})

        if (best_result is None) or (result["val_f1_macro"] > best_result["val_f1_macro"]):
            best_result = result
            best_pipeline = pipeline
            best_run_id = run_id
            print(f"[BEST UPDATE] New best F1-macro: {best_result['val_f1_macro']:.4f}")

    print("\n" + "=" * 60)
    print("[TUNING] Selesai.")
    print("[TUNING] Best params       :", best_result["params"])
    print("[TUNING] Best val_accuracy :", best_result["val_accuracy"])
    print("[TUNING] Best val_f1_macro :", best_result["val_f1_macro"])
    print("[TUNING] Best run_id       :", best_run_id)

    return best_result, best_pipeline, best_run_id, all_results


def run_one_mode(mode: str):
    """
    mode = 'local'  -> ./mlruns (localhost)
    mode = 'remote' -> DagsHub
    """
    print("\n" + "#" * 70)
    print(f"[MODE] Running tuning in mode: {mode}")
    print("#" * 70)

    X, y = load_preprocessed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    param_grid = {
        "max_features": [20000, 30000, 50000],
        "ngram_range": [(1, 1), (1, 2)],
        "C": [0.5, 1.0, 2.0],
    }

    if mode == "local":
        setup_mlflow_local(experiment_name="genre_game_tuning_local")
        run_prefix = "local"
        model_suffix = "local"

    elif mode == "remote":
        setup_mlflow_remote(experiment_name="genre_game_tuning_remote")
        run_prefix = "remote"
        model_suffix = "remote"

    else:
        raise ValueError("mode harus 'local' atau 'remote'")

    best_result, best_pipeline, best_run_id, all_results = tuning_loop(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid=param_grid,
        run_prefix=run_prefix,
    )

    # Evaluasi best model di test set dalam run terpisah
    print("\n" + "=" * 60)
    print(f"[TEST-{mode}] Evaluasi best model di test set...")

    y_test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    print(f"[TEST-{mode}] accuracy     : {test_acc:.4f}")
    print(f"[TEST-{mode}] f1_macro     : {test_f1_macro:.4f}")
    print(f"[TEST-{mode}] f1_weighted  : {test_f1_weighted:.4f}")
    print("\n[TEST] Classification report:\n")
    print(classification_report(y_test, y_test_pred))

    with mlflow.start_run(run_name=f"{run_prefix}_best_model_test_evaluation") as run:
        mlflow.log_param("source_best_tuning_run_id", best_run_id)
        mlflow.log_param("best_max_features", best_result["params"]["max_features"])
        mlflow.log_param("best_ngram_range", str(best_result["params"]["ngram_range"]))
        mlflow.log_param("best_C", best_result["params"]["C"])

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)

        # log model final sebagai artifact
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
        print("[MLFLOW] Test evaluation run_id:", run.info.run_id)

    # Simpan best model ke file lokal
    model_path = MODELS_DIR / f"tfidf_svc_genre_game_best_tuned_{model_suffix}.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"[SAVE] Best tuned model ({mode}) disimpan ke:", model_path)


# ============================================================
# 4. Main
# ============================================================

def main():
    # 1) Run lokal (mlruns -> 127.0.0.1:5000) untuk Basic/Skilled
    run_one_mode("local")

    # 2) Run remote (DagsHub) untuk Advanced
    run_one_mode("remote")


if __name__ == "__main__":
    main()
