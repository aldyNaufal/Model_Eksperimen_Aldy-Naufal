"""
modelling.py
============

Basic modelling:
- Load dataset preprocessed (videos_preprocessed.csv)
- Train TF-IDF + LinearSVC (tanpa tuning)
- Tracking ke MLflow (DagsHub) dengan autolog
- Save model lokal (models/tfidf_svc_genre_game_basic.pkl)
"""

import os
from pathlib import Path

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
# 1. Path & data loader
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PREP_DATA_PATH = PROJECT_ROOT / "data_preprocessing" / "videos_with_genre.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_preprocessed_data() -> tuple[pd.Series, pd.Series]:
    """
    Load dataset bersih hasil preprocessing.
    Diasumsikan sudah punya kolom:
      - 'text'           : gabungan title+description+tags
      - 'primary_genre'  : label genre
    Kalau belum ada kolom 'text', akan dibuat ulang.
    """
    print(f"[INFO] Load data dari: {PREP_DATA_PATH}")
    df = pd.read_csv(PREP_DATA_PATH)

    if "primary_genre" not in df.columns:
        raise ValueError("Kolom 'primary_genre' tidak ditemukan di videos_preprocessed.csv")

    # Buang baris tanpa genre
    df = df.dropna(subset=["primary_genre"]).copy()
    print("[INFO] Jumlah data setelah drop NaN genre:", len(df))

    # Pastikan ada kolom 'text'
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


# ============================================================
# 2. Setup MLflow (DagsHub)
# ============================================================

def setup_mlflow(experiment_name: str = "genre_game_basic"):
    """
    Setup MLflow ke DagsHub menggunakan .env:
      - MLFLOW_TRACKING_URI
      - MLFLOW_TRACKING_USERNAME
      - MLFLOW_TRACKING_PASSWORD
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI tidak diset. "
            "Isi di .env atau environment (DagsHub MLflow URL)."
        )

    # kredensial DagsHub (kalau perlu basic auth)
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("[MLFLOW] Tracking URI :", tracking_uri)
    print("[MLFLOW] Experiment   :", experiment_name)


# ============================================================
# 3. Main basic modelling
# ============================================================

def main():
    setup_mlflow(experiment_name="genre_game_basic")

    X, y = load_preprocessed_data()

    # Train-test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("[INFO] Train size:", len(X_train))
    print("[INFO] Test size :", len(X_test))

    # Pipeline TF-IDF + LinearSVC
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )

    # Aktifkan autolog (Basic Dicoding)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="tfidf_linear_svc_basic"):
        print("[INFO] Training model...")
        model.fit(X_train, y_train)

        print("[INFO] Evaluasi test set...")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print(f"[TEST] Accuracy : {acc:.4f}")
        print(f"[TEST] F1-macro : {f1_macro:.4f}")
        print("\n[TEST] Classification report:\n")
        print(classification_report(y_test, y_pred))

        # Manual logging tambahan (opsional, walau Basic tidak wajib)
        mlflow.log_metric("test_accuracy_manual", acc)
        mlflow.log_metric("test_f1_macro_manual", f1_macro)

        # Simpan model ke file lokal
        model_path = MODELS_DIR / "tfidf_svc_genre_game_basic.pkl"
        joblib.dump(model, model_path)
        print("[SAVE] Model basic disimpan ke:", model_path)


if __name__ == "__main__":
    main()
