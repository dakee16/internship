from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(__file__))
import config  
import features as F

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_REPO_ROOT, "data", "labeled_columns.json")
MODEL_PATH = os.path.join(_REPO_ROOT, config.MODEL_FILENAME)
FEATURE_CACHE = os.path.join(_REPO_ROOT, "data", "features.npz")

TEST_SIZE: float = 0.2

RF_N_ESTIMATORS: int = 300
RF_MAX_DEPTH: int | None = None

XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 8
XGB_LEARNING_RATE: float = 0.1


def load_dataset(path: str = DATA_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run: python data/generate_labeled_columns.py"
        )
    with open(path) as f:
        return json.load(f)


def build_feature_matrix(
    dataset: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for col in dataset:
        vec = F.extract_feature_vector(col["name"], col["values"])
        rows.append(vec)
        labels.append(col["type"])
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=object)


def train_random_forest(
    x_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(x_train: np.ndarray, y_train: np.ndarray):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        objective="multi:softprob",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    preds = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def main() -> None:
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} columns")

    print("Extracting features...")
    x, y = build_feature_matrix(dataset)
    print(f"  feature matrix: {x.shape}")

    np.savez_compressed(FEATURE_CACHE, x=x, y=y)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_enc,
        test_size=TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_enc,
    )
    print(f"  train={len(x_train)}  test={len(x_test)}\n")

    results: dict[str, dict] = {}

    print("Training RandomForest...")
    rf = train_random_forest(x_train, y_train)
    results["RandomForest"] = {"model": rf, **evaluate(rf, x_test, y_test)}
    print(
        f"  accuracy={results['RandomForest']['accuracy']:.4f}  "
        f"macro_f1={results['RandomForest']['macro_f1']:.4f}"
    )

    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        xgb = train_xgboost(x_train, y_train)
        results["XGBoost"] = {"model": xgb, **evaluate(xgb, x_test, y_test)}
        print(
            f"  accuracy={results['XGBoost']['accuracy']:.4f}  "
            f"macro_f1={results['XGBoost']['macro_f1']:.4f}"
        )
    else:
        print("XGBoost not installed — skipping (RandomForest will be used).")
        print("  Install with: pip install xgboost")

    winner_name = max(results, key=lambda k: results[k]["macro_f1"])
    winner = results[winner_name]
    print(f"\nWinner: {winner_name} (macro_f1={winner['macro_f1']:.4f})")

    artifact = {
        "model": winner["model"],
        "model_type": winner_name,
        "feature_names": list(F.FEATURE_NAMES),
        "label_encoder": label_encoder,
        "test_macro_f1": winner["macro_f1"],
        "test_accuracy": winner["accuracy"],
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()