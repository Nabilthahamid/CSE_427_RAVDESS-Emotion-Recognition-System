"""Classical ML model training and evaluation for RAVDESS emotion recognition.

This module trains and evaluates five baseline models on handcrafted audio features:
  1. SVM (RBF): Support Vector Machine with radial basis function kernel
  2. MLP: Multi-layer Perceptron neural network
  3. Random Forest: Ensemble of decision trees
  4. XGBoost: Gradient boosting on decision trees
  5. LightGBM: Lightweight gradient boosting machine

All models handle class imbalance via class_weight and are evaluated with:
  - Accuracy: overall correctness
  - Macro F1: unweighted mean of per-class F1 scores (best for imbalanced data)
  - Weighted F1: F1 weighted by class support (reflects dataset distribution)
  - Confusion matrix: per-emotion classification breakdown
  - Classification report: per-class precision, recall, F1
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


FEATURE_PREFIX = "f_"
"""Prefix used to identify feature columns in the feature table."""


def _xy_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from a feature table.
    
    Args:
        frame: DataFrame with columns starting with FEATURE_PREFIX (features) and 'emotion' (label).
    
    Returns:
        Tuple of (X, y) where X is float32 array of features and y is string array of emotion labels.
    """
    feature_cols = [c for c in frame.columns if c.startswith(FEATURE_PREFIX)]
    x = frame[feature_cols].to_numpy(dtype=np.float32)
    y = frame["emotion"].to_numpy()
    return x, y


def _evaluate_and_save(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: list[str],
    output_dir: Path,
) -> dict[str, float]:
    """Evaluate model predictions and save metrics, reports, and confusion matrix.
    
    Args:
        model_name: Name of the model (used in filenames).
        y_true: True encoded labels (integer IDs).
        y_pred: Predicted encoded labels (integer IDs).
        display_labels: Human-readable emotion names for reporting.
        output_dir: Directory to save artifacts.
    
    Returns:
        Dictionary with keys: 'accuracy', 'f1_macro', 'f1_weighted'.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(display_labels))),
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )
    with (output_dir / f"{model_name}_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(display_labels))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=display_labels, yticklabels=display_labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close()

    return metrics


def train_and_evaluate_models(
    split_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train all baseline models and evaluate on test set.
    
    Trains five models on combined train+val data, saves each model and its evaluation artifacts,
    then returns a DataFrame with metrics sorted by macro F1 (best metric for imbalanced data).
    
    Args:
        split_frames: Dict with keys 'train', 'val', 'test' containing DataFrames with features and labels.
        output_dir: Directory to save models, reports, and comparison plots.
        random_state: Random seed for reproducibility.
    
    Returns:
        DataFrame with columns: model, accuracy, f1_macro, f1_weighted (sorted by f1_macro descending).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = split_frames["train"]
    test_df = split_frames["test"]
    val_df = split_frames.get("val")  # Optional validation set
    
    # Extract features and labels from all splits
    x_train, y_train = _xy_from_frame(train_df)
    x_test, y_test = _xy_from_frame(test_df)
    
    # Prepare training data: with/without validation set
    if val_df is None:
        # No validation set: use only train data
        x_train_full = x_train
        y_train_full = y_train
    else:
        # With validation set: combine train+val for final training
        x_val, y_val = _xy_from_frame(val_df)
        x_train_full = np.concatenate([x_train, x_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

    # Encode string labels to integers for model training
    encoder = LabelEncoder()
    all_labels = np.concatenate([y_train_full, y_test], axis=0)
    encoder.fit(all_labels)
    display_labels = encoder.classes_.tolist()
    
    y_train_full_enc = encoder.transform(y_train_full)
    y_test_enc = encoder.transform(y_test)

    models = {
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        gamma="scale",
                        C=10.0,
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=400,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    if HAS_LIGHTGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    rows = []
    for name, model in models.items():
        model.fit(x_train_full, y_train_full_enc)
        preds = model.predict(x_test)

        metrics = _evaluate_and_save(name, y_test_enc, preds, display_labels, out)
        rows.append({"model": name, **metrics})

        joblib.dump(model, out / f"{name}.joblib")

    metrics_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(out / "model_metrics.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=metrics_df, x="model", y="f1_macro", color="#4C72B0")
    plt.title("Model Comparison (Macro F1)")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out / "model_comparison_macro_f1.png", dpi=180)
    plt.close()

    return metrics_df
