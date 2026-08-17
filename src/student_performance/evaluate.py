"""Evaluation helpers: metrics table, confusion matrix, feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never try to open a GUI window when run headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from . import config


def score_predictions(y_true, y_pred) -> dict:
    """Return the headline metrics for a single model's test predictions."""
    report = classification_report(
        y_true, y_pred, labels=config.CLASS_LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "f1_macro": f1_score(y_true, y_pred, labels=config.CLASS_LABELS, average="macro"),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
    }


def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    """Turn a list of per-model result dicts into a sorted comparison table."""
    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df


def plot_confusion_matrix(y_true, y_pred, model_name: str, path: Path = config.CONFUSION_MATRIX_PATH):
    """Save a confusion matrix plot for the best-performing model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=config.CLASS_LABELS, ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix -- {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_feature_importance(pipeline, x_test, y_test, top_n: int = 15) -> tuple[pd.DataFrame, str]:
    """Rank features by importance for the fitted pipeline's model.

    Uses the model's built-in `feature_importances_` when available (tree
    ensembles like Random Forest / Decision Tree). Otherwise falls back to
    permutation importance -- measuring the test-set score drop when a
    feature's values are shuffled -- which works for *any* estimator
    (linear models, MLP, Perceptron) and, unlike coefficients, is on a
    comparable scale across one-hot encoded categorical columns.
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        # Built-in importances are indexed by the *expanded* post-encoding
        # feature names (one entry per one-hot column, etc.).
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = model.feature_importances_
        method = "impurity-based (built-in)"
    else:
        # Permutation importance runs against the whole pipeline (it
        # shuffles a column, re-predicts, and checks the score drop), so
        # it's indexed by the *raw* pre-encoding input columns instead.
        feature_names = np.asarray(x_test.columns)
        result = permutation_importance(
            pipeline,
            x_test,
            y_test,
            n_repeats=10,
            random_state=config.RANDOM_STATE,
            scoring="f1_macro",
        )
        importances = result.importances_mean
        method = "permutation (test-set f1_macro drop)"

    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    # ColumnTransformer prefixes names with the transformer id (e.g.
    # "numeric__raisedhands"); drop that prefix, it's implementation
    # detail rather than something a reader needs.
    df["feature"] = df["feature"].str.replace(r"^(numeric|categorical|ordinal)__", "", regex=True)
    return df.head(top_n), method


def plot_feature_importance(
    importance_df: pd.DataFrame,
    method: str,
    model_name: str,
    path: Path = config.FEATURE_IMPORTANCE_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = importance_df.iloc[::-1]  # largest bar on top
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#3b6fa0")
    ax.set_xlabel(f"Importance ({method})")
    ax.set_title(f"Top Feature Importances -- {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
