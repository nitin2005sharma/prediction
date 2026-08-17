"""Preprocessing pipeline and the set of models compared during training.

Wrapping preprocessing and the estimator together in a single sklearn
``Pipeline`` means the OneHotEncoder is fit only on the training fold in
every cross-validation split and on the held-out test set -- it can never
peek at rows it shouldn't, which a manual "encode everything, then split"
approach (the original script's approach) does not guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Numeric columns are scaled, categoricals are one-hot encoded.

    GradeID is passed through untouched: it was already converted to an
    ordinal integer in ``data.prepare_features_and_target``.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_COLUMNS),
            ("ordinal", "passthrough", config.ORDINAL_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_COLUMNS,
            ),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap any sklearn classifier behind the shared preprocessing step."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict


def get_model_specs() -> list[ModelSpec]:
    """The five classifiers from the original script, now with small tuning
    grids instead of default-only hyperparameters."""
    return [
        ModelSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            param_grid={
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 2, 5],
            },
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
            param_grid={
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 8, 12],
            },
        ),
        ModelSpec(
            name="Perceptron",
            estimator=Perceptron(random_state=config.RANDOM_STATE),
            param_grid={
                "model__alpha": [1e-4, 1e-3, 1e-2],
            },
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            param_grid={
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        ModelSpec(
            name="MLP Classifier",
            estimator=MLPClassifier(
                activation="logistic",
                max_iter=3000,
                random_state=config.RANDOM_STATE,
            ),
            param_grid={
                "model__hidden_layer_sizes": [(50,), (100,)],
                "model__alpha": [1e-4, 1e-3],
            },
        ),
    ]
