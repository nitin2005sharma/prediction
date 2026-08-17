"""Data loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    config.NUMERIC_COLUMNS
    + config.ORDINAL_COLUMNS
    + config.CATEGORICAL_COLUMNS
    + config.DROPPED_COLUMNS
    + [config.TARGET_COLUMN]
)


def load_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Load and lightly validate the raw student records CSV.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If an expected column is missing or the target has unexpected
        values, so failures happen loudly at load time instead of
        silently downstream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. Expected the CSV at "
            f"data/AI-Data.csv relative to the project root."
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", *df.shape, path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    unexpected_labels = set(df[config.TARGET_COLUMN].unique()) - set(config.CLASS_LABELS)
    if unexpected_labels:
        raise ValueError(
            f"Target column contains unexpected class labels: {unexpected_labels}"
        )

    n_nulls = int(df.isnull().sum().sum())
    if n_nulls:
        logger.warning("Dataset contains %d null values", n_nulls)

    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into model-ready features (X) and target (y).

    GradeID is converted from "G-04" style strings to integers here, since
    that's a data-cleaning step rather than a model-fitting step -- it has
    to happen before the sklearn pipeline ever sees the column, but it does
    not need to be "learned" from the training data the way encoders do.
    """
    df = df.drop(columns=config.DROPPED_COLUMNS, errors="ignore").copy()

    df["GradeID"] = df["GradeID"].map(config.GRADE_ID_MAP)
    if df["GradeID"].isnull().any():
        bad = df.loc[df["GradeID"].isnull(), "GradeID"].index.tolist()
        raise ValueError(f"Unrecognized GradeID values at rows: {bad}")

    y = df[config.TARGET_COLUMN]
    x = df.drop(columns=[config.TARGET_COLUMN])
    return x, y
