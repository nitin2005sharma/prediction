"""Central configuration for the student performance pipeline.

Keeping paths, column lists and hyperparameters here (instead of scattered
through scripts) means every module agrees on the same definitions, and a
reviewer can see the modeling decisions in one place.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "AI-Data.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.png"
FEATURE_IMPORTANCE_CSV_PATH = OUTPUT_DIR / "feature_importance.csv"

# --------------------------------------------------------------------------
# Target / feature columns
# --------------------------------------------------------------------------
TARGET_COLUMN = "Class"
CLASS_LABELS = ["L", "M", "H"]  # Low, Medium, High -- used only for display order

# `PlaceofBirth` is dropped by default: it agrees with `NationalITy` for
# ~88% of rows in this dataset, so keeping both mostly adds redundant,
# correlated dimensions without adding predictive signal. Flip this to an
# empty list to keep every original column instead.
DROPPED_COLUMNS = ["PlaceofBirth"]

# GradeID (e.g. "G-07") is genuinely ordinal, so it's mapped to an integer
# rather than one-hot encoded -- this preserves the "grade 7 is between
# grade 6 and grade 8" relationship that one-hot encoding would throw away.
GRADE_ID_MAP = {f"G-{i:02d}": i for i in range(1, 13)}

NUMERIC_COLUMNS = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
]

ORDINAL_COLUMNS = ["GradeID"]

# Every other non-numeric, non-target, non-dropped column is treated as
# nominal categorical and one-hot encoded. Listed explicitly (rather than
# inferred at runtime) so the feature set is easy to audit and diff in code
# review.
CATEGORICAL_COLUMNS = [
    "gender",
    "NationalITy",
    "StageID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]

# --------------------------------------------------------------------------
# Train / eval configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
