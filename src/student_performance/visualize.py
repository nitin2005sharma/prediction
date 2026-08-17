"""Generate EDA plots and save them to outputs/, instead of the original
interactive input()-driven menu that blocked on plt.show() in a loop.

Run with:  python -m student_performance.visualize
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sb

from . import config, data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLASS_ORDER = config.CLASS_LABELS

# (column, plot title, optional value order)
COUNT_PLOTS = [
    ("Class", "Marks Class Count", CLASS_ORDER),
    ("Semester", "Marks Class -- Semester-wise", None),
    ("gender", "Marks Class -- Gender-wise", ["M", "F"]),
    ("NationalITy", "Marks Class -- Nationality-wise", None),
    ("SectionID", "Marks Class -- Section-wise", None),
    ("Topic", "Marks Class -- Topic-wise", None),
    ("StageID", "Marks Class -- Stage-wise", None),
    ("StudentAbsenceDays", "Marks Class -- Absence-wise", None),
]


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 7))
    sb.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    fig.tight_layout()
    path = config.OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_breakdown(df, column, title, order):
    fig, ax = plt.subplots(figsize=(9, 5))
    if column == "Class":
        sb.countplot(x="Class", data=df, order=order, ax=ax)
    else:
        sb.countplot(x=column, hue="Class", data=df, order=order, hue_order=CLASS_ORDER, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    safe_name = column.lower()
    path = config.OUTPUT_DIR / f"class_by_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate EDA plots for the student dataset.")
    parser.add_argument(
        "--only",
        choices=[c for c, _, _ in COUNT_PLOTS] + ["heatmap"],
        help="Generate only one plot instead of all of them.",
    )
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_data()

    if args.only == "heatmap":
        path = plot_correlation_heatmap(df)
        logger.info("Saved %s", path)
        return

    if args.only:
        column, title, order = next(c for c in COUNT_PLOTS if c[0] == args.only)
        path = plot_class_breakdown(df, column, title, order)
        logger.info("Saved %s", path)
        return

    path = plot_correlation_heatmap(df)
    logger.info("Saved %s", path)
    for column, title, order in COUNT_PLOTS:
        path = plot_class_breakdown(df, column, title, order)
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
