"""Train and compare all models, then save the best pipeline to disk.

Run with:  python -m student_performance.train
"""

from __future__ import annotations

import argparse
import logging

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import config, data, evaluate
from .pipeline import build_pipeline, get_model_specs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train_and_compare(random_state: int = config.RANDOM_STATE) -> tuple:
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)

    # Stratified + random_state means this split is both reproducible and
    # representative of the ~44/30/26% Medium/High/Low class balance --
    # the original script's manual slice-after-(non-)shuffle guaranteed
    # neither.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Train rows: %d | Test rows: %d", len(x_train), len(x_test))

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state)

    results = []
    fitted_pipelines = {}
    for spec in get_model_specs():
        logger.info("Tuning %s ...", spec.name)
        pipe = build_pipeline(spec.estimator)
        search = GridSearchCV(
            pipe, spec.param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        search.fit(x_train, y_train)
        best_pipe = search.best_estimator_
        fitted_pipelines[spec.name] = best_pipe

        y_pred = best_pipe.predict(x_test)
        metrics = evaluate.score_predictions(y_test, y_pred)
        metrics.update(
            {
                "model": spec.name,
                "best_params": search.best_params_,
                "cv_f1_macro": search.best_score_,
            }
        )
        results.append(metrics)
        logger.info(
            "%-20s test accuracy=%.3f  f1_macro=%.3f  (cv f1_macro=%.3f)",
            spec.name,
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["cv_f1_macro"],
        )

    leaderboard = evaluate.build_leaderboard(results)
    return leaderboard, fitted_pipelines, (x_test, y_test)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and compare student-performance models.")
    parser.add_argument(
        "--random-state", type=int, default=config.RANDOM_STATE, help="Seed for split + CV."
    )
    args = parser.parse_args(argv)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard, fitted_pipelines, (x_test, y_test) = train_and_compare(args.random_state)

    print("\n=== Model comparison (sorted by test macro-F1) ===")
    print(
        leaderboard[
            ["model", "accuracy", "f1_macro", "precision_macro", "recall_macro", "cv_f1_macro"]
        ].to_string(index=False)
    )

    leaderboard.to_csv(config.LEADERBOARD_PATH, index=False)
    logger.info("Saved leaderboard to %s", config.LEADERBOARD_PATH)

    best_row = leaderboard.iloc[0]
    best_name = best_row["model"]
    best_pipeline = fitted_pipelines[best_name]

    y_pred_best = best_pipeline.predict(x_test)
    cm_path = evaluate.plot_confusion_matrix(y_test, y_pred_best, best_name)
    logger.info("Saved confusion matrix to %s", cm_path)

    importance_df, method = evaluate.compute_feature_importance(best_pipeline, x_test, y_test)
    importance_df.to_csv(config.FEATURE_IMPORTANCE_CSV_PATH, index=False)
    fi_path = evaluate.plot_feature_importance(importance_df, method, best_name)
    logger.info("Saved feature importance (%s) to %s", method, fi_path)
    print(f"\n=== Top features for {best_name} ({method}) ===")
    print(importance_df.to_string(index=False))

    joblib.dump(
        {"pipeline": best_pipeline, "model_name": best_name, "feature_columns": list(x_test.columns)},
        config.MODEL_PATH,
    )
    logger.info("Saved best model (%s) to %s", best_name, config.MODEL_PATH)


def ensure_model_trained() -> None:
    """Train and save a model if one isn't already on disk.

    Used by the Streamlit app so a fresh clone without a committed model
    file (or a wiped `models/` dir) still works -- it just trains once,
    lazily, on first load instead of failing outright.
    """
    if config.MODEL_PATH.exists():
        return
    logger.info("No saved model found at %s -- training one now.", config.MODEL_PATH)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main(argv=[])


if __name__ == "__main__":
    main()
