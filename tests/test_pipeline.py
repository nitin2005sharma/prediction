from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from student_performance import config, data, evaluate
from student_performance.pipeline import build_pipeline, get_model_specs


def test_pipeline_fits_and_predicts_valid_labels():
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )

    pipe = build_pipeline(DecisionTreeClassifier(random_state=0))
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)

    assert len(preds) == len(x_test)
    assert set(preds) <= set(config.CLASS_LABELS)


def test_pipeline_handles_unseen_category_without_error():
    # OneHotEncoder(handle_unknown="ignore") should not blow up on a
    # nationality that wasn't present at fit time.
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)

    pipe = build_pipeline(DecisionTreeClassifier(random_state=0))
    pipe.fit(x, y)

    novel_row = x.iloc[[0]].copy()
    novel_row["NationalITy"] = "Atlantis"
    prediction = pipe.predict(novel_row)
    assert prediction[0] in config.CLASS_LABELS


def test_all_registered_models_build_valid_pipelines():
    specs = get_model_specs()
    assert len(specs) == 5
    for spec in specs:
        pipe = build_pipeline(spec.estimator)
        assert pipe.named_steps["model"] is spec.estimator


def test_feature_importance_tree_model_uses_builtin_importances():
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    pipe = build_pipeline(DecisionTreeClassifier(random_state=0))
    pipe.fit(x_train, y_train)

    importance_df, method = evaluate.compute_feature_importance(pipe, x_test, y_test, top_n=5)

    assert method == "impurity-based (built-in)"
    assert len(importance_df) == 5
    assert not importance_df["feature"].str.contains("__").any()  # prefixes stripped
    assert importance_df["importance"].is_monotonic_decreasing


def test_feature_importance_falls_back_to_permutation_for_linear_model():
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    pipe = build_pipeline(LogisticRegression(max_iter=200, random_state=0))
    pipe.fit(x_train, y_train)

    importance_df, method = evaluate.compute_feature_importance(pipe, x_test, y_test, top_n=5)

    assert "permutation" in method
    assert len(importance_df) == 5
