import pandas as pd
import pytest

from student_performance import config, data


def test_load_data_returns_expected_shape():
    df = data.load_data()
    assert len(df) > 0
    assert config.TARGET_COLUMN in df.columns


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_data(tmp_path / "does_not_exist.csv")


def test_load_data_rejects_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    df = data.load_data()
    df.drop(columns=["gender"]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        data.load_data(bad_csv)


def test_load_data_rejects_unexpected_class_label(tmp_path):
    bad_csv = tmp_path / "bad_labels.csv"
    df = data.load_data()
    df = df.copy()
    df.loc[0, config.TARGET_COLUMN] = "X"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="unexpected class labels"):
        data.load_data(bad_csv)


def test_prepare_features_and_target_encodes_grade_and_drops_target():
    df = data.load_data()
    x, y = data.prepare_features_and_target(df)

    assert config.TARGET_COLUMN not in x.columns
    assert "PlaceofBirth" not in x.columns
    assert pd.api.types.is_integer_dtype(x["GradeID"])
    assert set(y.unique()) <= set(config.CLASS_LABELS)
    assert len(x) == len(y) == len(df)


def test_prepare_features_and_target_rejects_unknown_grade():
    df = data.load_data().copy()
    df.loc[0, "GradeID"] = "G-99"
    with pytest.raises(ValueError, match="Unrecognized GradeID"):
        data.prepare_features_and_target(df)
