"""Command-line prediction using the saved model pipeline.

Because encoding now lives inside the saved sklearn Pipeline, this script
never re-implements label mappings by hand -- it just builds a one-row
dataframe with the raw feature values and calls ``pipeline.predict``. This
is what fixes the original script's swapped H/M/L decode bug: there is no
manual decode table left to get wrong.

Examples
--------
Single prediction via flags:

    python -m student_performance.predict \\
        --gender M --nationality KW --stage-id lowerlevel --grade-id G-04 \\
        --section-id A --topic IT --semester F --relation Father \\
        --raised-hands 15 --visited-resources 16 --announcements-view 2 \\
        --discussion 20 --parent-answering-survey Yes \\
        --parent-school-satisfaction Good --absence-days Under-7

Interactive prompts:

    python -m student_performance.predict --interactive

Batch predictions from a JSON file (list of records with the same keys):

    python -m student_performance.predict --json-file students.json
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd

from . import config

# Maps CLI flag names to the raw dataframe column names the pipeline expects.
FIELD_TO_COLUMN = {
    "gender": "gender",
    "nationality": "NationalITy",
    "stage_id": "StageID",
    "grade_id": "GradeID",
    "section_id": "SectionID",
    "topic": "Topic",
    "semester": "Semester",
    "relation": "Relation",
    "raised_hands": "raisedhands",
    "visited_resources": "VisITedResources",
    "announcements_view": "AnnouncementsView",
    "discussion": "Discussion",
    "parent_answering_survey": "ParentAnsweringSurvey",
    "parent_school_satisfaction": "ParentschoolSatisfaction",
    "absence_days": "StudentAbsenceDays",
}

NUMERIC_FIELDS = {"raised_hands", "visited_resources", "announcements_view", "discussion"}


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m student_performance.train` first."
        )
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["model_name"], bundle["feature_columns"]


def _record_to_frame(record: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {}
    for field, column in FIELD_TO_COLUMN.items():
        raw = record[field]
        if column == "GradeID":
            row[column] = config.GRADE_ID_MAP[raw]
        elif field in NUMERIC_FIELDS:
            row[column] = int(raw)
        else:
            row[column] = raw
    return pd.DataFrame([row], columns=feature_columns)


def predict_one(record: dict) -> str:
    pipeline, _, feature_columns = load_model()
    frame = _record_to_frame(record, feature_columns)
    return pipeline.predict(frame)[0]


def _prompt_record() -> dict:
    prompts = {
        "gender": "Gender (M/F): ",
        "nationality": "Nationality (e.g. KW, USA, Egypt): ",
        "stage_id": "Stage (lowerlevel/MiddleSchool/HighSchool): ",
        "grade_id": "Grade ID (e.g. G-04): ",
        "section_id": "Section (A/B/C): ",
        "topic": "Topic (e.g. IT, Math, Science): ",
        "semester": "Semester (F/S): ",
        "relation": "Relation (Father/Mum): ",
        "raised_hands": "Raised hands (count): ",
        "visited_resources": "Visited resources (count): ",
        "announcements_view": "Announcements viewed (count): ",
        "discussion": "Discussion contributions (count): ",
        "parent_answering_survey": "Parent answered survey (Yes/No): ",
        "parent_school_satisfaction": "Parent school satisfaction (Good/Bad): ",
        "absence_days": "Absence days (Under-7/Above-7): ",
    }
    return {field: input(text).strip() for field, text in prompts.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict a student's performance class (L/M/H).")
    parser.add_argument("--interactive", action="store_true", help="Prompt for each field.")
    parser.add_argument("--json-file", type=str, help="Path to a JSON file with one record or a list of records.")
    for field in FIELD_TO_COLUMN:
        parser.add_argument(f"--{field.replace('_', '-')}", dest=field, type=str)

    args = parser.parse_args(argv)

    if args.interactive:
        record = _prompt_record()
        print(f"\nPredicted class: {predict_one(record)}")
        return

    if args.json_file:
        with open(args.json_file) as f:
            payload = json.load(f)
        records = payload if isinstance(payload, list) else [payload]
        for i, record in enumerate(records):
            print(f"Record {i}: predicted class = {predict_one(record)}")
        return

    record = {field: getattr(args, field) for field in FIELD_TO_COLUMN}
    missing = [f for f, v in record.items() if v is None]
    if missing:
        parser.error(
            "Missing required fields: "
            + ", ".join(missing)
            + "\n(or pass --interactive / --json-file instead)"
        )
    print(f"Predicted class: {predict_one(record)}")


if __name__ == "__main__":
    sys.exit(main())
