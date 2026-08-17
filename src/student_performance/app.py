"""Interactive demo: predict a student's performance class in the browser.

Run locally with:  streamlit run src/student_performance/app.py
(or, once installed:  streamlit run app.py  from the project root, see README)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from student_performance import config, data
from student_performance.predict import FIELD_TO_COLUMN, _record_to_frame, load_model
from student_performance.train import ensure_model_trained

st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="centered")

CLASS_INFO = {
    "H": ("High", "#2e7d32"),
    "M": ("Medium", "#f9a825"),
    "L": ("Low", "#c62828"),
}


@st.cache_resource
def get_model():
    ensure_model_trained()
    return load_model()


@st.cache_data
def get_reference_data():
    return data.load_data()


def main():
    st.title("📊 Student Performance Predictor")
    st.write(
        "Predicts whether a student's performance class is **Low**, **Medium**, "
        "or **High** from behavioral engagement data -- not grades. Backed by a "
        "Random Forest classifier, ~78.5% test accuracy, 0.79 macro-F1."
    )

    try:
        pipeline, model_name, feature_columns = get_model()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python -m student_performance.train` "
            "first to generate `models/best_model.joblib`."
        )
        return

    df = get_reference_data()
    nationalities = sorted(df["NationalITy"].unique())
    topics = sorted(df["Topic"].unique())
    sections = sorted(df["SectionID"].unique())

    st.caption(f"Model in use: **{model_name}**")

    with st.form("prediction_form"):
        st.subheader("Engagement")
        c1, c2 = st.columns(2)
        with c1:
            raised_hands = st.slider("Raised hands (count)", 0, 100, 50)
            visited_resources = st.slider("Resources visited (count)", 0, 100, 65)
        with c2:
            announcements_view = st.slider("Announcements viewed (count)", 0, 100, 33)
            discussion = st.slider("Discussion contributions (count)", 0, 100, 39)

        absence_days = st.radio("Absence days", ["Under-7", "Above-7"], horizontal=True)

        st.subheader("Student details")
        c3, c4, c5 = st.columns(3)
        with c3:
            gender = st.selectbox("Gender", ["M", "F"])
            stage_id = st.selectbox("Stage", ["lowerlevel", "MiddleSchool", "HighSchool"])
            section_id = st.selectbox("Section", sections)
        with c4:
            grade_id = st.selectbox("Grade", list(config.GRADE_ID_MAP.keys()), index=3)
            semester = st.selectbox("Semester", ["F", "S"])
            relation = st.selectbox("Parent relation", ["Father", "Mum"])
        with c5:
            nationality = st.selectbox("Nationality", nationalities)
            topic = st.selectbox("Course topic", topics)

        st.subheader("Parent engagement")
        c6, c7 = st.columns(2)
        with c6:
            parent_answering_survey = st.radio("Parent answered survey", ["Yes", "No"], horizontal=True)
        with c7:
            parent_school_satisfaction = st.radio("Parent satisfaction", ["Good", "Bad"], horizontal=True)

        submitted = st.form_submit_button("Predict performance class", use_container_width=True)

    if submitted:
        record = {
            "gender": gender,
            "nationality": nationality,
            "stage_id": stage_id,
            "grade_id": grade_id,
            "section_id": section_id,
            "topic": topic,
            "semester": semester,
            "relation": relation,
            "raised_hands": raised_hands,
            "visited_resources": visited_resources,
            "announcements_view": announcements_view,
            "discussion": discussion,
            "parent_answering_survey": parent_answering_survey,
            "parent_school_satisfaction": parent_school_satisfaction,
            "absence_days": absence_days,
        }
        assert set(record) == set(FIELD_TO_COLUMN)  # keep app + predict.py in sync

        frame = _record_to_frame(record, feature_columns)
        prediction = pipeline.predict(frame)[0]
        label, color = CLASS_INFO[prediction]

        st.markdown(
            f"""
            <div style="padding:1.2rem;border-radius:0.5rem;background-color:{color}22;
                        border:1px solid {color};text-align:center;margin-top:1rem;">
                <span style="font-size:1.1rem;">Predicted performance class</span><br/>
                <span style="font-size:2rem;font-weight:700;color:{color};">{label} ({prediction})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(frame)[0]
            classes = pipeline.named_steps["model"].classes_
            proba_df = pd.DataFrame({"class": classes, "probability": proba}).set_index("class")
            proba_df = proba_df.reindex(config.CLASS_LABELS)
            st.bar_chart(proba_df)

    st.divider()
    st.caption(
        "Dataset: [xAPI-Edu-Data](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data), "
        "480 students. See the project README for methodology and full results."
    )


if __name__ == "__main__":
    main()
