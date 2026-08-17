from pathlib import Path

from streamlit.testing.v1 import AppTest

from student_performance import config
from student_performance.train import ensure_model_trained

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "student_performance" / "app.py")

# Train once (if needed) before any AppTest run -- training the 5-model
# grid search takes ~40s on a cold cache, which blows past AppTest's
# default script-run timeout. Doing it here, outside the simulated
# script run, keeps each individual test fast and focused on the UI.
ensure_model_trained()


def test_app_loads_without_error():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_app_predicts_on_submit():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    # Push engagement sliders high and submit -- expect a rendered result,
    # not an exception, and a class badge somewhere in the markdown output.
    for slider in at.slider:
        slider.set_value(90)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    rendered = "\n".join(md.value for md in at.markdown)
    assert any(label in rendered for label in config.CLASS_LABELS)
