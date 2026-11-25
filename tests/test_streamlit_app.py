from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_synthetic_demo_runs_without_tensorflow():
    app_path = Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    app.checkbox[0].set_value(True).run(timeout=30)

    assert not app.exception
    assert any(element.value == "Route metrics" for element in app.subheader)
