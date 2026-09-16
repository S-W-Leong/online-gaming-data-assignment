from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_readme_documents_how_to_run_the_prototype():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "streamlit run prototype/app.py" in text
    assert "online_gaming_analysis.ipynb" in text


def test_overview_surfaces_the_factual_dataset_summary():
    app_path = Path(__file__).parents[1] / "prototype" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not at.exception
    assert at.title[0].value == "Engagement Overview"
    assert [(metric.label, metric.value) for metric in at.metric] == [
        ("Player profiles", "40,034"),
        ("Engagement classes", "3"),
        ("Medium engagement", "48.39%"),
    ]
    assert set(at.sidebar.radio[0].options) == {
        "Overview",
        "Predict Player",
        "Model Insights",
    }


def test_predict_page_lists_all_six_trained_models():
    app_path = Path(__file__).parents[1] / "prototype" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15).run()

    at.sidebar.radio[0].set_value("Predict Player").run()

    assert not at.exception
    assert at.title[0].value == "Predict Player"
    prediction_model = next(selectbox for selectbox in at.selectbox if selectbox.label == "Prediction model")
    assert prediction_model.value == "Random Forest"
    assert tuple(prediction_model.options) == (
        "Logistic Regression (Baseline)",
        "K-Nearest Neighbours",
        "Decision Tree",
        "Random Forest",
        "Histogram Gradient Boosting",
        "RBF Support Vector Machine",
    )


def test_predict_page_uses_the_supported_ten_feature_contract():
    app_path = Path(__file__).parents[1] / "prototype" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15).run()

    at.sidebar.radio[0].set_value("Predict Player").run()

    input_labels = {widget.label for widget in (*at.number_input, *at.selectbox)}
    assert input_labels == {
        "Prediction model",
        "Age",
        "In-game purchases",
        "Sessions per week",
        "Average session duration (minutes)",
        "Player level",
        "Achievements unlocked",
        "Gender",
        "Location",
        "Game genre",
        "Game difficulty",
    }


def test_model_insights_leads_with_the_deployment_recommendation():
    app_path = Path(__file__).parents[1] / "prototype" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15).run()

    at.sidebar.radio[0].set_value("Model Insights").run()

    assert not at.exception
    assert at.title[0].value == "Model Insights"
    assert [(metric.label, metric.value) for metric in at.metric[:3]] == [
        ("Recommended model", "Random Forest"),
        ("Test macro F1", "0.912"),
        ("Test accuracy", "0.916"),
    ]
