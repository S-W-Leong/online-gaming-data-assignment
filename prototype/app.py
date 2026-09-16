import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype.artifacts import ArtifactLoadError, load_metadata, load_models, validate_artifact_contract
from prototype.contracts import ASSET_MANIFEST, FIGURES_DIR, MODEL_NAMES, PredictionResult
from prototype.insights import InsightsLoadError, decorate_comparison_table, load_comparison_table, recommended_model
from prototype.prediction import predict_player
from prototype.presentation import (
    behavioural_signals_figure,
    engagement_mix_figure,
    install_player_pulse_theme,
    model_comparison_figure,
    probability_figure,
    random_forest_recall_figure,
    render_sidebar,
)


PIXEL_BACKGROUND = PROJECT_ROOT / "prototype" / "assets" / "pixel-world-background.png"

NUMERIC_INPUTS = {
    "Age": ("Age", 15, 49, 32),
    "InGamePurchases": ("In-game purchases", 0, 1, 0),
    "SessionsPerWeek": ("Sessions per week", 0, 19, 9),
    "AvgSessionDurationMinutes": ("Average session duration (minutes)", 10, 179, 95),
    "PlayerLevel": ("Player level", 1, 99, 49),
    "AchievementsUnlocked": ("Achievements unlocked", 0, 49, 25),
}

CATEGORICAL_INPUTS = {
    "Gender": ("Gender", ("Female", "Male")),
    "Location": ("Location", ("Asia", "Europe", "Other", "USA")),
    "GameGenre": ("Game genre", ("Action", "RPG", "Simulation", "Sports", "Strategy")),
    "GameDifficulty": ("Game difficulty", ("Easy", "Medium", "Hard")),
}

PROFILE_GROUPS = (
    ("Player", (("Age", "PlayerLevel"), ("InGamePurchases", "AchievementsUnlocked"))),
    ("Play pattern", (("SessionsPerWeek", "AvgSessionDurationMinutes"),)),
    ("Game context", (("Gender", "Location"), ("GameGenre", "GameDifficulty"))),
)

PROFILE_WIDGET_KEYS = ["profile_model"] + [
    f"profile_{feature}" for feature in (*NUMERIC_INPUTS, *CATEGORICAL_INPUTS)
]


st.set_page_config(
    page_title="Player Pulse",
    page_icon=":material/sports_esports:",
    layout="wide",
    initial_sidebar_state="expanded",
)
install_player_pulse_theme(PIXEL_BACKGROUND)
page = render_sidebar()


def render_visual(caption: str) -> None:
    image_path = FIGURES_DIR / ASSET_MANIFEST[caption]
    if image_path.exists():
        st.image(str(image_path), caption=caption, width="stretch")
    else:
        st.error(f"Missing visual: {image_path.name}. Run online_gaming_analysis.ipynb to regenerate it.")


def render_number_input(feature: str) -> int:
    label, minimum, maximum, default = NUMERIC_INPUTS[feature]
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=1,
        key=f"profile_{feature}",
    )


def render_categorical_input(feature: str) -> str:
    label, options = CATEGORICAL_INPUTS[feature]
    return st.selectbox(label, options, key=f"profile_{feature}")


def render_profile_inputs(metadata: dict[str, object]) -> dict[str, object]:
    supported = set(metadata["numeric_features"]) | set(metadata["categorical_features"])
    expected = set(NUMERIC_INPUTS) | set(CATEGORICAL_INPUTS)
    if supported != expected:
        raise ValueError("Saved feature metadata does not match the supported player profile.")

    values: dict[str, object] = {}
    for group_name, rows in PROFILE_GROUPS:
        st.markdown(f"#### {group_name}")
        for row in rows:
            columns = st.columns(2)
            for column, feature in zip(columns, row, strict=True):
                with column:
                    if feature in NUMERIC_INPUTS:
                        values[feature] = render_number_input(feature)
                    else:
                        values[feature] = render_categorical_input(feature)
    return values


def clear_profile_inputs() -> None:
    for key in PROFILE_WIDGET_KEYS:
        st.session_state.pop(key, None)


def render_prediction_result(result: PredictionResult) -> None:
    st.markdown('<div class="result-label">Prediction result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value">{result.label} engagement</div>', unsafe_allow_html=True)
    st.write(result.guidance)

    if result.probabilities is None:
        st.info("Confidence is unavailable for this model.")
        return

    st.markdown("#### Probability distribution")
    figure = probability_figure(result.probabilities)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.success(f"Recommended next action: {result.guidance}", icon=":material/arrow_forward:")


def render_overview() -> None:
    st.title("Engagement Overview")
    st.caption("Understand the synthetic player dataset and the signals linked with engagement.")

    with st.container(border=True, key="overview_summary"):
        profiles, classes, medium = st.columns(3)
        profiles.metric("Player profiles", "40,034")
        classes.metric("Engagement classes", "3")
        medium.metric("Medium engagement", "48.39%")

    chart_column, signal_column = st.columns([1.42, 1], gap="medium")
    with chart_column:
        with st.container(border=True, key="overview_chart"):
            st.markdown("#### Engagement mix")
            figure = engagement_mix_figure()
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    with signal_column:
        with st.container(border=True, key="overview_signals"):
            st.markdown("#### Strongest behavioural signals")
            figure = behavioural_signals_figure()
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    with st.container(border=True, key="overview_note"):
        message, caveat = st.columns([1.5, 1])
        message.success(
            "Session frequency and duration separate engagement most clearly.",
            icon=":material/star:",
        )
        caveat.markdown(
            '<p class="caveat">Synthetic dataset · association, not causation.</p>',
            unsafe_allow_html=True,
        )

    with st.expander("Explore all EDA figures"):
        captions = (
            "Target distribution",
            "Categorical distributions",
            "Numeric distributions",
            "Numeric correlation heatmap",
            "Numeric features by engagement",
            "Engagement session heatmap",
            "Categorical composition by engagement",
            "Binned engagement rates",
            "Playtime duration consistency",
            "Behavioural segment summary",
        )
        columns = st.columns(2)
        for index, caption in enumerate(captions):
            with columns[index % 2]:
                render_visual(caption)


def render_predict_player() -> None:
    st.title("Predict Player")
    st.caption("Enter a player profile to estimate their engagement level and guide a retention review.")

    try:
        metadata = load_metadata()
        models = load_models()
        validate_artifact_contract(models, metadata)
    except ArtifactLoadError as exc:
        st.error(str(exc))
        return

    form_column, result_column = st.columns([1.1, 1], gap="medium")
    with form_column:
        with st.container(border=True, key="prediction_form_shell"):
            with st.form("prediction_form"):
                selected_model = st.selectbox(
                    "Prediction model",
                    MODEL_NAMES,
                    index=MODEL_NAMES.index("Random Forest"),
                    key="profile_model",
                )
                try:
                    values = render_profile_inputs(metadata)
                except ValueError as exc:
                    st.error(str(exc))
                    values = {}

                primary, secondary = st.columns([1.25, 1])
                submitted = primary.form_submit_button(
                    "Predict engagement",
                    type="primary",
                    icon=":material/bolt:",
                    width="stretch",
                )
                secondary.form_submit_button(
                    "Reset profile",
                    icon=":material/refresh:",
                    width="stretch",
                    on_click=clear_profile_inputs,
                )

    result: PredictionResult | None = None
    if submitted and values:
        try:
            result = predict_player(models[selected_model], values, metadata)
        except ValueError as exc:
            st.error(str(exc))

    with result_column:
        with st.container(border=True, key="prediction_result"):
            if result is None:
                st.markdown('<div class="result-label">Prediction result</div>', unsafe_allow_html=True)
                st.info(
                    "Complete the player profile and run a prediction to see the engagement level, confidence, and suggested next action.",
                    icon=":material/insights:",
                )
                st.markdown("#### What you will get")
                st.write("Low, Medium, or High engagement classification")
                st.write("Class probability distribution when available")
                st.write("A concise, retention-oriented interpretation")
            else:
                render_prediction_result(result)


def render_model_insights() -> None:
    st.title("Model Insights")
    st.caption("Compare six tuned classifiers and understand the deployment recommendation.")

    try:
        table = load_comparison_table()
        winner = recommended_model(table)
    except InsightsLoadError as exc:
        st.error(str(exc))
        return

    winner_row = table.loc[table["model"] == winner].iloc[0]
    with st.container(border=True, key="model_recommendation"):
        model, macro_f1, accuracy = st.columns([1.35, 1, 1])
        model.metric("Recommended model", winner)
        macro_f1.metric("Test macro F1", f"{winner_row['macro_f1']:.3f}")
        accuracy.metric("Test accuracy", f"{winner_row['test_accuracy']:.3f}")
        st.caption("Highest test macro F1; Histogram Gradient Boosting is a close alternative at 0.910.")

    comparison_column, recall_column = st.columns([1.42, 1], gap="medium")
    with comparison_column:
        with st.container(border=True, key="model_chart"):
            st.markdown("#### Test macro F1 by model")
            figure = model_comparison_figure(table)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    with recall_column:
        with st.container(border=True, key="model_recall"):
            st.markdown("#### Random Forest class performance")
            st.caption("Per-class recall on the tuned test set.")
            figure = random_forest_recall_figure()
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            st.markdown("#### Key insight")
            st.write("Medium is easiest to identify; Low and High remain credible but need validation.")

    with st.container(border=True, key="model_note"):
        st.markdown(
            '<p class="caveat">Synthetic dataset · validate on real, time-ordered data before operational use.</p>',
            unsafe_allow_html=True,
        )

    with st.expander("Detailed evaluation artifacts"):
        decorated = decorate_comparison_table(table)
        display_columns = [
            column
            for column in (
                "model",
                "role",
                "macro_f1",
                "test_accuracy",
                "macro_precision",
                "macro_recall",
                "fit_seconds",
            )
            if column in decorated.columns
        ]
        display_table = decorated.loc[:, display_columns].copy()
        numeric_columns = display_table.select_dtypes(include="number").columns
        display_table.loc[:, numeric_columns] = display_table.loc[:, numeric_columns].round(3)
        st.dataframe(display_table, hide_index=True, width="stretch")
        render_visual("Model confusion matrices")
        render_visual("Tuned model confusion matrices")


if page == "Overview":
    render_overview()
elif page == "Predict Player":
    render_predict_player()
else:
    render_model_insights()
