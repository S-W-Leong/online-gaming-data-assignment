import base64
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


INK = "#F7F4FF"
MUTED = "#B8B4CA"
PANEL = "#15172B"
GRID = "#37394F"
BLUE = "#668FBE"
LIME = "#B8F23C"
CORAL = "#FF696D"
PURPLE = "#C386FF"

ENGAGEMENT_COLORS = {"Low": BLUE, "Medium": LIME, "High": CORAL}


def install_player_pulse_theme(pixel_background: Path) -> None:
    background_data = base64.b64encode(pixel_background.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        :root {{
            --pp-bg: #0B0C18;
            --pp-sidebar: #111225;
            --pp-panel: #15172B;
            --pp-panel-soft: #1B1C33;
            --pp-border: #3D3F59;
            --pp-ink: #F7F4FF;
            --pp-muted: #B8B4CA;
            --pp-purple: #C386FF;
            --pp-coral: #FF696D;
            --pp-lime: #B8F23C;
            --pp-blue: #668FBE;
        }}

        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background: var(--pp-bg);
            color: var(--pp-ink);
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
            height: 0;
        }}

        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        footer {{
            display: none !important;
        }}

        [data-testid="stToolbar"] {{
            display: none !important;
        }}

        header[data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]) {{
            height: 52px;
        }}

        [data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) {{
            display: flex !important;
            background: transparent !important;
        }}

        .stApp:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"] {{
            padding-left: 64px;
        }}

        .stApp:has([data-testid="stExpandSidebarButton"]) [data-testid="stSidebar"] {{
            min-width: 0 !important;
            max-width: 0 !important;
            width: 0 !important;
            border-right: 0;
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--pp-sidebar);
            background-image: url("data:image/png;base64,{background_data}");
            background-repeat: no-repeat;
            background-size: auto 51%;
            background-position: left bottom;
            border-right: 1px solid var(--pp-border);
            min-width: 284px;
            max-width: 284px;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgb(17 18 37 / 82%);
        }}

        [data-testid="stSidebarContent"] {{
            padding: 0;
        }}

        [data-testid="stSidebarUserContent"] {{
            position: relative;
            top: -40px;
        }}

        [data-testid="stSidebarCollapseButton"] {{
            display: flex !important;
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 2;
        }}

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] * {{
            visibility: visible !important;
        }}

        .brand-lockup {{
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 22px 26px 18px;
            border-bottom: 1px solid var(--pp-border);
            background: rgb(17 18 37 / 90%);
        }}

        .brand-name {{
            color: var(--pp-ink);
            font-family: "Courier New", ui-monospace, monospace;
            font-size: 27px;
            line-height: 1;
            font-weight: 800;
            letter-spacing: .02em;
        }}

        .brand-kicker {{
            color: var(--pp-purple);
            font-size: 10px;
            font-weight: 800;
            letter-spacing: .16em;
            margin-top: 10px;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] {{
            padding-top: 10px;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
            display: none;
        }}

        [data-testid="stSidebar"] div[role="radiogroup"] {{
            align-items: stretch;
            gap: 0;
        }}

        [data-testid="stSidebar"] div[role="radiogroup"] label {{
            box-sizing: border-box;
            width: 100%;
            min-height: 62px;
            padding: 0 26px;
            margin: 0;
            border-left: 4px solid transparent;
            background-color: rgb(17 18 37 / 78%);
            color: var(--pp-muted);
            font-size: 16px;
            transition: background-color 120ms ease, color 120ms ease;
        }}

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: rgb(35 35 62 / 92%);
            color: var(--pp-ink);
        }}

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            border-left-color: var(--pp-coral);
            background-color: rgb(39 38 67 / 96%);
            color: var(--pp-coral);
            font-weight: 750;
        }}

        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {{
            display: none !important;
        }}

        [data-testid="stSidebar"] [data-testid="stRadioOption"] > span {{
            display: none !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 1260px;
            padding: 28px 32px 52px;
        }}

        h1 {{
            color: var(--pp-ink) !important;
            font-size: 43px !important;
            line-height: 1.12 !important;
            letter-spacing: -.025em !important;
            margin: 0 0 8px !important;
            padding: 0 !important;
        }}

        h2, h3, h4 {{
            color: var(--pp-purple) !important;
        }}

        h4 {{
            font-size: 20px !important;
            margin-top: 8px !important;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--pp-muted);
            font-size: 15px;
            margin-bottom: 6px;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: var(--pp-border) !important;
            border-radius: 8px !important;
            background: var(--pp-panel);
            box-shadow: none !important;
        }}

        [data-testid="stMetric"] {{
            padding: 18px 20px;
            min-height: 92px;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--pp-muted);
            font-size: 14px;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--pp-ink);
            font-size: 30px;
            font-weight: 800;
        }}

        .st-key-overview_summary {{
            margin-top: 8px;
            margin-bottom: 14px;
        }}

        .st-key-model_recommendation {{
            margin-top: -20px;
            margin-bottom: 14px;
        }}

        .st-key-prediction_form_shell,
        .st-key-prediction_result {{
            margin-top: -28px;
        }}

        .st-key-overview_chart,
        .st-key-overview_signals,
        .st-key-model_chart,
        .st-key-model_recall,
        .st-key-prediction_form_shell,
        .st-key-prediction_result {{
            padding: 12px 18px 18px;
        }}

        .st-key-model_recommendation {{
            background: #14251F !important;
            border-color: #86BC32 !important;
        }}

        .st-key-model_recommendation [data-testid="stMetricValue"] {{
            color: var(--pp-lime);
        }}

        .st-key-overview_note,
        .st-key-model_note {{
            margin-top: 20px;
        }}

        [data-testid="stForm"] {{
            border: 0;
            padding: 0;
        }}

        [data-testid="stWidgetLabel"] p {{
            color: #E8E4F3;
            font-size: 14px;
            font-weight: 650;
        }}

        [data-baseweb="input"],
        [data-baseweb="select"] > div {{
            background: #24253A !important;
            border-color: #484A65 !important;
        }}

        [data-baseweb="input"]:focus-within,
        [data-baseweb="select"] > div:focus-within {{
            border-color: var(--pp-purple) !important;
            box-shadow: 0 0 0 1px var(--pp-purple) !important;
        }}

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {{
            border-radius: 6px;
            min-height: 44px;
            font-weight: 750;
        }}

        [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
            background: var(--pp-coral);
            border-color: #FFA0A3;
            color: #FFFFFF;
        }}

        [data-testid="stAlert"] {{
            border-radius: 7px;
            border: 1px solid var(--pp-border);
            background: #1B1C31;
            color: var(--pp-ink);
        }}

        .result-label {{
            color: var(--pp-purple);
            font-weight: 800;
            font-size: 18px;
            margin-bottom: 10px;
        }}

        .result-value {{
            color: var(--pp-lime);
            font-weight: 850;
            font-size: 36px;
            line-height: 1.1;
            margin-bottom: 8px;
        }}

        .caveat {{
            color: var(--pp-muted);
            font-size: 13px;
        }}

        @media (max-width: 900px) {{
            [data-testid="stSidebar"] {{
                min-width: 245px;
                max-width: 245px;
            }}

            [data-testid="stMainBlockContainer"] {{
                padding: 30px 20px 42px;
            }}

            h1 {{
                font-size: 34px !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-name">Player Pulse</div>
            <div class="brand-kicker">ENGAGEMENT INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Navigate",
        ("Overview", "Predict Player", "Model Insights"),
        label_visibility="collapsed",
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=10)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)


def engagement_mix_figure() -> plt.Figure:
    labels = ["Low", "Medium", "High"]
    values = [10_324, 19_374, 10_336]
    fig, axis = plt.subplots(figsize=(7.5, 6.0), facecolor=PANEL)
    bars = axis.bar(labels, values, color=[ENGAGEMENT_COLORS[label] for label in labels], width=0.62)
    _style_axis(axis)
    axis.set_ylim(0, 22_000)
    axis.set_ylabel("Players", fontsize=11)
    axis.set_xlabel("Engagement level", fontsize=11)
    axis.yaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 480,
            f"{value:,}",
            ha="center",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.1)
    return fig


def behavioural_signals_figure() -> plt.Figure:
    labels = ["Low", "Medium", "High"]
    colors = [ENGAGEMENT_COLORS[label] for label in labels]
    values = {
        "Sessions per week": [4.53, 9.55, 14.25],
        "Avg. session duration (min)": [66.88, 89.86, 131.92],
    }
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.0), facecolor=PANEL)
    for axis, (title, measurements) in zip(axes, values.items(), strict=True):
        bars = axis.barh(labels, measurements, color=colors, height=0.48)
        _style_axis(axis)
        axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
        axis.grid(axis="y", visible=False)
        axis.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left", pad=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(measurements) * 1.28)
        axis.set_xticks([])
        for bar, measurement in zip(bars, measurements, strict=True):
            axis.text(
                measurement + max(measurements) * 0.035,
                bar.get_y() + bar.get_height() / 2,
                f"{measurement:.2f}",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )
    fig.tight_layout(pad=1.3, h_pad=1.6)
    return fig


def probability_figure(probabilities: Mapping[str, float]) -> plt.Figure:
    labels = [label for label in ("Low", "Medium", "High") if label in probabilities]
    values = [probabilities[label] for label in labels]
    fig, axis = plt.subplots(figsize=(5.4, 3.15), facecolor=PANEL)
    bars = axis.bar(labels, values, color=[ENGAGEMENT_COLORS[label] for label in labels], width=0.58)
    _style_axis(axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability", fontsize=10)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.0%}",
            ha="center",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.1)
    return fig


def model_comparison_figure(table: pd.DataFrame) -> plt.Figure:
    ranked = table.sort_values("macro_f1", ascending=True)
    colors = [
        LIME if model == "Random Forest" else CORAL if model == "Histogram Gradient Boosting" else BLUE
        for model in ranked["model"]
    ]
    labels = ranked["model"].str.replace(" (Baseline)", "\n(Baseline)", regex=False)
    fig, axis = plt.subplots(figsize=(7.6, 5.3), facecolor=PANEL)
    bars = axis.barh(labels, ranked["macro_f1"], color=colors, height=0.52)
    _style_axis(axis)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0.70, 0.94)
    axis.set_xlabel("Test macro F1", fontsize=11)
    for bar, value in zip(bars, ranked["macro_f1"], strict=True):
        axis.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.2)
    return fig


def random_forest_recall_figure() -> plt.Figure:
    confusion = np.array([[1847, 158, 60], [109, 3668, 98], [73, 171, 1823]])
    labels = ["Low", "Medium", "High"]
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    fig, axis = plt.subplots(figsize=(5.1, 3.6), facecolor=PANEL)
    bars = axis.barh(labels, recalls, color=[ENGAGEMENT_COLORS[label] for label in labels], height=0.5)
    _style_axis(axis)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0.75, 1.0)
    axis.set_xlabel("Recall", fontsize=10)
    axis.invert_yaxis()
    for bar, value in zip(bars, recalls, strict=True):
        axis.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout(pad=1.1)
    return fig
