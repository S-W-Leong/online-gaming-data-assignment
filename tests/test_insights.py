import pandas as pd
import pytest

from prototype.insights import (
    InsightsLoadError,
    decorate_comparison_table,
    load_comparison_table,
    recommended_model,
)


def test_recommended_model_uses_highest_macro_f1():
    table = pd.DataFrame({"model": ["A", "B"], "macro_f1": [0.81, 0.91]})

    assert recommended_model(table) == "B"


def test_decorated_table_marks_the_baseline_and_recommendation():
    table = pd.DataFrame({
        "model": ["Logistic Regression (Baseline)", "Random Forest"],
        "macro_f1": [0.81, 0.91],
    })

    decorated = decorate_comparison_table(table)

    assert decorated.loc[0, "role"] == "Baseline"
    assert decorated.loc[1, "role"] == "Recommended"


def test_load_comparison_table_reports_actionable_error_for_empty_csv(tmp_path):
    empty_table = tmp_path / "model_comparison.csv"
    empty_table.write_text("")

    with pytest.raises(InsightsLoadError, match="Run online_gaming_analysis.ipynb"):
        load_comparison_table(empty_table)


def test_load_comparison_table_rejects_a_header_only_table(tmp_path):
    comparison_table = tmp_path / "model_comparison.csv"
    comparison_table.write_text("model,macro_f1\n", encoding="utf-8")

    with pytest.raises(InsightsLoadError, match="Run online_gaming_analysis.ipynb"):
        load_comparison_table(comparison_table)


def test_load_comparison_table_rejects_all_nan_macro_f1_values(tmp_path):
    comparison_table = tmp_path / "model_comparison.csv"
    comparison_table.write_text("model,macro_f1\nA,\nB,\n", encoding="utf-8")

    with pytest.raises(InsightsLoadError, match="Run online_gaming_analysis.ipynb"):
        load_comparison_table(comparison_table)
