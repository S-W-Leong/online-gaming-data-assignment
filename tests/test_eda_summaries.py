import pandas as pd
import nbformat


def load_notebook_helpers():
    notebook = nbformat.read("online_gaming_analysis.ipynb", as_version=4)
    helper_cell = next(
        cell for cell in notebook.cells
        if cell.metadata.get("tags") == ["eda-summary-helpers"]
    )
    namespace = {}
    exec(helper_cell.source, namespace)
    return namespace["cramers_v"], namespace["engagement_composition_table"]


def test_cramers_v_is_one_for_perfect_association():
    cramers_v, _ = load_notebook_helpers()
    table = pd.DataFrame([[10, 0], [0, 10]])

    assert cramers_v(table) == 1.0


def test_engagement_composition_has_one_row_per_feature_target_pair():
    _, engagement_composition_table = load_notebook_helpers()
    df = pd.DataFrame(
        {
            "Difficulty": ["Easy", "Easy", "Hard"],
            "EngagementLevel": ["Low", "High", "Low"],
        }
    )

    result = engagement_composition_table(df, "Difficulty")

    assert result[["Difficulty", "EngagementLevel", "Percentage"]].to_dict("records") == [
        {"Difficulty": "Easy", "EngagementLevel": "Low", "Percentage": 50.0},
        {"Difficulty": "Easy", "EngagementLevel": "High", "Percentage": 50.0},
        {"Difficulty": "Hard", "EngagementLevel": "Low", "Percentage": 100.0},
    ]
