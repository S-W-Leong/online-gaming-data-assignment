from pathlib import Path

import pandas as pd

from prototype.contracts import TABLES_DIR


class InsightsLoadError(RuntimeError):
    """Raised when persisted model-evaluation output is unavailable."""


def load_comparison_table(
    path: Path = TABLES_DIR / "tuned_model_comparison.csv",
) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise InsightsLoadError(
            f"Evaluation table is unavailable: {path}. "
            "Run online_gaming_analysis.ipynb to regenerate it."
        ) from exc
    if not {"model", "macro_f1"}.issubset(table.columns):
        raise InsightsLoadError("Evaluation table must contain model and macro_f1 columns.")

    macro_f1 = pd.to_numeric(table["macro_f1"], errors="coerce")
    if table.empty or not macro_f1.notna().any():
        raise InsightsLoadError(
            "Evaluation table must contain at least one row with a numeric macro_f1 value. "
            "Run online_gaming_analysis.ipynb to regenerate it."
        )
    table = table.copy()
    table["macro_f1"] = macro_f1
    return table.sort_values("macro_f1", ascending=False, ignore_index=True)


def recommended_model(table: pd.DataFrame) -> str:
    return str(table.loc[table["macro_f1"].idxmax(), "model"])


def decorate_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    decorated = table.copy()
    winner = recommended_model(decorated)
    decorated["role"] = "Candidate"
    decorated.loc[
        decorated["model"] == "Logistic Regression (Baseline)", "role"
    ] = "Baseline"
    decorated.loc[decorated["model"] == winner, "role"] = "Recommended"
    return decorated
