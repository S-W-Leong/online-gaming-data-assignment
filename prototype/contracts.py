from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "all_engagement_models.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "all_engagement_models_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

MODEL_NAMES = (
    "Logistic Regression (Baseline)",
    "K-Nearest Neighbours",
    "Decision Tree",
    "Random Forest",
    "Histogram Gradient Boosting",
    "RBF Support Vector Machine",
)
TARGET_ORDER = ("Low", "Medium", "High")
ASSET_MANIFEST: Mapping[str, str] = {
    "Target distribution": "target_distribution.png",
    "Categorical distributions": "categorical_distributions.png",
    "Numeric distributions": "numeric_distributions.png",
    "Numeric correlation heatmap": "numeric_correlation_heatmap.png",
    "Numeric features by engagement": "numeric_features_by_target.png",
    "Engagement session heatmap": "engagement_session_heatmap.png",
    "Categorical composition by engagement": "categorical_composition_by_target.png",
    "Binned engagement rates": "binned_engagement_rates.png",
    "Playtime duration consistency": "playtime_duration_consistency.png",
    "Behavioural segment summary": "behavioural_segment_summary.png",
    "Model confusion matrices": "model_confusion_matrices.png",
    "Tuned model confusion matrices": "tuned_model_confusion_matrices.png",
}


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probabilities: dict[str, float] | None
    guidance: str
