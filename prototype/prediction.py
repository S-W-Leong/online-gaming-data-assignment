from typing import Any

import pandas as pd

from prototype.contracts import PredictionResult, TARGET_ORDER


def feature_defaults(metadata: dict[str, Any]) -> dict[str, object]:
    return {feature: 0.0 for feature in metadata["numeric_features"]} | {
        feature: "" for feature in metadata["categorical_features"]
    }


def build_feature_frame(values: dict[str, object], metadata: dict[str, Any]) -> pd.DataFrame:
    expected = metadata["numeric_features"] + metadata["categorical_features"]
    missing = [feature for feature in expected if feature not in values or values[feature] in (None, "")]
    unexpected = sorted(set(values) - set(expected))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected features: {', '.join(unexpected)}")
    return pd.DataFrame([{feature: values[feature] for feature in expected}])


def engagement_guidance(label: str) -> str:
    messages = {
        "Low": "Prioritise re-engagement outreach, such as a tailored incentive or content recommendation.",
        "Medium": "Monitor this player and test targeted content or progression prompts to build engagement.",
        "High": "Maintain satisfaction with relevant content and consider loyalty or advocacy opportunities.",
    }
    return messages.get(label, "Review the player profile before making a retention decision.")


def predict_player(model: Any, values: dict[str, object], metadata: dict[str, Any]) -> PredictionResult:
    frame = build_feature_frame(values, metadata)
    label = str(model.predict(frame)[0])
    probabilities = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        raw = model.predict_proba(frame)[0]
        aligned = dict(zip(model.classes_, raw, strict=True))
        probabilities = {target: float(aligned[target]) for target in TARGET_ORDER if target in aligned}
    return PredictionResult(label=label, probabilities=probabilities, guidance=engagement_guidance(label))
