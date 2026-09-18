import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from prototype.contracts import METADATA_PATH, MODEL_BUNDLE_PATH, MODEL_NAMES


SUPPORTED_NUMERIC_FEATURES = (
    "Age",
    "InGamePurchases",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked",
)
SUPPORTED_CATEGORICAL_FEATURES = ("Gender", "Location", "GameGenre", "GameDifficulty")


class ArtifactLoadError(RuntimeError):
    """Raised when a persisted training artifact cannot safely be used."""


def _regeneration_message(path: Path) -> str:
    return f"Required artifact is unavailable: {path}. Run online_gaming_analysis.ipynb to regenerate model artifacts."


def _git_lfs_message(path: Path) -> str:
    return (
        f"Required model artifact was not downloaded: {path}. "
        "Install Git LFS if necessary, then run `git lfs pull` from the project root."
    )


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(_regeneration_message(path)) from exc
    if not isinstance(metadata, dict):
        raise ArtifactLoadError(_regeneration_message(path))
    return metadata


@st.cache_resource(show_spinner="Loading saved engagement models…")
def load_models(path: Path = MODEL_BUNDLE_PATH) -> dict[str, Any]:
    if _is_git_lfs_pointer(path):
        raise ArtifactLoadError(_git_lfs_message(path))

    try:
        loaded = joblib.load(path)
    except (OSError, ValueError, EOFError, KeyError, pickle.UnpicklingError, ImportError, AttributeError) as exc:
        raise ArtifactLoadError(_regeneration_message(path)) from exc
    if not isinstance(loaded, dict):
        raise ArtifactLoadError(f"Model bundle must be a dictionary: {path}")
    return loaded


def validate_artifact_contract(models: dict[str, Any], metadata: dict[str, Any]) -> None:
    metadata_models = metadata.get("models") if isinstance(metadata, dict) else None
    if not isinstance(metadata_models, (list, tuple)) or not all(
        isinstance(model_name, str) for model_name in metadata_models
    ):
        raise ArtifactLoadError(
            "Saved model contract is incomplete; models metadata must be a list or tuple of strings. "
            "Run online_gaming_analysis.ipynb to regenerate model artifacts."
        )

    missing_models = set(MODEL_NAMES) - set(models)
    missing_metadata = set(MODEL_NAMES) - set(metadata_models)
    if missing_models or missing_metadata:
        raise ArtifactLoadError(
            f"Saved model contract is incomplete; missing models: {sorted(missing_models | missing_metadata)}. "
            "Run online_gaming_analysis.ipynb to regenerate model artifacts."
        )

    for feature_key, supported_features in (
        ("numeric_features", SUPPORTED_NUMERIC_FEATURES),
        ("categorical_features", SUPPORTED_CATEGORICAL_FEATURES),
    ):
        features = metadata.get(feature_key)
        if not isinstance(features, (list, tuple)) or not all(
            isinstance(feature, str) and feature.strip() for feature in features
        ):
            raise ArtifactLoadError(
                f"Saved model contract is incomplete; {feature_key} metadata must be a list or tuple of non-empty strings. "
                "Run online_gaming_analysis.ipynb to regenerate model artifacts."
            )
        if len(features) != len(supported_features) or set(features) != set(supported_features):
            raise ArtifactLoadError(
                f"Saved model contract is incomplete; {feature_key} metadata does not match the supported input features. "
                "Run online_gaming_analysis.ipynb to regenerate model artifacts."
            )
