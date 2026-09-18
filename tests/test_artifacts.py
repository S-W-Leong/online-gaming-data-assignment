import json

import pytest

import prototype.artifacts as artifacts
from prototype.artifacts import ArtifactLoadError, load_metadata, load_models, validate_artifact_contract


VALID_NUMERIC_FEATURES = [
    "Age",
    "InGamePurchases",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked",
]
VALID_CATEGORICAL_FEATURES = ["Gender", "Location", "GameGenre", "GameDifficulty"]


def valid_metadata():
    return {
        "models": [
            "Logistic Regression (Baseline)",
            "K-Nearest Neighbours",
            "Decision Tree",
            "Random Forest",
            "Histogram Gradient Boosting",
            "RBF Support Vector Machine",
        ],
        "numeric_features": VALID_NUMERIC_FEATURES,
        "categorical_features": VALID_CATEGORICAL_FEATURES,
    }


def valid_models():
    return {model_name: object() for model_name in valid_metadata()["models"]}


def test_load_metadata_rejects_a_missing_file(tmp_path):
    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        load_metadata(tmp_path / "missing.json")


def test_validate_artifact_contract_rejects_missing_model():
    metadata = {"models": ["Random Forest"]}
    with pytest.raises(ArtifactLoadError, match="missing models"):
        validate_artifact_contract({}, metadata)


def test_load_metadata_reads_json(tmp_path):
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"models": ["Random Forest"]}), encoding="utf-8")
    assert load_metadata(metadata_file)["models"] == ["Random Forest"]


def test_load_metadata_rejects_non_object_json(tmp_path):
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        load_metadata(metadata_file)


@pytest.mark.parametrize("models_metadata", [None, "Random Forest", ["Random Forest", 1]])
def test_validate_artifact_contract_rejects_malformed_models_metadata(models_metadata):
    with pytest.raises(ArtifactLoadError, match="models metadata must be a list or tuple of strings"):
        validate_artifact_contract({}, {"models": models_metadata})


@pytest.mark.parametrize("feature_key", ["numeric_features", "categorical_features"])
def test_validate_artifact_contract_rejects_missing_feature_metadata(feature_key):
    metadata = valid_metadata()
    del metadata[feature_key]

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        validate_artifact_contract(valid_models(), metadata)


@pytest.mark.parametrize(
    ("feature_key", "feature_values"),
    [
        ("numeric_features", "Age"),
        ("categorical_features", ["Gender", "Location", "GameGenre", ""]),
    ],
)
def test_validate_artifact_contract_rejects_invalid_feature_metadata(feature_key, feature_values):
    metadata = valid_metadata()
    metadata[feature_key] = feature_values

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        validate_artifact_contract(valid_models(), metadata)


def test_validate_artifact_contract_rejects_unsupported_feature_metadata():
    metadata = valid_metadata()
    metadata["numeric_features"] = [*VALID_NUMERIC_FEATURES[:-1], "UnsupportedFeature"]

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        validate_artifact_contract(valid_models(), metadata)


def test_load_models_normalizes_incompatible_deserialization_errors(monkeypatch, tmp_path):
    def raise_missing_model_module(_path):
        raise ModuleNotFoundError("saved_model_dependency")

    monkeypatch.setattr(artifacts.joblib, "load", raise_missing_model_module)

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        load_models(tmp_path / "model.joblib")


def test_load_models_normalizes_invalid_pickle_opcode_errors(monkeypatch, tmp_path):
    def raise_invalid_pickle_opcode(_path):
        raise KeyError(118)

    monkeypatch.setattr(artifacts.joblib, "load", raise_invalid_pickle_opcode)

    with pytest.raises(ArtifactLoadError, match="Run online_gaming_analysis.ipynb"):
        load_models(tmp_path / "model.joblib")


def test_load_models_identifies_a_git_lfs_pointer(tmp_path):
    model_file = tmp_path / "model.joblib"
    model_file.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactLoadError, match="git lfs pull"):
        load_models(model_file)
