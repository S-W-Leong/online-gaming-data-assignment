import pytest

from prototype.prediction import build_feature_frame, engagement_guidance, predict_player


class StubModel:
    classes_ = ["High", "Low", "Medium"]

    def predict(self, frame):
        assert list(frame.columns) == ["Age", "Gender"]
        return ["Low"]

    def predict_proba(self, frame):
        return [[0.1, 0.7, 0.2]]


METADATA = {"numeric_features": ["Age"], "categorical_features": ["Gender"]}


def test_build_feature_frame_requires_exact_feature_set():
    with pytest.raises(ValueError, match="Missing required features: Gender"):
        build_feature_frame({"Age": 22}, METADATA)


def test_predict_player_aligns_probabilities_to_target_labels():
    result = predict_player(StubModel(), {"Age": 22, "Gender": "Female"}, METADATA)

    assert result.label == "Low"
    assert result.probabilities == {"Low": 0.7, "Medium": 0.2, "High": 0.1}


def test_low_engagement_guidance_prioritises_reengagement():
    assert "re-engagement" in engagement_guidance("Low").lower()
