from prototype.contracts import ASSET_MANIFEST, MODEL_NAMES, TARGET_ORDER


def test_contract_constants_match_the_project_model_contract():
    assert MODEL_NAMES == (
        "Logistic Regression (Baseline)",
        "K-Nearest Neighbours",
        "Decision Tree",
        "Random Forest",
        "Histogram Gradient Boosting",
        "RBF Support Vector Machine",
    )
    assert TARGET_ORDER == ("Low", "Medium", "High")
    assert ASSET_MANIFEST["Target distribution"] == "target_distribution.png"
