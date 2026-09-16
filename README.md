# Online Gaming Engagement Classification

Coursework repository for **BMDS2003 Data Science**. The project applies the CRISP-DM process to a synthetic Online Gaming dataset and builds a multiclass classifier for the recorded `EngagementLevel` of each player profile: `Low`, `Medium`, or `High`.

The intended stakeholder is a game publisher's product and analytics team. Results are decision support for investigating player segments—not a causal explanation of engagement, a forecast of future retention, or an automated action system. The deployed prototype is available [here](https://online-gaming-data-assignment-demo.streamlit.app)

## Project objectives

- Audit and explore the supplied dataset, including data quality, class balance, distributions, and feature relationships.
- Prepare a reproducible, leakage-safe modelling dataset.
- Compare six multiclass classifiers using macro F1 as the primary selection metric:
  - Logistic Regression (baseline)
  - K-Nearest Neighbours
  - Decision Tree
  - Random Forest
  - Histogram Gradient Boosting
  - RBF Support Vector Machine
- Document the analysis, limitations, and business implications in a CRISP-DM-aligned report.
- Provide a simple Streamlit prototype that predicts a single player's engagement level and presents the saved model results.

The current recommended model is **Random Forest**, selected from the tuned comparison by macro F1. Treat this result cautiously: the dataset is synthetic, the target-generation procedure is undisclosed, and it has no time dimension.

## Repository structure

```text
.
├── online_gaming_behavior_dataset.csv  # Source dataset (40,034 synthetic profiles)
├── online_gaming_analysis.ipynb        # Reproducible CRISP-DM analysis and model export
├── requirements.txt                    # Python dependencies for analysis, prototype, and tests
├── prototype/                          # Streamlit deployment prototype
│   ├── app.py                          # Three-page user interface
│   ├── artifacts.py                    # Saved-model and metadata loading/validation
│   ├── prediction.py                   # Single-player feature validation and inference
│   ├── insights.py                     # Persisted model-comparison loading
│   └── contracts.py                    # Shared paths, model names, and feature contracts
├── models/
│   ├── all_engagement_models.joblib    # Persisted fitted pipelines (local/generated)
│   └── all_engagement_models_metadata.json
├── outputs/
│   ├── figures/                        # EDA and confusion-matrix images
│   └── tables/                         # Baseline and tuned model-comparison CSVs
├── tests/                              # Prototype unit and Streamlit smoke tests
```

## Modelling workflow

The notebook follows CRISP-DM:

1. **Business understanding** — frames engagement classification as analyst decision support.
2. **Data understanding** — validates the source schema and generates descriptive/EDA outputs.
3. **Data preparation** — excludes `PlayerID` and ambiguous `PlayTimeHours`; uses six numeric and four categorical features in leakage-safe scikit-learn pipelines.
4. **Modelling** — trains the baseline and five alternative classifiers.
5. **Evaluation** — compares macro F1, accuracy, class-level metrics, ROC-AUC, and confusion matrices; tuning uses stratified cross-validation.
6. **Deployment** — saves fitted pipelines, metadata, visuals, and evaluation tables for the Streamlit app.

The prototype intentionally loads saved artifacts only. It does **not** clean data, split data, tune hyperparameters, retrain, or recompute evaluation metrics at startup.

## Run the prototype locally (optional) 

Create a local environment and install dependencies.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal session before activating:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```cmd
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Start the app:

### macOS / Linux

```bash
.venv/bin/python -m streamlit run prototype/app.py
```

### Windows

```powershell
.\.venv\Scripts\python.exe -m streamlit run prototype/app.py
```

Or, after activating the virtual environment in PowerShell or Command Prompt, you can simply run:

```powershell
python -m streamlit run prototype/app.py
```

The app provides three pages:

- **Data Overview** — selected EDA figures and project context.
- **Predict Player** — a validated one-player form with selectable saved model, predicted class, confidence where available, and retention-oriented guidance.
- **Model Insights** — persisted evaluation metrics, baseline/recommended model labels, and confusion matrices.

### Required prototype artifacts

The prototype expects these generated assets:

- `models/all_engagement_models.joblib`
- `models/all_engagement_models_metadata.json`
- `outputs/tables/tuned_model_comparison.csv`
- figures in `outputs/figures/`

If an artifact is missing or invalid, the app shows an actionable message. Re-run the export/evaluation sections of `online_gaming_analysis.ipynb` in a compatible Jupyter environment to regenerate it. The `.joblib` bundle is intentionally local/generated and may not be present in a fresh Git checkout.

## Reproduce the analysis

Open `online_gaming_analysis.ipynb` with the environment above and run its cells in order. The notebook validates `online_gaming_behavior_dataset.csv`, creates figures/tables under `outputs/`, and exports the model bundle and metadata under `models/` once the relevant execution flags are enabled.

Do not use the prototype as a substitute for the notebook: the notebook is the source of truth for data preparation, training, tuning, and evaluation.


## Responsible use and limitations

- Predictions estimate the dataset's recorded label only; they do not prove why a player is engaged or predict future behaviour.
- The synthetic dataset and undisclosed label-generation method limit external validity and may introduce circularity.
- Do not use outputs to manipulate vulnerable players, encourage excessive play/spending, or make high-impact decisions without human review.
- Validate on real, time-ordered data and inspect subgroup performance before any operational use.
