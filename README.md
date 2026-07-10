# Heart Failure Classification using SVM with Firefly Algorithm Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red.svg)](https://streamlit.io)

A machine learning pipeline that predicts heart failure outcomes (DEATH_EVENT) using a **Support Vector Machine (SVM)** with an RBF kernel, optimized via the **Firefly Algorithm (FA)**. A Streamlit web application provides an interactive interface for real-time predictions, data exploration, and model performance visualization.

## Actual Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **92.68%** |
| **Class 0 (Survived)** — Precision / Recall / F1 | 94.87% / 90.24% / 92.50% |
| **Class 1 (Died)** — Precision / Recall / F1 | 90.70% / 95.12% / 92.86% |
| **5-Fold CV Mean ± Std** | 86.11% ± 3.79% |
| **Best CV Accuracy During FA Search** | 86.77% |
| **Optimal C (regularization)** | 1.97 |
| **Optimal γ (gamma)** | 0.122 |

Confusion matrix (82 test samples after SMOTE + 80/20 split):

```
              Predicted
              Survived  Died
Actual Survived   37      4
       Died        2     39
```

## Project Structure

```
.
├── app/
│   ├── app.py              # Streamlit application (single-page, radio-nav)
│   └── utils.py            # Shared utilities, loaders, predictors
├── data/
│   └── heart_failure.csv   # Dataset (299 patients, 12 features)
├── models/
│   ├── heart_failure_svm.joblib  # Trained SVM model
│   ├── scaler.joblib             # Fitted StandardScaler
│   └── metrics.json              # Evaluation metrics (from training)
├── scripts/
│   └── retrain_and_export.py     # Full training pipeline with FA optimization
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/RadityaRizqullah/fa-svm-heartfailure-classification.git
   cd fa-svm-heartfailure-classification
   ```

2. **Create and activate a Python environment** (3.9+ recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Uses the [Heart Failure Clinical Records](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) dataset. The file is expected at `data/heart_failure.csv` with these columns:

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Patient age (years) |
| `anaemia` | binary | 1 = yes, 0 = no |
| `creatinine_phosphokinase` | numeric | CPK enzyme level (mcg/L) |
| `diabetes` | binary | 1 = yes, 0 = no |
| `ejection_fraction` | numeric | Percentage of blood leaving heart |
| `high_blood_pressure` | binary | 1 = yes, 0 = no |
| `platelets` | numeric | Platelet count (kiloplatelets/mL) |
| `serum_creatinine` | numeric | Serum creatinine level (mg/dL) |
| `serum_sodium` | numeric | Serum sodium level (mEq/L) |
| `sex` | binary | 1 = male, 0 = female |
| `smoking` | binary | 1 = yes, 0 = no |
| `time` | numeric | Follow-up period (days) |
| **`DEATH_EVENT`** | **binary** | **Target: 1 = deceased, 0 = survived** |

## Training (Retrain from Scratch)

To reproduce the full pipeline — winsorization, SMOTE balancing, Firefly Algorithm hyperparameter search, and final model export:

```bash
python scripts/retrain_and_export.py
```

This will:
1. Load `data/heart_failure.csv`
2. Detect and winsorize outliers (5th / 95th percentile)
3. Scale features with `StandardScaler`
4. Balance classes with SMOTE (1:1 ratio)
5. Run the Firefly Algorithm (population=30, max_evals=50) to search optimal `C ∈ [1, 3]` and `γ ∈ [0.1, 1.0]`
6. Train a final SVM (RBF, `probability=True`) with the best parameters
7. Save artifacts to:
   - `models/heart_failure_svm.joblib`
   - `models/scaler.joblib`
   - `models/metrics.json`

### Firefly Algorithm Details

The `FireflyAlgorithm` class implements a swarm-intelligence metaheuristic inspired by firefly bioluminescence:
- **Population size**: 30 fireflies
- **Movement**: Each firefly moves toward brighter ones (lower fitness = higher "brightness")
- **Parameters tuned**: `C` (regularization, [1.0–3.0]) and `γ` (RBF kernel width, [0.1–1.0])
- **Fitness**: Negative 10-fold cross-validation accuracy (minimized)
- **Attraction decay**: `β = β_min · exp(−γ · r²)` — brightness diminishes with distance

## Running the Streamlit App

```bash
streamlit run app/app.py
```

### App Sections

| Section | Description |
|---|---|
| **🏠 Prediction** | Single-patient form with real-time risk probability + optional batch CSV upload |
| **📊 Data Insights** | Dataset overview, class balance, distribution explorer, correlation heatmap |
| **🧪 Model Performance** | Saved metrics display + optional quick evaluation on a random split |
| **ℹ️ About** | Methodology summary and usage notes |

### Feature Order

The model expects input features in this exact order:

```python
["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
 "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex",
 "smoking", "time"]
```

The app's `utils.py` enforces this order to prevent misaligned predictions.

## Methodology

1. **Outlier Handling** — Winsorization at the 5th and 95th percentiles on IQR-detected outliers
2. **Feature Scaling** — `StandardScaler` (zero mean, unit variance)
3. **Class Balancing** — SMOTE oversampling to 1:1 ratio
4. **Hyperparameter Optimization** — Firefly Algorithm (swarm intelligence) searches optimal `C` and `γ`
5. **Model** — SVM with RBF kernel, `probability=True` for calibrated risk scores
6. **Validation** — 5-fold cross-validation on tuned model, held-out test set evaluation

## Troubleshooting

| Problem | Solution |
|---|---|
| Missing model/scaler | Run `python scripts/retrain_and_export.py` |
| Missing dataset | Place `heart_failure.csv` in `data/` (see Dataset section) |
| Feature mismatch | Ensure all 12 features are present with correct column names |
| Module / import errors | `pip install -r requirements.txt`; run from repo root |

## License

MIT
