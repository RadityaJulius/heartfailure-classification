import os
import json
from typing import Tuple, List, Optional # Untuk tipe data

import joblib # Untuk memuat model  
import numpy as np
import pandas as pd
import streamlit as st

# Standardized directories and file paths 
MODELS_DIR = "models"
DATA_DIR = "data"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "heart_failure_svm.joblib")
DEFAULT_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
DEFAULT_METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "heart_failure.csv")

# Feature order expected by the trained model (derived from main.py preprocessing)
FEATURE_ORDER: List[str] = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

# Untuk memastikan direktori data dan model ada
def ensure_paths() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

# Untuk memuat model dan scaler
@st.cache_resource(show_spinner=False) # Untuk caching model dan scaler
def load_model_and_scaler(
    model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH
) -> Tuple[Optional[object], Optional[object], List[str]]: 
    """
    Memuat model dan scaler dengan validasi dasar dan kembalikan bersama dengan peringatan.
    Dicached untuk menghindari pengambilan ulang disk IO.
    """
    warnings: List[str] = [] # Untuk menyimpan peringatan
    model = None
    scaler = None

    if not os.path.exists(model_path): # Jika model tidak ditemukan
        warnings.append(f"Model file not found at {model_path}") # Menambahkan peringatan
    else: # Jika model ditemukan
        try: # Untuk mengecek apakah model dapat dimuat
            model = joblib.load(model_path)
        except Exception as e:
            warnings.append(f"Failed to load model: {e}") # Menambahkan peringatan

    if not os.path.exists(scaler_path): # Jika scaler tidak ditemukan
        warnings.append(f"Scaler file not found at {scaler_path}") # Menambahkan peringatan
    else:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            warnings.append(f"Failed to load scaler: {e}")

    return model, scaler, warnings

# Untuk memastikan kolom sesuai dengan FEATURE_ORDER
def _coerce_dataframe_to_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_ORDER if c not in df.columns] # Untuk mengecek apakah kolom sesuai dengan FEATURE_ORDER
    if missing: # Jika ada kolom yang hilang
        raise ValueError(f"Missing required features: {missing}") # Menampilkan error
    # Reorder and select
    return df[FEATURE_ORDER]

# Untuk memproses input satu sampel
def preprocess_single_input(scaler, df_row: pd.DataFrame) -> np.ndarray:
    """
    df_row harus berupa DataFrame dengan satu baris dan kolom yang sesuai dengan FEATURE_ORDER.
    """
    df_row = _coerce_dataframe_to_feature_order(df_row) 
    arr = df_row.values
    return scaler.transform(arr)

# Untuk melakukan prediksi satu sampel
def predict_single(model, scaler, df_row: pd.DataFrame) -> Tuple[int, float]:
    """
    Returns predicted class label and probability of class=1 if available.
    """
    X = preprocess_single_input(scaler, df_row)
    y_pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
    else:
        # Fallback: use decision_function scaled to 0-1 via logistic approximation
        if hasattr(model, "decision_function"):
            from math import exp
            d = float(model.decision_function(X)[0])
            prob = 1.0 / (1.0 + exp(-d))
        else:
            prob = 1.0 if y_pred == 1 else 0.0
    return int(y_pred), float(prob)

# Untuk melakukan prediksi batch
def predict_batch_dataframe(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    X_df = _coerce_dataframe_to_feature_order(df.copy())
    X_scaled = scaler.transform(X_df.values)
    preds = model.predict(X_scaled)

    # Probabilitas terbaik
    if hasattr(model, "predict_proba"): # Jika model memiliki predict_proba
        probs = model.predict_proba(X_scaled)[:, 1] # Menghitung probabilitas
    elif hasattr(model, "decision_function"): # Jika model memiliki decision_function
        from scipy.special import expit
        probs = expit(model.decision_function(X_scaled)) # Menghitung probabilitas
    else: # Jika model tidak memiliki predict_proba atau decision_function
        probs = (preds == 1).astype(float) # Menghitung probabilitas

    out = df.copy()
    out["prediction"] = preds.astype(int)
    out["probability_class_1"] = probs.astype(float)
    return out

# Untuk memuat dataset
@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

# Untuk memuat metrics
@st.cache_data(show_spinner=False)
def load_metrics_cached(path: str = DEFAULT_METRICS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)