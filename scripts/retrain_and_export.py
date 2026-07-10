import os  # Untuk operasi sistem
import json  # Untuk serialisasi dan deserialisasi JSON
import joblib  # Untuk menyimpan dan memuat model
import pandas as pd  # Untuk manipulasi data
import numpy as np  # Untuk operasi matematika

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)  # Untuk pembagian data
from sklearn.svm import SVC  # Untuk SVM
from sklearn.preprocessing import StandardScaler  # Untuk skalering fitur
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Untuk evaluasi model
from imblearn.over_sampling import SMOTE  # Untuk balancing data

from scipy.stats.mstats import winsorize  # Untuk menangani outlier dengan winsorizing

# Deterministic seeds untuk reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Standardized paths untuk data dan model
DATA_DIR = "data"
MODELS_DIR = "models"
STANDARD_DATASET_PATH = os.path.join(DATA_DIR, "heart_failure.csv")
FALLBACK_DATASET_PATH = "heart_failure_clinical_records_dataset.csv"

MODEL_PATH = os.path.join(MODELS_DIR, "heart_failure_svm.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_ORDER = [
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
    "time", # Lama observasi
]


def detect_outliers(df: pd.DataFrame):
    info = {}
    # looping untuk setiap kolom numerik, kecuali target
    for col in df.select_dtypes(include=["number"]).columns:
        if col != "DEATH_EVENT":  # kolom target
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR  # Batas bawah & atas
            cnt = df[(df[col] < lb) | (df[col] > ub)].shape[0]  # Jumlah outlier
            if cnt > 0:
                info[col] = cnt  # Simpan jumlah outlier per kolom
    return info


def winsorize_outliers(df: pd.DataFrame, lower=0.05, upper=0.95):
    # Melakukan Winsorizing hanya pada fitur yang memiliki outlier berdasarkan IQR
    df_w = df.copy()
    outlier_info = detect_outliers(df)
    for col in outlier_info.keys():  # Hanya proses fitur yang memiliki outlier
        df_w[col] = winsorize(df_w[col], limits=(lower, 1 - upper))
    return df_w


# Firefly Algorithm (ported from main.py with light cleanups)
from numpy.random import default_rng


class FireflyAlgorithm:
    def __init__(
        self, pop_size=30, alpha=0.2, betamin=1.0, gamma=1.0, seed=GLOBAL_SEED
    ):
        # Inisialisasi parameter Firefly Algorithm
        self.pop_size = pop_size  # Jumlah firefly (solusi dalam populasi)
        self.alpha = alpha  # Ukuran langkah acak (semakin kecil, pencarian semakin detail)
        self.betamin = betamin  # Nilai minimum daya tarik antar firefly (semakin besar, semakin firefly lebih cepat berkumpul)
        self.gamma = gamma  # Mengontrol seberapa cepat cahaya/firefly meredup saat jarak bertambah ()
        self.rng = default_rng(seed)  # RNG supaya hasil bisa direplicate dari seed

    def optimize(self, function, dim, lb, ub, max_evals):
        # Inisialisasi posisi semua firefly secara acak dalam batas bawah (lb) dan atas (ub)
        fireflies = self.rng.uniform(
            lb, ub, (self.pop_size, dim)
        )  # Menghasilkan angka acak dari distribusi uniform (rata) dalam rentang [lb, ub)

        # Hitung nilai fitness untuk setiap firefly
        intensity = np.apply_along_axis(function, 1, fireflies) # (svm_fitness (C dan Gamma), 1 = per baris, array2D)

        # Mencari firefly dengan nilai fitness terbaik (paling rendah)
        best_idx = np.argmin(
            intensity
        )  # Mengembalikan indeks elemen dengan nilai terkecil dalam array.
        best_solution = fireflies[best_idx].copy()
        best_fitness = float(intensity[best_idx])

        evaluations = self.pop_size  # Jumlah evaluasi awal = jumlah firefly
        new_alpha = self.alpha  # Ukuran langkah awal
        search_range = np.array(ub) - np.array(lb)  # Jarak ruang pencarian tiap dimensi

        # Lanjutkan pencarian sampai mencapai batas maksimum evaluasi
        while evaluations < max_evals:
            new_alpha *= (
                0.97  # Kurangi langkah alpha secara bertahap agar solusi makin bagus
            )

            # Bandingkan setiap firefly dengan yang lainnya
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    # Jika firefly j lebih baik dari firefly i, maka firefly i akan bergerak ke arah j
                    if intensity[i] >= intensity[j]:
                        # Hitung jarak kuadrat antara firefly i dan j
                        r = np.sum(np.square(fireflies[i] - fireflies[j]))

                        # Hitung daya tarik (betamin) antara firefly i dan j
                        beta = self.betamin * np.exp(-self.gamma * r)

                        # Buat langkah acak dan gerakkan firefly i
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range 
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps

                        # Pastikan posisi baru tetap dalam batas lb dan ub
                        fireflies[i] = np.clip(
                            fireflies[i], lb, ub
                        )  # Memotong nilai agar tidak keluar dari batas tertentu.

                        # Evaluasi solusi baru dari firefly i
                        intensity[i] = function(fireflies[i])
                        evaluations += 1

                        # Perbarui solusi terbaik jika ditemukan yang lebih baik
                        if intensity[i] < best_fitness:
                            best_fitness = float(intensity[i])
                            best_solution = fireflies[i].copy()

            # Print informasi evaluasi setiap iterasi
            print(
                f"Evaluation: {evaluations}, Best Acc: {-best_fitness:.4f}, C: {best_solution[0]:.4f}, Sigma: {best_solution[1]:.4f}"
            )

        # Return solusi terbaik dan akurasinya (dalam bentuk positif karena fitnessnya berbentuk negatif akurasi)
        return best_solution, -best_fitness


# Load dataset
def load_dataset() -> pd.DataFrame:
    # Membuat direktori jika belum ada
    os.makedirs(DATA_DIR, exist_ok=True)

    # Memeriksa apakah dataset standar ada (jika ada, gunakan dataset standar)
    if os.path.exists(STANDARD_DATASET_PATH):
        return pd.read_csv(STANDARD_DATASET_PATH)

    # Jika tidak ada, periksa apakah dataset fallback ada (jika ada, gunakan dataset fallback)
    if os.path.exists(FALLBACK_DATASET_PATH):
        return pd.read_csv(FALLBACK_DATASET_PATH)

    # Jika tidak ada dataset standar maupun fallback, raise error
    raise FileNotFoundError(
        f"Dataset not found. Place CSV at '{STANDARD_DATASET_PATH}' or repo root as '{FALLBACK_DATASET_PATH}'."
    )


def prepare_features(df: pd.DataFrame):
    # Pastikan dataset memiliki kolom target 'DEATH_EVENT'
    if "DEATH_EVENT" not in df.columns:
        raise ValueError("Dataset must include 'DEATH_EVENT' target column.")

    # Winsorize outliers hanya pada fitur yang memiliki outlier
    df_clean = winsorize_outliers(df)

    # Drop kolom target dan scale fitur numerik
    X = df_clean.drop(columns=["DEATH_EVENT"])
    y = df_clean["DEATH_EVENT"].astype(int)

    # Pastikan semua fitur yang diperlukan ada
    missing = [c for c in FEATURE_ORDER if c not in X.columns]
    if missing:
        raise ValueError(f"Dataset missing required features: {missing}")
    X = X[FEATURE_ORDER]

    # Scale fitur numerik
    scaler = (
        StandardScaler()
    )  # Menghindari masalah dominasi jarak fitur besar yang membuat fitur kecil terabaikan
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, y.values, scaler


def main():
    # Membuat direktori jika belum ada
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    df = load_dataset()

    # Preprocess
    X_scaled, y, scaler = prepare_features(df)

    # Melakukan balancing dengan SMOTE
    smote = SMOTE(
        sampling_strategy=1.0, random_state=GLOBAL_SEED
    )  # Sampling strategy 1.0 berarti setiap kelas memiliki jumlah sampel yang sama
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        random_state=GLOBAL_SEED,
        stratify=y_res,  # 20% dari data di uji dan sisanya dilatih
    )

    # Definiskan fitness function untuk fungsi penilaian yang dipakai oleh algoritma optimasi
    def svm_fitness(params):
        C, sigma = (
            params  # C adalah parameter regularisasi, sigma adalah parameter kernel (gamma)
        )
        model = SVC(
            C=float(C),
            kernel="rbf",
            gamma=float(sigma),
            probability=True,
            random_state=GLOBAL_SEED,
        )  # Kernel RBF (Radial Basis Function)
        scores = cross_val_score(
            model, X_train, y_train, cv=10, scoring="accuracy"
        )  # Evaluasi model dengan 10-Fold CV
        return -np.mean(
            scores
        )  # Return fitness negatif karena kita ingin memaksimalkan akurasi

    lower_bounds = [1.0, 0.1]  # Batas bawah untuk C dan sigma
    upper_bounds = [3.0, 1.0]  # Batas atas untuk C dan sigma
    dimension = 2  # Jumlah parameter yang dioptimasi (C dan sigma)
    max_evals = 50  # Batas maksimum evaluasi

    # Inisialisasi Firefly Algorithm
    fa = FireflyAlgorithm(
        pop_size=30, alpha=0.2, betamin=1.0, gamma=1.0, seed=GLOBAL_SEED
    )

    # Optimisasi FA untuk mencari  C dan sigma terbaik
    best_params, best_cv_acc = fa.optimize(
        svm_fitness, dimension, lower_bounds, upper_bounds, max_evals
    )

    # Nilai C dan sigma terbaik
    optimal_C, optimal_sigma = float(best_params[0]), float(best_params[1])

    # Train final model with optimized params
    final_model = SVC(
        C=optimal_C,
        kernel="rbf",
        gamma=optimal_sigma,
        probability=True,
        random_state=GLOBAL_SEED,
    )

    # Evaluasi model dengan 5-Fold CV
    cv_scores_final = cross_val_score(
        final_model, X_train, y_train, cv=5, scoring="accuracy"
    )

    # Latih model dengan parameter terbaik
    final_model.fit(X_train, y_train)

    # Evaluasi model pada data uji
    y_pred = final_model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred, target_names=["Survived", "Died"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Simpan model dan scaler
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Simpan metrics
    metrics = {
        "accuracy_test": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "fa_optimized": {
            "optimal_C": optimal_C,
            "optimal_gamma": optimal_sigma,
            "best_cv_accuracy_during_search": float(best_cv_acc),
            "bounds": {"C": {"lb": 1.0, "ub": 3.0}, "gamma": {"lb": 0.1, "ub": 1.0}},
            "max_evals": max_evals,
            "pop_size": 30,
        },
        "cv_final_mean": float(np.mean(cv_scores_final)),
        "cv_final_std": float(np.std(cv_scores_final)),
        "feature_order": FEATURE_ORDER,
        "model_params": {
            "C": final_model.C,
            "kernel": final_model.kernel,
            "gamma": final_model.gamma,
            "probability": True,
            "random_state": GLOBAL_SEED,
        },
        "paths": {
            "model": MODEL_PATH,
            "scaler": SCALER_PATH,
            "dataset_used": (
                STANDARD_DATASET_PATH
                if os.path.exists(STANDARD_DATASET_PATH)
                else (
                    FALLBACK_DATASET_PATH
                    if os.path.exists(FALLBACK_DATASET_PATH)
                    else None
                )
            ),
        },
    }
    # Simpan metrics ke file
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete with Firefly-optimized SVM.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Test accuracy: {acc:.4f}")
    print(
        f"Final CV (5-fold) accuracy: {np.mean(cv_scores_final):.4f} ± {np.std(cv_scores_final):.4f}"
    )
    print(f"Best CV during FA search: {best_cv_acc:.4f}")


if __name__ == "__main__":
    main()
