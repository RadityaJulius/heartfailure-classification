import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # SMOTE untuk oversampling
import random

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

df.head()

df.tail()

df.info()

df.describe()

num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"\nJumlah data duplikat: {num_duplicates}")
else:
    print("\nTidak ada data duplikat.")

plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Korelasi Antar Fitur")
plt.show()

def detect_outliers(df):
    outlier_info = {}
    # looping untuk setiap kolom numerik, kecuali target
    for col in df.select_dtypes(include=["number"]).columns:
        if col != "DEATH_EVENT": # kolom target
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR # Batas bawah & atas
            num_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0] # Jumlah outlier

            if num_outliers > 0:
                outlier_info[col] = num_outliers # Simpan jumlah outlier per kolom

    return outlier_info

def plot_outliers(df, outlier_info):
    if not outlier_info:
        print("Tidak ada outlier untuk divisualisasikan")
        return
    # Buat subplot untuk tiap kolom yang memiliki outlier
    plt.figure(figsize=(15, 8))
    for i, (col, count) in enumerate(outlier_info.items(), 1):
        plt.subplot(2, (len(outlier_info) // 2) + 1, i)
        sns.boxplot(y=df[col], width=0.4)
        plt.title(f"{col}\n{count} Outliers", fontsize=12)
        plt.ylabel("")

    plt.tight_layout()
    plt.show()

# call function
outliers = detect_outliers(df)
plot_outliers(df, outliers)

from scipy.stats.mstats import winsorize

def winsorize_outliers(df, lower_percentile=0.05, upper_percentile=0.95):
    # Melakukan Winsorizing hanya pada fitur yang memiliki outlier berdasarkan IQR
    df_winsorized = df.copy()
    outlier_info = detect_outliers(df)

    for col in outlier_info.keys(): # Hanya proses fitur yang memiliki outlier
        df_winsorized[col] = winsorize(df_winsorized[col], limits=(lower_percentile, 1 - upper_percentile))

    return df_winsorized

# Winsorizing fitur yang memiliki outlier
df_clean = winsorize_outliers(df)

print(f"\nJumlah data sebelum winsorizing: {df.shape[0]}")
print(f"Jumlah data setelah winsorizing: {df_clean.shape[0]}")

# Menentukan fitur (X) dan target (y)
X = df_clean.drop(columns=["DEATH_EVENT"])
y = df_clean["DEATH_EVENT"]

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cek distribusi kelas
class_counts = pd.Series(y).value_counts()
print("\nDistribusi Kelas (Sebelum Split):")
print(class_counts)

# Visualisasi distribusi kelas sebelum SMOTE
plt.figure(figsize=(6, 4))
sns.barplot(x=y.value_counts().index, y=y.value_counts().values, palette="coolwarm")
plt.xticks([0, 1], ["Survived", "Died"])
plt.ylabel("Jumlah Sampel")
plt.title("Distribusi Kelas Target Sebelum SMOTE")
plt.show()

# Melakukan SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Cek distribusi kelas setelah SMOTE
class_counts_resampled = pd.Series(y_resampled).value_counts()
print("\nDistribusi Kelas Setelah SMOTE (Sebelum Split):")
print(class_counts_resampled)

# Visualisasi distribusi kelas setelah SMOTE
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts_resampled.index, y=class_counts_resampled.values, palette="coolwarm")
plt.xticks([0, 1], ["Survived", "Died"])
plt.ylabel("Jumlah Sampel")
plt.title("Distribusi Kelas Target Setelah SMOTE")
plt.show()

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

from numpy.random import default_rng

class FireflyAlgorithm:
    def __init__(self, pop_size=30, alpha=0.2, betamin=1.0, gamma=1.0, seed=None):
        # Inisialisasi parameter Firefly Algorithm
        self.pop_size = pop_size        # Jumlah firefly (solusi dalam populasi)
        self.alpha = alpha              # Ukuran langkah acak (semakin kecil, semakin teliti)
        self.betamin = betamin          # Nilai minimum daya tarik antar firefly
        self.gamma = gamma              # Mengontrol seberapa cepat cahaya/firefly meredup saat jarak bertambah
        self.rng = default_rng(seed)    # RNG supaya hasil bisa direplicate dari seed

    def optimize(self, function, dim, lb, ub, max_evals):
        # Inisialisasi posisi semua firefly secara acak dalam batas bawah (lb) dan atas (ub)
        fireflies = self.rng.uniform(lb, ub, (self.pop_size, dim))

        # Hitung nilai fitness untuk setiap firefly
        intensity = np.apply_along_axis(function, 1, fireflies)

        # Mencari firefly dengan nilai fitness terbaik (paling rendah)
        best_idx = np.argmin(intensity)
        best_solution = fireflies[best_idx]
        best_fitness = intensity[best_idx]

        evaluations = self.pop_size                 # Jumlah evaluasi awal = jumlah firefly
        new_alpha = self.alpha                      # Ukuran langkah awal
        search_range = np.array(ub) - np.array(lb)  # Jarak ruang pencarian tiap dimensi

        # Lanjutkan pencarian sampai mencapai batas maksimum evaluasi
        while evaluations < max_evals:
            new_alpha *= 0.97  # Kurangi langkah alpha secara bertahap agar solusi makin bagus

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
                        fireflies[i] = np.clip(fireflies[i], lb, ub)

                        # Evaluasi solusi baru dari firefly i
                        intensity[i] = function(fireflies[i])
                        evaluations += 1

                        # Perbarui solusi terbaik jika ditemukan yang lebih baik
                        if intensity[i] < best_fitness:
                            best_fitness = intensity[i]
                            best_solution = fireflies[i]

            # Print informasi evaluasi setiap iterasi
            print(f"Evaluation: {evaluations}, Best Acc: {-best_fitness:.4f}, C: {best_solution[0]:.4f}, Sigma: {best_solution[1]:.4f}")

        # Return solusi terbaik dan akurasinya (dalam bentuk positif karena fitnessnya berbentuk negatif akurasi)
        return best_solution, -best_fitness
    
def svm_fitness(params):
    C, sigma = params
    model = SVC(C=C, kernel="rbf", gamma=sigma)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy") # Evaluasi model dengan 10-Fold CV
    return -np.mean(scores) # negatif akurasi karena Firefly Algorithm berfungsi untuk meminimalkan nilai

lower_bounds = [1.0, 0.1] # Batas bawah C dan sigma
upper_bounds = [3.0, 1.0] # Batas atas C dan sigma
dimension = 2             # Jumlah parameter yang dioptimasi (C dan sigma)
max_evals = 50            # Jumlah evaluasi maksimal

# Inisialisasi FA
fa = FireflyAlgorithm(pop_size=30, alpha=0.2, betamin=1.0, gamma=1.0)

# Optimisasi FA untuk mencari  C dan sigma terbaik
best_params, best_accuracy = fa.optimize(svm_fitness, dimension, lower_bounds, upper_bounds, max_evals)

# Nilai C dan sigma terbaik
optimal_C, optimal_sigma = best_params
print(f"\nOptimal C: {optimal_C:.4f}, Optimal Sigma: {optimal_sigma:.4f}, Best Acc: {best_accuracy:.4f}")

final_model = SVC(C=optimal_C, kernel="rbf", gamma=optimal_sigma)
final_model.fit(X_train, y_train)

import pickle

with open("fa_svm_model.pkl", "wb") as model_file, open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(final_model, model_file)
    pickle.dump(scaler, scaler_file)

# Memuat model SVM dan scaler dari file pickle
def load_model_and_scaler(model_path="fa_svm_model.pkl", scaler_path="scaler.pkl"):
    with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return model, scaler

saved_final_model, saved_scaler = load_model_and_scaler()

y_pred = saved_final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"test accuracy: {final_accuracy:.4f}")

print(classification_report(y_test, y_pred, target_names=["Survived", "Died"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Survived", "Died"],
            yticklabels=["Survived", "Died"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()