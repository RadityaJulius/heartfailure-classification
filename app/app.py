import os
import json
import pandas as pd
import numpy as np
import streamlit as st  # Untuk membuat aplikasi web

# Import fungsi-fungsi dari utils.py
from utils import (
    load_model_and_scaler,  # Untuk memuat model dan scaler
    ensure_paths,  # Untuk memastikan direktori data dan model ada
    FEATURE_ORDER,  # Untuk urutan fitur
    load_dataset_cached,  # Untuk memuat dataset
    load_metrics_cached,  # Untuk memuat metrics
    predict_single,  # Untuk melakukan prediksi satu sampel
    predict_batch_dataframe,  # Untuk melakukan prediksi batch
)

# Konfigurasi halaman
st.set_page_config(
    page_title="Heart Failure Classification",
    page_icon="❤️",
    layout="wide",
)

# Sidebar: Navigation
st.sidebar.title("Heart Failure Classification")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Prediction", "📊 Data Insights", "🧪 Model Performance", "ℹ️ About"],
)

# Pastikan direktori data dan model ada
ensure_paths()

# Konfigurasi path untuk data dan model
MODELS_DIR = "models"
DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "heart_failure.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "heart_failure_svm.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# Render halaman prediksi
def render_prediction():
    st.header("Real-time Risk Prediction")
    st.write(
        "Berikan informasi pasien di bawah ini. Model akan menskalakan input dan memprediksi risiko kematian."
    )

    # Membuat form input yang sesuai dengan FEATURE_ORDER
    with st.form("prediction_form"):
        cols = st.columns(3)
        controls = {}

        # Mapping UI dan helper
        numeric_ranges = {
            "age": (18, 120, 1), # (lb, ub, step)
            "creatinine_phosphokinase": (10, 50000, 10),
            "ejection_fraction": (5, 80, 1),
            "platelets": (25000.0, 850000.0, 1000.0),
            "serum_creatinine": (0.1, 10.0, 0.1),
            "serum_sodium": (110, 160, 1),
            "time": (0, 350, 1),
        }

        # Mapping untuk field biner
        binary_fields = {
            "anaemia": "Anaemia (1=True, 0=False)",
            "diabetes": "Diabetes (1=True, 0=False)",
            "high_blood_pressure": "High blood pressure (1=True, 0=False)",
            "sex": "Sex (1=Male, 0=Female)",
            "smoking": "Smoking (1=True, 0=False)",
        }

        # Render controls dalam 3 kolom secara konsisten
        for idx, feat in enumerate(FEATURE_ORDER): # enumerate memberi pasangan (index, nilai).
            
            # idx % 3 akan berulang 0→1→2→0→1→2... sehingga setiap fitur ditempatkan bergantian ke kolom 1,2,3 secara merata.
            col = cols[idx % 3]

            if (
                feat in numeric_ranges
            ):  # Jika fitur numerik dengan batas & step terdefinisi
                lo, hi, step = numeric_ranges[feat] # Unpacking tuple (min, max, step) dari kamus numeric_ranges untuk fitur tsb.

                if isinstance(
                    step, float
                ):  # Kalau langkahnya float, buat number_input bertipe float
                    controls[feat] = col.number_input(
                        feat.replace(
                            "_", " "
                        ).title(),  # label: ubah snake_case ke Title Case
                        min_value=float(lo),  # batas bawah (float)
                        max_value=float(hi),  # batas atas (float)
                        value=float(
                            (lo + hi) / 2
                        ),  # nilai awal (titik tengah), bertipe float
                        step=float(step),  # ukuran lompatan tiap klik (float)
                    )
                else:
                    # Kalau langkahnya integer, buat number_input bertipe int
                    controls[feat] = col.number_input(
                        feat.replace("_", " ").title(),  # label
                        min_value=int(lo),  # batas bawah (int)
                        max_value=int(hi),  # batas atas (int)
                        value=int(
                            (lo + hi) // 2
                        ),  # nilai awal (tengah, dibulatkan ke bawah)
                        step=int(step),  # lompatan (int)
                    )

            elif feat in binary_fields:  # Jika fitur biner (0/1)
                controls[feat] = col.selectbox(
                    binary_fields[feat],  # label lebih ramah dari kamus binary_fields
                    options=[0, 1],  # pilihan 0 atau 1
                    index=0,  # default pilih 0
                    help="Select 1 for Yes/True, 0 for No/False",  # tooltip bantuan
                )

            else:
                # Fitur lain (fallback): number_input tanpa batas eksplisit, default float 0.0
                controls[feat] = col.number_input(
                    feat.replace("_", " ").title(), value=0.0
                )

        submitted = st.form_submit_button(
            "Predict"
        )  # Tombol submit form; True hanya pada saat user klik

    if submitted:
        # Membuat dataframe input yang sesuai dengan FEATURE_ORDER
        input_df = pd.DataFrame(
            [[controls[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER
        )

        # Memuat model dan scaler
        model, scaler, warnings = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

        if warnings:  # Jika ada warning
            for w in warnings:
                st.warning(w)

        if model is None or scaler is None:  # Jika model atau scaler tidak ditemukan
            st.error(
                "Model or scaler not found. Please train/export the model to proceed. "
                "Expected files: models/heart_failure_svm.joblib and models/scaler.joblib"
            )
            return

        # Melakukan prediksi
        pred_label, pred_prob = predict_single(model, scaler, input_df)

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns([1, 2])
        with col1:
            if pred_label == 1:  # Jika prediksi adalah 1
                st.error("Predicted: Meninggal")
            else:  # Jika prediksi adalah 0
                st.success("Predicted: Selamat")
            st.metric("Risk Probability (class=1)", f"{pred_prob * 100:.2f}%")

        with col2:  # Menampilkan ringkasan input
            st.caption("Input Summary")
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

        st.info(  # Menampilkan informasi
            "Catatan: Probabilitas mencerminkan keyakinan model untuk kelas=1 (peristiwa kematian). "
            "Tentukan bersama dengan penilaian klinis."
        )

    st.divider()  # Membuat pemisah
    st.subheader("Batch Predictions (Optional)")  # Menampilkan subheader
    uploaded = st.file_uploader(
        "Upload CSV with the same feature columns (no DEATH_EVENT column).",
        type=["csv"],
    )  # Menampilkan tombol upload
    if uploaded is not None:  # Jika ada file yang diupload
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())  # Menampilkan preview
            model, scaler, warnings = load_model_and_scaler(
                MODEL_PATH, SCALER_PATH
            )  # Memuat model dan scaler
            if warnings:  # Jika ada warning
                for w in warnings:
                    st.warning(w)
            if (
                model is None or scaler is None
            ):  # Jika model atau scaler tidak ditemukan
                st.error("Model or scaler not found. Train/export first.")
                return
            out = predict_batch_dataframe(model, scaler, df)  # Melakukan prediksi batch
            st.write("Predictions Preview:", out.head())
            st.download_button(  # Menampilkan tombol download
                "Download Results CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:  # Jika ada error
            st.exception(e)  # Menampilkan error


# Render halaman data insights
def render_data_insights():
    st.header("Data Insights")  # Menampilkan header
    st.write(
        "Exploratory analysis of the heart failure dataset."
    )  # Menampilkan deskripsi

    if not os.path.exists(DATASET_PATH):  # Jika dataset tidak ditemukan
        st.error(
            f"Dataset not found at {DATASET_PATH}. Please place a CSV file there."
        )  # Menampilkan error
        return

    df = load_dataset_cached(DATASET_PATH)  # Memuat dataset
    st.subheader("Overview")  # Menampilkan subheader
    c1, c2, c3 = st.columns(3)  # Membuat 3 kolom
    c1.metric("Rows", df.shape[0])  # Menampilkan jumlah baris
    c2.metric("Columns", df.shape[1])  # Menampilkan jumlah kolom
    c3.metric(
        "Missing values", int(df.isna().sum().sum())
    )  # Menampilkan jumlah missing values

    st.write("First 10 rows")  # Menampilkan 10 baris pertama
    st.dataframe(df.head(10))  # Menampilkan 10 baris pertama

    # Class balance if target present
    if "DEATH_EVENT" in df.columns:  # Jika kolom DEATH_EVENT ada
        st.subheader("Class Balance")  # Menampilkan subheader
        counts = (
            df["DEATH_EVENT"].value_counts().sort_index()
        )  # Menghitung jumlah kelas
        st.bar_chart(
            counts, x=None, y=None, use_container_width=True
        )  # Menampilkan bar chart

    # Column selector for distributions
    st.subheader("Distribution Plots")  # Menampilkan subheader
    num_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]  # Menampilkan kolom numerik
    if num_cols:  # Jika ada kolom numerik
        sel_col = st.selectbox(
            "Select a numeric column", options=num_cols
        )  # Memilih kolom numerik
        if sel_col:  # Jika ada kolom numerik yang dipilih
            st.line_chart(
                df[sel_col], use_container_width=True
            )  # Menampilkan line chart
            st.caption(
                "Line chart of sorted values (quick look). For exact distributions, consider histogram via Altair/Plotly add-ons if needed."
            )  # Menampilkan caption
    else:
        st.info("No numeric columns detected.")  # Menampilkan informasi

    # Correlation heatmap
    st.subheader("Correlation Heatmap")  # Menampilkan subheader
    try:
        import seaborn as sns  # Untuk membuat heatmap
        import matplotlib.pyplot as plt  # Untuk membuat plot

        corr = (
            df[num_cols].corr() if num_cols else pd.DataFrame()
        )  # Menghitung korelasi
        if corr.empty:  # Jika tidak ada korelasi
            st.info(
                "Not enough numeric columns for correlation heatmap."
            )  # Menampilkan informasi
        else:  # Jika ada korelasi
            fig, ax = plt.subplots(figsize=(10, 6))  # Membuat plot
            sns.heatmap(
                corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
            )  # Membuat heatmap
            st.pyplot(fig)  # Menampilkan plot
    except Exception as e:  # Jika ada error
        st.exception(e)  # Menampilkan error


# Render halaman model performance
def render_model_performance():
    st.header("Model Performance")  # Menampilkan header
    st.write(
        "View evaluation metrics and confusion matrix if available."
    )  # Menampilkan deskripsi

    if os.path.exists(METRICS_PATH):  # Jika file metrics ada
        metrics = load_metrics_cached(METRICS_PATH)  # Memuat metrics
        st.subheader("Metrics Summary")  # Menampilkan subheader
        st.json(metrics)  # Menampilkan metrics
    else:
        st.info(
            "metrics.json not found. You can generate it during training."
        )  # Menampilkan informasi

    # Menghitung evaluasi cepat jika dataset dan model ada
    if st.checkbox(
        "Hitung evaluasi cepat pada pembagian acak (indikatif)"
    ):  # Jika checkbox dicentang
        if not os.path.exists(DATASET_PATH):  # Jika dataset tidak ditemukan
            st.error(
                "Dataset not found. Provide dataset to proceed."
            )  # Menampilkan error
            return
        model, scaler, warnings = load_model_and_scaler(
            MODEL_PATH, SCALER_PATH
        )  # Memuat model dan scaler
        for w in warnings:  # Jika ada warning
            st.warning(w)  # Menampilkan warning
        if model is None or scaler is None:  # Jika model atau scaler tidak ditemukan
            st.error(
                "Model or scaler missing. Train/export first."
            )  # Menampilkan error
            return

        from sklearn.model_selection import train_test_split  # Untuk membagi data
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix, 
        )  # Untuk menghitung akurasi, laporan klasifikasi, dan matriks kebingungan

        df = load_dataset_cached(DATASET_PATH)  # Memuat dataset
        if "DEATH_EVENT" not in df.columns:  # Jika kolom DEATH_EVENT tidak ada
            st.error(
                "Dataset missing DEATH_EVENT column for evaluation."
            ) 
            return

        X = df.drop(columns=["DEATH_EVENT"])  # Menghapus kolom DEATH_EVENT
        y = df["DEATH_EVENT"].astype(
            int
        )  # Mengubah tipe data kolom DEATH_EVENT menjadi integer

        # Ensure columns align to FEATURE_ORDER
        missing = [
            c for c in FEATURE_ORDER if c not in X.columns
        ]  # Menghitung kolom yang hilang
        if missing:  # Jika ada kolom yang hilang
            st.error(
                f"Dataset missing required features: {missing}"
            )  # Menampilkan error
            return
        X = X[FEATURE_ORDER]  # Mengurutkan kolom sesuai FEATURE_ORDER

        X_scaled = scaler.transform(X.values)  # Mengubah skala fitur
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )  # Membagi data menjadi data latihan dan data uji

        y_pred = model.predict(X_test)  # Melakukan prediksi
        acc = float(accuracy_score(y_test, y_pred))  # Menghitung akurasi
        st.metric("Accuracy (random split)", f"{acc:.4f}")  # Menampilkan akurasi

        st.text("Classification Report")  # Menampilkan laporan klasifikasi
        st.code(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Survived", "Died"],
            )
        )  # Menampilkan laporan klasifikasi

        from seaborn import heatmap  # Untuk membuat heatmap
        import matplotlib.pyplot as plt  # Untuk membuat plot

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Menghitung confusion matrix
        fig, ax = plt.subplots()  # Membuat plot
        heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Survived", "Died"],
            yticklabels=["Survived", "Died"],
            ax=ax,
        )  # Membuat heatmap
        ax.set_xlabel("Predicted") 
        ax.set_ylabel("Actual")
        st.pyplot(fig)  # Menampilkan plot

def render_about():
    st.header("About")
    st.markdown(
        """
Aplikasi ini memprediksi luaran gagal jantung menggunakan SVM dengan kernel RBF. Pipeline dari pelatihan meliputi:
- Penanganan outlier melalui winsorisasi
- Penskalaan fitur menggunakan StandardScaler
- Penyeimbangan kelas melalui SMOTE
- Optimasi hiperparameter menggunakan Algoritma Firefly
Model dan skaler berada di:
- models/heart_failure_svm.joblib
- models/scaler.joblib
Dataset insights:
- data/heart_failure.csv
        """
    )
    st.caption(
        "Catatan: Prediksi hanya untuk tujuan informasi dan tidak boleh menggantikan penilaian klinis."
    )


if page == "🏠 Prediction":
    render_prediction()
elif page == "📊 Data Insights":
    render_data_insights()
elif page == "🧪 Model Performance":
    render_model_performance()
else:
    render_about()
