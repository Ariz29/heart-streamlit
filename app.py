import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =============================
# 1. Path dan Load Model
# =============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"

# Cek file
if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not ENCODER_PATH.exists():
    st.error("❌ File model, scaler, atau encoder tidak ditemukan di folder 'model/'.")
    st.stop()

# Load semua file
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# =============================
# 2. Judul Aplikasi
# =============================
st.title("💓 Prediksi Penyakit Jantung")
st.write("Masukkan data pasien di bawah ini untuk memprediksi kemungkinan penyakit jantung menggunakan model Random Forest.")

# =============================
# 3. Input Data Pengguna
# =============================
age = st.number_input("Usia (tahun)", 20, 100, 40)
sex = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
cp = st.selectbox("Tipe Nyeri Dada", ("NAP", "ATA", "TA", "ASY"))
restingbp = st.number_input("Tekanan Darah Istirahat (mm Hg)", 80, 200, 120)
chol = st.number_input("Kolesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
restecg = st.selectbox("Hasil ECG", ("Normal", "ST", "LVH"))
maxhr = st.number_input("Denyut Jantung Maksimum", 60, 202, 150)
exang = st.selectbox("Exercise Angina", ("N", "Y"))
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)
slope = st.selectbox("ST Slope", ("Up", "Flat", "Down"))

# =============================
# 4. Mapping Kategori (sesuai preprocessing)
# =============================
sex_map = {"Perempuan": 0, "Laki-laki": 1}
cp_map = {"NAP": 0, "ATA": 1, "TA": 2, "ASY": 3}
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}
angina_map = {"N": 0, "Y": 1}

# =============================
# 5. Dataframe Input
# =============================
data_input = pd.DataFrame([{
    "Age": age,
    "Sex": sex_map[sex],
    "ChestPainType": cp_map[cp],
    "RestingBP": restingbp,
    "Cholesterol": chol,
    "FastingBS": fbs,
    "RestingECG": ecg_map[restecg],
    "MaxHR": maxhr,
    "ExerciseAngina": angina_map[exang],
    "Oldpeak": oldpeak,
    "ST_Slope": slope_map[slope]
}])

# =============================
# 6. Scaling
# =============================
data_scaled = scaler.transform(data_input)

# =============================
# 7. Prediksi
# =============================
if st.button("🔍 Prediksi Sekarang"):
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Pasien kemungkinan memiliki penyakit jantung. Probabilitas: **{prob:.2f}**")
    else:
        st.success(f"✅ Pasien kemungkinan *tidak* memiliki penyakit jantung. Probabilitas: **{prob:.2f}**")

# =============================
# 8. Catatan Tambahan
# =============================
st.caption("Model ini menggunakan Random Forest Classifier yang telah dilatih pada data penyakit jantung.")
