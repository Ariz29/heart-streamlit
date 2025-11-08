import streamlit as st
import joblib
import numpy as np

# =============================
# 1. Load model & preprocessing
# =============================
model = joblib.load("model/random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

# =============================
# 2. Judul aplikasi
# =============================
st.title("💓 Prediksi Penyakit Jantung")
st.write("Masukkan data pasien di bawah ini untuk memprediksi kemungkinan penyakit jantung.")

# =============================
# 3. Input kolom pengguna
# =============================
col1, col2 = st.columns(2)

with col1:
    age = st.text_input("Usia (Age)")
with col2:
    sex = st.selectbox("Jenis Kelamin", ["male", "female"])

with col1:
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
with col2:
    trestbps = st.text_input("Tekanan Darah Saat Istirahat (trestbps)")

with col1:
    chol = st.text_input("Kadar Kolesterol (chol)")
with col2:
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", [0, 1])

with col1:
    restecg = st.selectbox("Hasil Elektrokardiografi (restecg)", [0, 1, 2])
with col2:
    thalach = st.text_input("Detak Jantung Maksimum (thalach)")

with col1:
    exang = st.selectbox("Nyeri Dada Karena Olahraga (exang)", [0, 1])
with col2:
    oldpeak = st.text_input("Depresi ST (oldpeak)")

with col1:
    slope = st.selectbox("Kemiringan ST (slope)", [0, 1, 2])
with col2:
    ca = st.selectbox("Jumlah Pembuluh Darah (ca)", [0, 1, 2, 3, 4])

with col1:
    thal = st.selectbox("Thal", ["normal", "fixed defect", "reversible defect"])

# =============================
# 4. Prediksi
# =============================
hasil_prediksi = ""

if st.button("🔍 Tes Prediksi Penyakit Jantung"):
    try:
        # Ubah input numerik ke float
        numeric_data = np.array([[float(age), float(trestbps), float(chol),
                                  float(thalach), float(oldpeak)]])
        
        # Data kategorikal
        categorical_data = np.array([[sex, int(cp), int(fbs),
                                      int(restecg), int(exang),
                                      int(slope), int(ca), thal]])
        
        # Scaling dan encoding
        scaled_numeric = scaler.transform(numeric_data)
        encoded_categorical = encoder.transform(categorical_data)
        
        # Gabungkan semua fitur
        final_input = np.concatenate((scaled_numeric, encoded_categorical), axis=1)
        
        # Prediksi
        prediction = model.predict(final_input)
        
        if prediction[0] == 1:
            hasil_prediksi = "⚠️ Pasien berisiko mengidap penyakit jantung."
        else:
            hasil_prediksi = "✅ Pasien tidak berisiko penyakit jantung."
        
        st.success(hasil_prediksi)
    
    except ValueError as e:
        st.error(f"Kesalahan input: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
