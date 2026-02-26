
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. Muat Model dan Scaler yang Tersimpan ---

# Muat model Gradient Boosting dari file pickle
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Muat scaler dari file pickle
with open('feature_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# --- 2. Definisikan Mappings untuk Preprocessing (Berdasarkan Tahap Training) ---

pendidikan_mapping = {
    'D3': 0,
    'S1': 1,
    'SMA': 2,
    'SMK': 3
}
jurusan_mapping = {
    'ADMINISTRASI': 0,
    'DESAIN GRAFIS': 1,
    'OTOMOTIF': 2,
    'TEKNIK LAS': 3,
    'TEKNIK LISTRIK': 4
}

# Urutan kolom final setelah preprocessing dan scaling (kecuali kolom target Gaji_Pertama_Juta)
final_feature_columns = [
    'Pendidikan', 'Jurusan', 'Jenis_Kelamin_L', 'Jenis_Kelamin_P',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja',
    'Usia', 'Durasi_Jam', 'Nilai_Ujian'
]

# --- 3. Fungsi untuk Preprocessing Data Baru ---
def preprocess_new_data(new_data):
    df_new = pd.DataFrame([new_data])

    df_new['Pendidikan'] = df_new['Pendidikan'].map(pendidikan_mapping)
    df_new['Jurusan'] = df_new['Jurusan'].map(jurusan_mapping)

    df_onehot_new = pd.get_dummies(df_new[['Jenis_Kelamin', 'Status_Bekerja']])
    df_onehot_new = df_onehot_new.astype(int)

    df_processed = pd.concat([
        df_new[['Pendidikan', 'Jurusan', 'Usia', 'Durasi_Jam', 'Nilai_Ujian']],
        df_onehot_new
    ], axis=1)

    # Reindex untuk memastikan urutan dan keberadaan kolom sesuai dengan training
    df_processed = df_processed.reindex(columns=final_feature_columns, fill_value=0)

    # Aplikasikan Standard Scaling
    scaled_features = loaded_scaler.transform(df_processed)
    df_scaled = pd.DataFrame(scaled_features, columns=final_feature_columns)

    return df_scaled

# --- 4. Fungsi untuk Melakukan Prediksi ---
def predict_salary(data_input):
    processed_data = preprocess_new_data(data_input)
    prediction = loaded_model.predict(processed_data)
    return prediction[0]

# --- 5. Streamlit App ---
st.set_page_config(page_title="Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")

st.write("Aplikasi ini memprediksi gaji awal (dalam juta Rupiah) berdasarkan data calon peserta.")

# Input dari Pengguna
st.sidebar.header("Input Data Calon Peserta")

usia = st.sidebar.slider("Usia", 18, 60, 25)
durasi_jam = st.sidebar.slider("Durasi Pelatihan (Jam)", 20, 100, 60)
nilai_ujian = st.sidebar.slider("Nilai Ujian", 50.0, 100.0, 75.0)
pendidikan = st.sidebar.selectbox("Pendidikan", list(pendidikan_mapping.keys()))
jurusan = st.sidebar.selectbox("Jurusan", list(jurusan_mapping.keys()))
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ['L', 'P'])
status_bekerja = st.sidebar.selectbox("Status Bekerja", ['Sudah Bekerja', 'Belum Bekerja'])

# Tombol Prediksi
if st.sidebar.button("Prediksi Gaji Awal"):
    input_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    predicted_salary = predict_salary(input_data)

    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi Gaji Awal: **Rp {predicted_salary:.2f} Juta**")
    st.write("--- Data Input Anda ---")
    st.json(input_data)
