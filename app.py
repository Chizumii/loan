import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIG HALAMAN ---
st.set_page_config(page_title="Credit Scoring System", page_icon="🏦", layout="centered")

# --- 2. LOAD SEMUA ARTIFACTS ---
@st.cache_resource
def load_all_artifacts():
    try:
        # Load kedua model + scaler + feature names
        xgb = joblib.load("xgb_model.pkl")
        lr = joblib.load("logreg_model.pkl")
        scaler = joblib.load("scaler.pkl")
        # Jika feature_list.pkl ada, kita load, jika tidak kita define manual
        try:
            features = joblib.load("feature_list.pkl")
        except:
            features = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti']
            
        return xgb, lr, scaler, features
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan ({e}). Harap jalankan model.ipynb dulu!")
        return None, None, None, None

xgb_model, lr_model, scaler, feature_names = load_all_artifacts()

# --- 3. FUNGSI PREPROCESSING (SUDAH DIPERBAIKI) ---
def preprocess_user_input(annual_inc_idr, loan_amnt_idr, term_input, int_rate, dti):
    # 1. Konversi Rupiah ke USD (Asumsi kurs 16.000)
    kurs = 16000
    annual_inc_usd = annual_inc_idr / kurs
    loan_amnt_usd = loan_amnt_idr / kurs
    
    # 2. Konversi Term ke Angka
    term_val = 36 if term_input == "36 Bulan" else 60
    
    # 3. PERBAIKAN PENTING: Log Transform pada Income
    # Model dilatih menggunakan np.log1p, maka input juga harus di-log
    annual_inc_log = np.log1p(annual_inc_usd)
    
    # 4. Clipping DTI (agar konsisten dengan training)
    dti_final = min(dti, 100.0) 
    
    # 5. Susun DataFrame (Urutan kolom harus sama persis dengan saat training)
    data = {
        'loan_amnt': [loan_amnt_usd],
        'term': [term_val],
        'int_rate': [int_rate],
        'annual_inc': [annual_inc_log], # Gunakan nilai yang sudah di-Log
        'dti': [dti_final]
    }
    return pd.DataFrame(data)

# --- 4. SIDEBAR (PILIH MODEL) ---
st.sidebar.title("⚙️ Pengaturan")
model_choice = st.sidebar.radio(
    "Pilih Model Prediksi:",
    ("XGBoost (Recommended)", "Logistic Regression")
)

st.sidebar.info(
    """
    **Info Model:**
    - **XGBoost**: Lebih akurat untuk pola kompleks.
    - **Logistic Regression**: Lebih sederhana dan linear.
    """
)

# --- 5. UI UTAMA ---
st.title("🏦 Sistem Analisis Kredit")
st.write(f"Model aktif: **{model_choice}**")

with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Default value disesuaikan agar terlihat 'wajar'
        annual_inc = st.number_input("Pendapatan Tahunan (Rp)", min_value=10_000_000, value=120_000_000, step=5_000_000, help="Total pendapatan kotor per tahun")
        loan_amount = st.number_input("Jumlah Pinjaman (Rp)", min_value=1_000_000, value=10_000_000, step=1_000_000)
        term = st.selectbox("Jangka Waktu", ["36 Bulan", "60 Bulan"])
    
    with col2:
        int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        dti = st.number_input("Rasio Hutang (DTI %)", min_value=0.0, max_value=100.0, value=15.0, help="Persentase cicilan bulanan dibanding pendapatan bulanan")

    submit_btn = st.form_submit_button("🔍 Cek Kelayakan")

# --- 6. LOGIKA PREDIKSI (SUDAH DIPERBAIKI) ---
if submit_btn:
    if xgb_model is None or lr_model is None:
        st.error("Model belum dimuat. Pastikan file .pkl ada di folder yang sama.")
    else:
        # 1. Siapkan Data (sudah termasuk Log Transform)
        X_input = preprocess_user_input(annual_inc, loan_amount, term, int_rate, dti)
        
        # 2. Scaling (Wajib, karena training pakai StandardScaler)
        try:
            X_scaled = scaler.transform(X_input)
            
            # 3. Pilih Model
            if "XGBoost" in model_choice:
                selected_model = xgb_model
            else:
                selected_model = lr_model
                
            # 4. Prediksi
            # PERBAIKAN LOGIKA:
            # Saat training: target = (loan_status == 'Fully Paid') -> 1
            # Jadi: Class 1 = Lunas (Good), Class 0 = Gagal (Bad)
            
            prob_good = selected_model.predict_proba(X_scaled)[0][1] # Peluang Lunas
            prob_default = 1 - prob_good                             # Peluang Gagal Bayar
            
            # Skor Kredit (0 - 100)
            # Semakin tinggi prob_good, semakin tinggi skornya
            credit_score = int(prob_good * 100)
            
            # 5. Tampilkan Hasil
            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                # Warna dinamis untuk skor
                delta_color = "normal"
                if credit_score >= 80: delta_color = "normal" # Hijau (default streamlit metric naik)
                elif credit_score < 60: delta_color = "inverse" # Merah
                
                st.metric("Skor Kredit", f"{credit_score}/100", help="Skor 100 berarti sangat layak")
            
            with c2:
                st.subheader("Hasil Analisis")
                
                # Interpretasi Skor
                if credit_score >= 80:
                    st.success(f"**Sangat Layak (Low Risk)**\n\nNasabah memiliki probabilitas pelunasan tinggi ({prob_good*100:.1f}%). Risiko gagal bayar sangat kecil.")
                elif credit_score >= 60:
                    st.warning(f"**Dipertimbangkan (Medium Risk)**\n\nNasabah cukup layak namun perlu peninjauan manual. Peluang lunas sekitar {prob_good*100:.1f}%.")
                else:
                    st.error(f"**Ditolak (High Risk)**\n\nRisiko gagal bayar terlalu tinggi ({prob_default*100:.1f}%). Disarankan untuk menolak pengajuan.")

            # Debug info (Bisa disembunyikan jika sudah production)
            with st.expander("Lihat Data Input (Processed)"):
                st.write("Data yang masuk ke model (USD & Log Scale):")
                st.dataframe(X_input)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.warning("Pastikan jumlah fitur di 'scaler.pkl' cocok dengan input aplikasi.")