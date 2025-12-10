import streamlit as st
import pandas as pd

# 1. Konfigurasi Halaman (Judul Tab & Icon)
st.set_page_config(page_title="Proyek ALP Darren", page_icon="📊")

# 2. Judul Aplikasi
st.title("📊 Dashboard Analisis Data")
st.write("Upload file dataset Anda (CSV) untuk melihat isinya secara otomatis.")

# 3. Widget untuk Upload File
uploaded_file = st.file_uploader("Pilih file CSV...", type=["csv"])

# 4. Logika: Jika file sudah diupload, maka baca dan tampilkan
if uploaded_file is not None:
    # Membaca file CSV menjadi DataFrame Pandas
    df = pd.read_csv(uploaded_file)
    
    st.success("File berhasil diupload!")
    
    # Menampilkan Statistik Data
    st.subheader("Preview Data")
    st.dataframe(df) # Menampilkan tabel interaktif
    
    st.subheader("Statistik Deskriptif")
    st.write(df.describe()) # Menampilkan mean, min, max, dll.
else:
    st.info("Silakan upload file CSV untuk memulai.")