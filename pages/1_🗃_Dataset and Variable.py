import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="HOME",
    page_icon="🗃",
    layout="wide"
)

data = pd.read_csv("./data/Student_performance_data_.csv")

with st.sidebar:
    selected = option_menu(
        menu_title="DATASET",
        options=["DATASET", "PREPROCESSING"],
    )

if selected == "DATASET":
    st.title("DATASET")

    st.header("📋 Preview Data")
    st.dataframe(data.head())
    st.write(data.shape[0], "rows and", data.shape[1], "columns.")

    mem_usage = data.memory_usage(deep=True).sum()
    mem_in_kb = mem_usage / 1024
    mem_in_mb = mem_in_kb / 1024
    st.write(f"Total memory used: {mem_in_kb:.2f} KB ({mem_in_mb:.2f} MB)")

    st.markdown("---")
    st.dataframe(data.dtypes.to_frame(name='Data Type'))

    st.markdown("---")
    st.header("📋 Dataset Describe")
    st.dataframe(data.describe())

    st.markdown("---")
    st.title("👁‍🗨 DETAIL DATASET")

    st.header("🔑 Student ID")
    st.markdown("- **StudentID**: ID unik untuk setiap mahasiswa, mulai dari `1001` hingga `3392`.")

    st.header("👤 Demografi")
    st.markdown("""
    - **Age**: Usia mahasiswa, berkisar antara `15` hingga `18` tahun.  
    - **Gender**: Jenis kelamin:
        - `0`: Laki-laki  
        - `1`: Perempuan  
    - **Ethnicity**: Etnis mahasiswa:
        - `0`: Caucasian  
        - `1`: African American  
        - `2`: Asian  
        - `3`: Other  
    - **ParentalEducation**: Tingkat pendidikan orang tua:
        - `0`: None  
        - `1`: High School  
        - `2`: Some College  
        - `3`: Bachelor's  
        - `4`: Higher  
    """)

    st.header("📖 Kebiasaan Belajar")
    st.markdown("""
    - **StudyTimeWeekly**: Waktu belajar per minggu (jam), mulai dari `0` hingga `20`.  
    - **Absences**: Jumlah ketidakhadiran dalam setahun, dari `0` hingga `30`.  
    - **Tutoring**: Status les tambahan:
        - `0`: Tidak  
        - `1`: Ya  
    """)

    st.header("👨‍👩‍👧‍👦 Keterlibatan Orang Tua")
    st.markdown("""
    - **ParentalSupport**: Tingkat dukungan dari orang tua:
        - `0`: None  
        - `1`: Low  
        - `2`: Moderate  
        - `3`: High  
        - `4`: Very High  
    """)

    st.header("🎯 Kegiatan Ekstrakurikuler")
    st.markdown("""
    - **Extracurricular**: Partisipasi dalam kegiatan ekstrakurikuler:
        - `0`: Tidak  
        - `1`: Ya  
    - **Sports**: Ikut olahraga:  
        - `0`: Tidak  
        - `1`: Ya  
    - **Music**: Ikut kegiatan musik:
        - `0`: Tidak  
        - `1`: Ya  
    - **Volunteering**: Ikut kegiatan sosial/sukarela:
        - `0`: Tidak  
        - `1`: Ya  
    """)

    st.header("📈 Performa Akademik")
    st.markdown("""
    - **GPA**: Rata-rata nilai akhir (Grade Point Average), skala `2.0` hingga `4.0`. Dipengaruhi oleh kebiasaan belajar, dukungan orang tua, dan kegiatan ekstrakurikuler.

    - **GradeClass**: Klasifikasi nilai berdasarkan GPA:
        - `0`: A (GPA ≥ 3.5)  
        - `1`: B (3.0 ≤ GPA < 3.5)  
        - `2`: C (2.5 ≤ GPA < 3.0)  
        - `3`: D (2.0 ≤ GPA < 2.5)  
        - `4`: F (GPA < 2.0)  
    """)


if selected == "PREPROCESSING":
    st.write("# ⚙ PREPROCESSING DATASET")

    st.header("Analisis Statistik Deskriptif")
    st.markdown("---")

    st.header("Penanganan Missing Value")
    st.dataframe(data.isnull().sum().to_frame(name='Missing Values'))
    st.write("Jumlah missing value pada dataset: ", data.isnull().sum().sum())
    st.write("Hasil pengecekan missing value menunjukkan bahwa tidak ada nilai yang hilang di seluruh kolom, sehingga data dalam kondisi lengkap.")
    st.markdown("---")

    st.header("Normalisasi atau diskretisasi variabel")
    st.markdown("---")

    st.header("Analisis korelasi")
    st.markdown("---")

    st.header("Visualisasi distribusi data")