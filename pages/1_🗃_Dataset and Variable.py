import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dataset and Variable",
    page_icon="🗃",
    layout="wide"
)

data = pd.read_csv("./data/Student_performance_data_.csv")
data_normalization = pd.read_csv("./data/data_normalization.csv")

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
    st.subheader("Tipe Data")
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
    with st.sidebar:
        sub_selected = option_menu(
            menu_title="",
            options=["Analisis Statistik Deskriptif", "Penanganan Missing Value", "Normalisasi atau diskretisasi variabel", "Analisis Korelasi", "Visualisasi distribusi data untuk memahami pola"],
        )
        
    if sub_selected == "Analisis Statistik Deskriptif":
        st.header("Analisis Statistik Deskriptif")
        st.header("📋 Dataset Describe")
        st.dataframe(data.describe())

        st.write("### Gender Distribution")
        st.dataframe(data["Gender"].value_counts().to_frame())
        st.write("### Ethnicity Distribution")
        st.dataframe(data["Ethnicity"].value_counts().to_frame())
        st.write("### Parental Education Distribution")
        st.dataframe(data["ParentalEducation"].value_counts().to_frame())
        st.write("### Parental Support Distribution")
        st.dataframe(data["ParentalSupport"].value_counts().to_frame())
        st.write("### Extracurricular Activities Distribution")
        st.dataframe(data["Extracurricular"].value_counts().to_frame())
        st.write("### Sports Participation Distribution")
        st.dataframe(data["Sports"].value_counts().to_frame())
        st.write("### Absences Distribution")
        st.dataframe(data["Absences"].value_counts().to_frame())
        st.write("### Music Participation Distribution")
        st.dataframe(data["Music"].value_counts().to_frame())
        st.write("### Volunteering Distribution")
        st.dataframe(data["Volunteering"].value_counts().to_frame())
        st.write("### Age Distribution")
        st.dataframe(data["Age"].value_counts().to_frame())
        st.write("### Grade Class Distribution")
        st.dataframe(data["GradeClass"].value_counts().to_frame())

    if sub_selected == "Penanganan Missing Value":
        st.write("### Penanganan Missing Value")     
        st.dataframe(data.isnull().sum().to_frame(name='Missing Values'))
        st.write("Jumlah missing value pada dataset: ", data.isnull().sum().sum())
        st.write("Hasil pengecekan missing value menunjukkan bahwa tidak ada nilai yang hilang di seluruh kolom, sehingga data dalam kondisi lengkap.")
        
    if sub_selected == "Normalisasi atau diskretisasi variabel": 
        # copy data_mapped
        st.write("## Normalisasi atau diskretisasi variabel")

        st.write("")
        st.write("#### Kolom GPA dan StudyTimeWeekly")
        st.dataframe(data[["GPA", "StudyTimeWeekly"]].head())


        st.markdown("---")
        st.write('#### Diskretisasi Kolom GPA dan StudyTimeWeekly')
        code_discrit_1 = '''
        gpa_bins = [0, 2.0, 2.5, 3.0, 3.5, float('inf')]
        gpa_labels = [4, 3, 2, 1, 0]

        study_bins = [0, 5, 10, 15, float('inf')]
        study_labels = [0, 1, 2, 3]

        # Terapkan pd.cut untuk diskretisasi
        data_normalization['GPA_Disc'] = pd.cut(data_normalization['GPA'], bins=gpa_bins, labels=gpa_labels, right=False)
        data_normalization['StudyTimeWeekly_Disc'] = pd.cut(data_normalization['StudyTimeWeekly'], bins=study_bins, labels=study_labels, right=False)
        '''
        st.code(code_discrit_1, language='python')

        # Tampilkan hasil diskretisasi
        st.write("##### RESULT")
        st.table(data_normalization[['GPA_Disc', 'StudyTimeWeekly_Disc']].head())

        st.write("\n")
        st.markdown("---")
        st.write("\n")

        st.write('#### Diskretisasi Semua Kolom Selain StudyTimeWeekly_Disc dan GPA_Disc')
        code_discrit_2 = '''
        # Drop kolom asli yang sudah didiskritkan
        data_normalization = data_normalization.drop(["GPA", "StudyTimeWeekly"], axis=1)

        # Pisahkan fitur numerik yang perlu didiskritkan
        numerical_features_to_discretize = [col for col in data_normalization.columns if col not in ['GPA_Disc', 'StudyTimeWeekly_Disc']]

        # Diskretisasi sisa fitur yang belum diskrit (gunakan KBinsDiscretizer untuk fitur lainnya)
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        data_normalization[numerical_features_to_discretize] = discretizer.fit_transform(data_normalization[numerical_features_to_discretize])

        # Ubah ke int
        data_normalization[numerical_features_to_discretize] = data_normalization[numerical_features_to_discretize].astype(int)
        '''
        st.code(code_discrit_2, language='python')
        st.write("##### RESULT")
        st.dataframe(data_normalization.head(5))


    if sub_selected == "Analisis Korelasi": 
        # Salin dataframe
        data_encoded = data_normalization.copy()

        # Ubah kolom kategorikal menjadi numerik
        for col in data_encoded.select_dtypes(include='object').columns:
            data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

        # Hitung korelasi untuk semua kolom (tanpa StudentID) dengan metode Pearson
        correlation_matrix_all = data_encoded.drop(columns=['StudentID']).corr(method='pearson')

        st.write("### Analisis Korelasi All Table")
        # Visualisasi korelasi dengan heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Matriks Korelasi Seluruh Variabel (Termasuk Kategorikal)")
        st.pyplot(plt)

        st.write("\n")
        st.markdown("---")
        st.write("\n")

        st.write("### Pengaruh Absensi terhadap GPA Diskrit dan GradeClass")

        gpa_or_grade_col = 'GPA_Disc'  # Atau 'GradeClass'

        fig_abs_gpa_single = px.scatter(data_normalization, x='Absences', y=gpa_or_grade_col,
                                        title=f"Hubungan antara Absensi dan {gpa_or_grade_col}",
                                        labels={'Absences': 'Disc Absensi', gpa_or_grade_col: gpa_or_grade_col})
        st.plotly_chart(fig_abs_gpa_single)


        st.write("\n")
        st.markdown("---")
        st.write("\n")

        st.write("#### Distribusi GPA/GradeClass berdasarkan Kategori Absensi")

        # Membuat kategori absensi (jika belum dibuat)
        bins = [0, 2, 5, float('inf')]
        labels_abs = ['0-2', '3-5', '>5']
        data_normalization['Absences_Group'] = pd.cut(data_normalization['Absences'], bins=bins, labels=labels_abs, right=False)

        # Pilih salah satu yang relevan: 'GPA_Disc' atau 'GradeClass'
        gpa_or_grade_col = 'GPA_Disc'  # Atau 'GradeClass'

        fig_gpa_abs_box_single = px.box(data_normalization, x='Absences_Group', y=gpa_or_grade_col,
                                        title=f"Distribusi {gpa_or_grade_col} berdasarkan Kelompok Absensi",
                                        labels={'Absences_Group': 'Kelompok Absensi', gpa_or_grade_col: gpa_or_grade_col})
        st.plotly_chart(fig_gpa_abs_box_single)

        # Hapus kolom kategori setelah digunakan (opsional)
        if 'Absences_Group' in data_normalization.columns:
            data_normalization.drop('Absences_Group', axis=1, inplace=True)


    if sub_selected == "Visualisasi distribusi data untuk memahami pola": 
        # Pisahkan fitur numerik kontinu dan kategorikal-biner
        binary_or_categorical = ['Gender', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'Tutoring']
        diskrit_categorical = ['Age', 'StudyTimeWeekly_Disc', 'Absences', 'Ethnicity', 'ParentalEducation', 
                               'ParentalSupport',  'GPA_Disc', 'GradeClass']
        

        for feature_disc in diskrit_categorical:
            # Konversi nilai 0/1 menjadi "Ya"/"Tidak" untuk visualisasi
            plot_data = data_normalization[feature_disc]

            plt.figure(figsize=(8, 5))
            ax = sns.countplot(x=plot_data)
            plt.title(f'Distribusi {feature_disc}')
            plt.xlabel(feature_disc)
            plt.ylabel('Count')

            # Anotasi count di atas bar
            for p in ax.patches:
                height = p.get_height()
                x = p.get_x() + p.get_width() / 2
                y = height + 0.1
                plt.text(x, y, f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            st.pyplot(plt)

            # Hitung jumlah masing-masing nilai (pakai data asli)
            value_counts = data_normalization[feature_disc].value_counts()
            most_common_value = value_counts.idxmax()
            most_common_count = value_counts.max()
            least_common_value = value_counts.idxmin()
            least_common_count = value_counts.min()

            st.write(f"**Nilai terbanyak di kolom `{feature_disc}`**: {most_common_value} (jumlah: {most_common_count})")
            st.write(f"**Nilai tersedikit di kolom `{feature_disc}`**: {least_common_value} (jumlah: {least_common_count})")
            st.markdown("---")
            st.write("")


        # Visualisasi fitur kategorikal atau biner
        def convert_binary_series(series):
            return series.map({4: 'Ya', 0: 'Tidak'}).fillna(series)

        for feature_bin in binary_or_categorical:
            # Konversi nilai 0/1 menjadi "Ya"/"Tidak" untuk visualisasi
            plot_data = convert_binary_series(data_normalization[feature_bin])

            plt.figure(figsize=(8, 5))
            ax = sns.countplot(x=plot_data)
            plt.title(f'Distribusi {feature_bin}')
            plt.xlabel(feature_bin)
            plt.ylabel('Count')

            # Anotasi count di atas bar
            for p in ax.patches:
                height = p.get_height()
                x = p.get_x() + p.get_width() / 2
                y = height + 0.1
                plt.text(x, y, f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            st.pyplot(plt)

            # Hitung jumlah masing-masing nilai (pakai data asli)
            value_counts = data_normalization[feature_bin].value_counts()
            most_common_value = value_counts.idxmax()
            most_common_count = value_counts.max()
            least_common_value = value_counts.idxmin()
            least_common_count = value_counts.min()

            # Ubah label 0/1 ke teks
            def convert_binary(val):
                if val == 4:
                    return "Ya"
                elif val == 0:
                    return "Tidak"
                else:
                    return str(val)

            st.write(f"**Nilai terbanyak di kolom `{feature_bin}`**: {convert_binary(most_common_value)} (jumlah: {most_common_count})")
            st.write(f"**Nilai tersedikit di kolom `{feature_bin}`**: {convert_binary(least_common_value)} (jumlah: {least_common_count})")
            st.markdown("---")
            st.write("")
