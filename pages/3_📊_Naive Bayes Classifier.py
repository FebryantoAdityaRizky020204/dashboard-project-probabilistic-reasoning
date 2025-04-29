import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from pgmpy.estimators import HillClimbSearch, K2Score, BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Naive Bayes Classifier",
    page_icon="📊",
    layout="wide"
)

data_nbc = pd.read_csv("./data/data_normalization.csv")


with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Struktur NBC",
                 "Split Data",
                 "Model Dan CPT",
                 "Inferensi",
                 "Analisis Sensitifitas",
                 "Evaluasi Model",
                 "Validasi Model"],
    )


if selected == "Struktur NBC":
    # Step awal ambil data
    st.write("### Data Normalization")
    st.dataframe(data_nbc.head())
    st.markdown("---")


    # Definisikan target
    st.write("#### Target Variabel")
    target_variables = ['GPA_Disc', 'GradeClass']
    st.write(target_variables)
    
    st.write()
    st.write("#### Target Variabel")
    # Ambil semua fitur selain target
    feature_variables = [col for col in data_nbc.columns if col not in target_variables]
    st.write(feature_variables)
    st.session_state.feature_variables = feature_variables
    

    st.markdown("---")
    st.write("### Struktur Naive Bayes Classifier")
    # Buat Directed Graph
    G_nbc = nx.DiGraph()
    for feature in feature_variables:
        for target in target_variables:
            G_nbc.add_edge(feature, target)
    
    # Visualisasikan
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_nbc)
    nx.draw(G_nbc, pos, with_labels=True, node_size=3000, node_color="lightgreen", font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Struktur Naive Bayes Classifier (Semua Fitur ke Target)", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

if selected == "Split Data":
    st.write("### Split Data")
    # Copy feature variables from session state
    with st.echo():
        feature_variables = st.session_state.feature_variables.copy()
        
        # Pisahkan fitur dan target untuk data latih dan data uji
        X_nbc = data_nbc[feature_variables]
        y_gpa_disc_nbc = data_nbc['GPA_Disc']
        y_grade_class_nbc = data_nbc['GradeClass']
        
        # Split data untuk GPA_Disc
        X_train_nbc, X_test_nbc, y_gpa_disc_train_nbc, y_gpa_disc_test_nbc = train_test_split(
            X_nbc, y_gpa_disc_nbc, test_size=0.2, random_state=42, stratify=y_gpa_disc_nbc
        )
        
        # Split data untuk GradeClass (menggunakan X_nbc yang sama)
        _, _, y_grade_class_train_nbc, y_grade_class_test_nbc = train_test_split(
            X_nbc, y_grade_class_nbc, test_size=0.2, random_state=42, stratify=y_grade_class_nbc
        )
    
    # Simpan semua variabel ke session state
    st.session_state.update({
        'X_nbc': X_nbc,
        'y_gpa_disc_nbc': y_gpa_disc_nbc,
        'y_grade_class_nbc': y_grade_class_nbc,
        'X_train_nbc': X_train_nbc,
        'X_test_nbc': X_test_nbc,
        'y_gpa_disc_train_nbc': y_gpa_disc_train_nbc,
        'y_gpa_disc_test_nbc': y_gpa_disc_test_nbc,
        'y_grade_class_train_nbc': y_grade_class_train_nbc,
        'y_grade_class_test_nbc': y_grade_class_test_nbc
    })
    
    # Periksa jumlah kolom pada data latih dan uji
    st.write(f"Dimensi data latih: `{X_train_nbc.shape}`")
    st.write(f"Dimensi data uji: `{X_test_nbc.shape}`")
    
    # Pastikan kolom yang digunakan untuk prediksi sama antara latih dan uji
    assert X_train_nbc.shape[1] == X_test_nbc.shape[1], "Jumlah fitur pada data latih dan uji tidak sesuai!"
    

if selected == "Model Dan CPT":
    st.write("### Model dan CPT")
    with st.echo():
        X_train_nbc = st.session_state.X_train_nbc
        y_gpa_disc_train_nbc = st.session_state.y_gpa_disc_train_nbc
        y_grade_class_train_nbc = st.session_state.y_grade_class_train_nbc
        
        # Model untuk GPA_Disc
        model_gpa_disc_nbc = CategoricalNB()
        model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)
        
        # Model untuk GradeClass
        model_grade_class_nbc = CategoricalNB()
        model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

    # Misalkan X_train_nbc dan y_train_nbc sudah ada di session_state

    X_train_nbc = st.session_state.X_train_nbc
    y_gpa_disc_train_nbc = st.session_state.y_gpa_disc_train_nbc
    y_grade_class_train_nbc = st.session_state.y_grade_class_train_nbc

    # Model untuk GPA_Disc
    model_gpa_disc_nbc = CategoricalNB()
    model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

    # CPT untuk GPA_Disc
    st.write("### Tabel Probabilitas Fitur untuk GPA_Disc:")

    cpt_data = []

    # Mengambil probabilitas logaritma dan mengonversinya menjadi probabilitas biasa
    for class_idx, class_log_prob in enumerate(model_gpa_disc_nbc.feature_log_prob_):
        probs = np.exp(class_log_prob)  # Mengonversi log-probabilitas menjadi probabilitas
        for idx, prob in enumerate(probs):
            if isinstance(prob, np.ndarray):
                for cat_idx, p in enumerate(prob):
                    cpt_data.append({
                        "Class Index": class_idx,
                        "Feature Index": idx,
                        "Category Index": cat_idx,
                        "Feature Name": f"Feature {idx}",
                        "Category Name": f"Category {cat_idx}",
                        "Probabilitas": p
                    })
            else:
                cpt_data.append({
                    "Class Index": class_idx,
                    "Feature Index": idx,
                    "Category Index": "N/A",  # Tidak ada kategori jika hanya satu nilai
                    "Feature Name": f"Feature {idx}",
                    "Category Name": "N/A",
                    "Probabilitas": prob
                })

    # Mengonversi data CPT ke DataFrame untuk tampilan yang lebih rapi
    cpt_df = pd.DataFrame(cpt_data)

    # Menampilkan tabel dengan scroll dan height terbatas
    st.dataframe(cpt_df, height=300)  # Mengatur tinggi untuk scroll


    # CPT untuk GradeClass
    st.write("### Tabel Probabilitas Fitur untuk GradeClass:")

    cpt_data = []

    # Mengambil probabilitas logaritma dan mengonversinya menjadi probabilitas biasa
    for class_idx, class_log_prob in enumerate(model_grade_class_nbc.feature_log_prob_):
        probs = np.exp(class_log_prob)  # Mengonversi log-probabilitas menjadi probabilitas
        for idx, prob in enumerate(probs):
            if isinstance(prob, np.ndarray):
                for cat_idx, p in enumerate(prob):
                    cpt_data.append({
                        "Class Index": class_idx,
                        "Feature Index": idx,
                        "Category Index": cat_idx,
                        "Feature Name": f"Feature {idx}",
                        "Category Name": f"Category {cat_idx}",
                        "Probabilitas": p
                    })
            else:
                cpt_data.append({
                    "Class Index": class_idx,
                    "Feature Index": idx,
                    "Category Index": "N/A",  # Tidak ada kategori jika hanya satu nilai
                    "Feature Name": f"Feature {idx}",
                    "Category Name": "N/A",
                    "Probabilitas": prob
                })

    # Mengonversi data CPT ke DataFrame untuk tampilan yang lebih rapi
    cpt_df = pd.DataFrame(cpt_data)

    # Menampilkan tabel dengan scroll dan height terbatas
    st.dataframe(cpt_df, height=300)  # Mengatur tinggi untuk scroll


if selected == "Inferensi":
    st.write("### Inferensi")

    feature_variables = st.session_state.feature_variables.copy()
        
    # Pisahkan fitur dan target untuk data latih dan data uji
    X_nbc = data_nbc[feature_variables]
    y_gpa_disc_nbc = data_nbc['GPA_Disc']
    y_grade_class_nbc = data_nbc['GradeClass']
    
    # Split data untuk GPA_Disc
    X_train_nbc, X_test_nbc, y_gpa_disc_train_nbc, y_gpa_disc_test_nbc = train_test_split(
        X_nbc, y_gpa_disc_nbc, test_size=0.2, random_state=42, stratify=y_gpa_disc_nbc
    )
    
    # Split data untuk GradeClass (menggunakan X_nbc yang sama)
    _, _, y_grade_class_train_nbc, y_grade_class_test_nbc = train_test_split(
        X_nbc, y_grade_class_nbc, test_size=0.2, random_state=42, stratify=y_grade_class_nbc
    )

    X_train_nbc = st.session_state.X_train_nbc
    y_gpa_disc_train_nbc = st.session_state.y_gpa_disc_train_nbc
    y_grade_class_train_nbc = st.session_state.y_grade_class_train_nbc
    
    # Model untuk GPA_Disc
    model_gpa_disc_nbc = CategoricalNB()
    model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)
    
    # Model untuk GradeClass
    model_grade_class_nbc = CategoricalNB()
    model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

    # Misalkan X_train_nbc dan y_train_nbc sudah ada di session_state

    X_train_nbc = st.session_state.X_train_nbc
    y_gpa_disc_train_nbc = st.session_state.y_gpa_disc_train_nbc
    y_grade_class_train_nbc = st.session_state.y_grade_class_train_nbc

    # Model untuk GPA_Disc
    model_gpa_disc_nbc = CategoricalNB()
    model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

    with st.echo():
        # Menghapus kolom 'StudentID'
        X_train_nbc = X_train_nbc.drop(columns=['StudentID'])
        X_test_nbc = X_test_nbc.drop(columns=['StudentID'])

        # Gunakan KBinsDiscretizer untuk mendiskritisasi
        discretizer_nbc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

        # Transformasi semua fitur (tanpa StudentID)
        X_train_nbc_transformed = discretizer_nbc.fit_transform(X_train_nbc)
        X_test_nbc_transformed = discretizer_nbc.transform(X_test_nbc)

        # Model untuk GPA_Disc
        model_gpa_disc_nbc_inf = CategoricalNB()
        model_gpa_disc_nbc_inf.fit(X_train_nbc_transformed, y_gpa_disc_train_nbc)

        # Model untuk GradeClass
        model_grade_class_nbc_inf = CategoricalNB()
        model_grade_class_nbc_inf.fit(X_train_nbc_transformed, y_grade_class_train_nbc)

        # Prediksi untuk data uji
        y_pred_gpa_disc_nbc = model_gpa_disc_nbc_inf.predict(X_test_nbc_transformed)
        y_pred_grade_class_nbc = model_grade_class_nbc_inf.predict(X_test_nbc_transformed)

        # Evaluasi model GPA_Disc
        gpa_disc_accuracy = accuracy_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc)
        gpa_disc_class_report = classification_report(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc)

        # Evaluasi model GradeClass
        grade_class_accuracy = accuracy_score(y_grade_class_test_nbc, y_pred_grade_class_nbc)
        grade_class_class_report = classification_report(y_grade_class_test_nbc, y_pred_grade_class_nbc)

        # Menampilkan hasil evaluasi dengan tampilan yang lebih rapi
        st.markdown("## Evaluasi Model Naive Bayes")

        # Akurasi GPA_Disc
        st.markdown("### 1. Akurasi dan Klasifikasi untuk GPA_Disc")
        st.metric(label="Akurasi GPA_Disc", value=f"{gpa_disc_accuracy * 100:.2f}%")
        st.text_area("Classification Report GPA_Disc", gpa_disc_class_report, height=200)

        # Akurasi GradeClass
        st.markdown("### 2. Akurasi dan Klasifikasi untuk GradeClass")
        st.metric(label="Akurasi GradeClass", value=f"{grade_class_accuracy * 100:.2f}%")
        st.text_area("Classification Report GradeClass", grade_class_class_report, height=200)

if selected == "Analisis Sensitifitas":
    st.write("### Analisis Sensitifitas")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_nbc.corr()[['GPA_Disc']].sort_values(by='GPA_Disc', ascending=False), annot=True, cmap="coolwarm")
    plt.title("Sensitivity Analysis: Korelasi Variabel terhadap GPA_Disc", fontsize=16)
    st.pyplot(plt)
    
    st.write("Heatmap ini menunjukkan korelasi antara berbagai variabel dengan GPA_Disc. Warna merah menunjukkan korelasi positif, sementara biru menunjukkan korelasi negatif. GPA_Disc memiliki korelasi sempurna dengan dirinya sendiri (1), dan GradeClass serta Absences menunjukkan korelasi positif yang cukup kuat. Sebaliknya, ParentalSupport dan StudyTimeWeekly_Disc menunjukkan korelasi negatif yang lemah. Variabel lainnya menunjukkan korelasi yang sangat lemah, mendekati nol.")
    st.markdown("---")

    # st.write("### Contoh eksperimen dengan berbagai prior")
    # with st.echo():
    #     prior_types = ['uniform', 'empirical']
    #     for prior in prior_types:
    #         model = CategoricalNB(alpha=1.0, fit_prior=True)
    #         model.fit(X_train_nbc_transformed, y_gpa_disc_train_nbc)
    #         y_pred = model.predict(X_test_nbc_transformed)
    #         accuracy = accuracy_score(y_gpa_disc_test_nbc, y_pred)
    #         st.write(f"Akurasi dengan prior `{prior}`: `{accuracy:.4f}`")