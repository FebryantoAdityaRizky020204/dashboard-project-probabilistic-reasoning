import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import numpy as np
import pickle
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve



import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Testing",
    page_icon="🧪",
    layout="wide"
)

data_n = pd.read_csv("./data/data_normalization.csv")

with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Bayesian Network",
                 "Naive Bayes Classifier",],
    )

if selected == "Bayesian Network":
    st.title("Testing Model Bayesian Network")
    st.markdown("---")

    # Load model dan inference
    with open('./models/BN/model_bn.pkl', 'rb') as model_file:
        model_bn = pickle.load(model_file)

    with open('./models/BN/inference_bn.pkl', 'rb') as inference_file:
        inference_bn = pickle.load(inference_file)

    # Load fitur yang harus ada
    with open('./models/BN/features_bn.pkl', 'rb') as f:
        features_bn = pickle.load(f)
    feature_columns = features_bn.columns.tolist()

    # State names dari model
    state_names = {
        'StudyTimeWeekly_Disc': [0, 1, 2, 3],
        'ParentalEducation': [0, 1, 2, 3, 4],
        'Absences': [0, 1, 2, 3, 4],
        'ParentalSupport': [0, 1, 2, 3, 4],
        'Extracurricular': [0, 1],
        'Tutoring': [0, 1]
    }

    # Fungsi validasi
    def validate_evidence(evidence_input):
        for feature, value in evidence_input.items():
            valid_states = state_names.get(feature)
            if valid_states:
                if not isinstance(value, int):
                    value = int(value)
                if value not in valid_states:
                    raise ValueError(f"Nilai '{value}' untuk fitur '{feature}' tidak valid. Pilih dari {valid_states}")
        return evidence_input

    # Fungsi prediksi
    def predict_gpa_and_grade(evidence_input):
        evidence_input = validate_evidence(evidence_input)
        gpa_result = inference_bn.query(variables=['GPA_Disc'], evidence=evidence_input)
        gpa_pred = gpa_result.values.argmax()
        evidence_with_gpa = evidence_input.copy()
        evidence_with_gpa['GPA_Disc'] = gpa_pred
        grade_result = inference_bn.query(variables=['GradeClass'], evidence=evidence_with_gpa)
        grade_pred = grade_result.values.argmax()
        return gpa_pred, grade_pred

    # Input interaktif Streamlit
    st.header("Input Data untuk Prediksi")
    evidence_input = {
        'StudyTimeWeekly_Disc': st.selectbox("Waktu Belajar per Minggu (0=Sedikit, 3=Banyak)", state_names['StudyTimeWeekly_Disc'], index=0),
        'ParentalEducation': st.selectbox("Tingkat Pendidikan Orang Tua (0:None, 1:HighSc, 2:College, 3:Bachelor, 4:Hhigher)", state_names['ParentalEducation'], index=1),
        'Absences': st.selectbox("Absensi (0-4)", state_names['Absences'], index=1),
        'ParentalSupport': st.selectbox("Dukungan Orang Tua (0=Tidak ada, 4=Sangat tinggi)", state_names['ParentalSupport'], index=1),
        'Extracurricular': st.radio("Ikut Ekstrakurikuler?", options=state_names['Extracurricular'], index=0, format_func=lambda x: "Tidak" if x == 0 else "Ya"),
        'Tutoring': st.radio("Ikut Les/Tutoring?", options=state_names['Tutoring'], index=0, format_func=lambda x: "Tidak" if x == 0 else "Ya"),
    }

    # Tombol prediksi
    if st.button("Prediksi"):
        try:
            predicted_gpa, predicted_grade_class = predict_gpa_and_grade(evidence_input)
            st.success(f"Prediksi GPA_Disc: `{predicted_gpa}`")
            st.success(f"Prediksi GradeClass: `{predicted_grade_class}`")
        except ValueError as e:
            st.error(str(e))

if selected == "Naive Bayes Classifier":
    # Contoh fitur dari dataset sebelumnya (ubah sesuai dataset kamu)
    feature_variables = ['Semester', 'IPK_SemesterSebelumnya', 'Kehadiran', 'Tugas', 'Kedisiplinan']

    # Buat template input (1 baris data)
    input_template = pd.DataFrame([{col: 0 for col in feature_variables}])

    # Tampilkan editor
    input_df = st.data_editor(input_template, num_rows="fixed", use_container_width=True)
    st.dataframe(input_df)

    # Discretizer jika digunakan
    # asumsikan sudah didefinisikan: `discretizer_nbc`
    with open('./models/BN/model_bn.pkl', 'rb') as model_file:
        model_bn = pickle.load(model_file)

    # Tombol prediksi
    if st.button("Prediksi"):
        # Lakukan transformasi diskritisasi
        from pgmpy.inference import VariableElimination
        inference = VariableElimination(model_bn)
        evidence_dict = input_df.iloc[0].to_dict()
        query_result = inference.query(variables=["GPA_Disc"], evidence=evidence_dict)
        st.write(query_result)
