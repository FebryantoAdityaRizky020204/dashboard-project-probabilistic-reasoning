# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import networkx as nx
# import numpy as np
# import pickle
# import json
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve



# import streamlit as st
# from streamlit_option_menu import option_menu

# st.set_page_config(
#     page_title="Testing",
#     page_icon="🧪",
#     layout="wide"
# )

# # Kolom yang digunakan sebagai fitur input
# feature_variables = [
#     'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
#     'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
#     'Sports', 'Music', 'Volunteering', 'StudyTimeWeekly_Disc'
# ]

# # Kolom evidence yang dibutuhkan model Bayesian
# evidence_columns = [
#     'StudyTimeWeekly_Disc', 'ParentalEducation', 'Absences',
#     'ParentalSupport', 'Extracurricular', 'Tutoring'
# ]

# # Load data dummy awal
# data_test = pd.read_csv("./data/data_dummy_test.csv")
# data_test['Hapus'] = False
# columns_order = ['Hapus'] + feature_variables
# template_df = data_test[columns_order]

# # Tampilkan editor
# st.subheader("Input Data Uji")
# st.write("Data dummy `data_dummy_test.csv`")
# input_df = st.data_editor(
#     template_df,
#     num_rows="dynamic",
#     use_container_width=True,
#     key="data_editor_hapus"
# )

# # Filter baris yang tidak dihapus
# if not input_df.empty:
#     input_df['Hapus'] = input_df['Hapus'].fillna(False)
#     cleaned_df = input_df[input_df['Hapus'] == False].drop(columns='Hapus')
# else:
#     st.warning("Belum ada data yang valid untuk diproses.")
#     st.stop()

# # Load model Bayesian Network
# with open('./models/bn/model_bn.pkl', 'rb') as model_file:
#     model_bn = pickle.load(model_file)

# with open('./models/nbc/model_gpa_disc_nbc.pkl', 'rb') as model_file_gpa_disc:
#     model_gpa_nbc = pickle.load(model_file_gpa_disc)

# with open('./models/nbc/model_grade_class_nbc.pkl', 'rb') as model_file_gradeclass_disc:
#     model_gradeclass_nbc = pickle.load(model_file_gradeclass_disc)

# # Fungsi prediksi per baris
# def predict_per_row(model, evidence_dict):
#     # Gunakan model.predict() untuk memprediksi nilai target
#     gpa_pred = model.predict(pd.DataFrame([evidence_dict]))["GPA_Disc"].iloc[0]
#     grade_pred = model.predict(pd.DataFrame([evidence_dict]))["GradeClass"].iloc[0]
#     return gpa_pred, grade_pred


# # Tombol prediksi
# if st.button("Prediksi"):
#     prediction_results = []

#     for _, row in cleaned_df.iterrows():
#         try:
#             # Ambil nilai evidence dari baris saat ini
#             evidence_input = {col: int(row[col]) for col in evidence_columns}
#             gpa_pred, grade_pred = predict_per_row(model_bn, evidence_input)
#             prediction_results.append({
#                 "StudentID": row["StudentID"],
#                 "Predicted_GPA_Disc": gpa_pred,
#                 "Predicted_GradeClass": grade_pred
#             })
#         except Exception as e:
#             st.error(f"Kesalahan pada StudentID {row['StudentID']}: {e}")


#     if prediction_results:
#         result_df = pd.DataFrame(prediction_results)
#         st.subheader("Hasil Prediksi Bayesian Network")
#         st.dataframe(result_df)
#     else:
#         st.warning("Tidak ada hasil prediksi yang valid.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import numpy as np
import pickle
import joblib

import streamlit as st
from streamlit_option_menu import option_menu

from sklearn.preprocessing import KBinsDiscretizer

st.set_page_config(
    page_title="Testing",
    page_icon="🧪",
    layout="wide"
)

# ================================
# SETUP
# ================================
feature_variables = [
    'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
    'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
    'Sports', 'Music', 'Volunteering', 'StudyTimeWeekly_Disc'
]

evidence_columns = [
    'StudyTimeWeekly_Disc', 'ParentalEducation', 'Absences',
    'ParentalSupport', 'Extracurricular', 'Tutoring'
]

# ================================
# LOAD DATA
# ================================
data_test = pd.read_csv("./data/data_dummy_test.csv")
data_test['Hapus'] = False
columns_order = ['Hapus'] + feature_variables
template_df = data_test[columns_order]

st.subheader("Input Data Uji")
st.write("Data dummy `data_dummy_test.csv`")
input_df = st.data_editor(
    template_df,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor_hapus"
)

if not input_df.empty:
    input_df['Hapus'] = input_df['Hapus'].fillna(False)
    cleaned_df = input_df[input_df['Hapus'] == False].drop(columns='Hapus')
else:
    st.warning("Belum ada data yang valid untuk diproses.")
    st.stop()

# ================================
# LOAD MODELS
# ================================
with open('./models/BN/model_bn.pkl', 'rb') as model_file:
    model_bn = pickle.load(model_file)

with open('./models/NBC/model_gpa_disc_nbc.pkl', 'rb') as model_file_gpa_disc:
    model_gpa_nbc = pickle.load(model_file_gpa_disc)

with open('./models/NBC/model_grade_class_nbc.pkl', 'rb') as model_file_gradeclass_disc:
    model_gradeclass_nbc = pickle.load(model_file_gradeclass_disc)

# Optional: load saved discretizer
# try:
#     with open('./models/NBC/discretizer_nbc.pkl', 'rb') as disc_file:
#         discretizer_nbc = pickle.load(disc_file)
# except:
# st.warning("Discretizer tidak ditemukan. Menggunakan discretizer default.")
# discretizer_nbc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# discretizer_nbc.fit(cleaned_df.drop(columns=['StudentID']))  # dummy fit

# ================================
# FUNGSI PREDIKSI
# ================================
def predict_bn_row(model, evidence_dict):
    gpa_pred = model.predict(pd.DataFrame([evidence_dict]))["GPA_Disc"].iloc[0]
    grade_pred = model.predict(pd.DataFrame([evidence_dict]))["GradeClass"].iloc[0]
    return gpa_pred, grade_pred

def predict_nbc_row(model_gpa, model_grade, row_transformed):
    gpa_pred = model_gpa.predict([row_transformed])[0]
    grade_pred = model_grade.predict([row_transformed])[0]
    return gpa_pred, grade_pred

# ================================
# TOMBOL PREDIKSI
# ================================
if st.button("Prediksi"):
    prediction_results = []

    # Buat salinan untuk transformasi NBC (tanpa StudentID)
    cleaned_nbc_df = cleaned_df.drop(columns=['StudentID'])
    # cleaned_nbc_transformed = discretizer_nbc.transform(cleaned_nbc_df)

    # Pastikan bahwa cleaned_nbc_transformed adalah numpy array dan cocok dengan indeks cleaned_df
    cleaned_nbc_transformed_df = pd.DataFrame(cleaned_nbc_df, columns=cleaned_nbc_df.columns)

    # Prediksi untuk setiap baris pada cleaned_df
    for i, row in cleaned_df.iterrows():
        try:
            # --- Bayesian --- (prediksi seperti biasa)
            evidence_input = {col: int(row[col]) for col in evidence_columns}
            gpa_bn, grade_bn = predict_bn_row(model_bn, evidence_input)

            # --- Naive Bayes --- (gunakan baris yang sudah ditransformasikan)
            row_nbc = cleaned_nbc_transformed_df.iloc[i]  # Ambil baris yang sudah ditransformasikan
            gpa_nbc, grade_nbc = predict_nbc_row(model_gpa_nbc, model_gradeclass_nbc, row_nbc)

            # Simpan hasil prediksi
            prediction_results.append({
                "StudentID": row["StudentID"],
                "Predicted_GPA_Disc_BN": gpa_bn,
                "Predicted_GradeClass_BN": grade_bn,
                "Predicted_GPA_Disc_NBC": gpa_nbc,
                "Predicted_GradeClass_NBC": grade_nbc
            })

        except Exception as e:
            st.error(f"Kesalahan pada StudentID {row['StudentID']}: {e}")

    if prediction_results:
        result_df = pd.DataFrame(prediction_results)
        st.subheader("Hasil Prediksi: Bayesian Network vs Naive Bayes")
        st.dataframe(result_df)
    else:
        st.warning("Tidak ada hasil prediksi yang valid.")
