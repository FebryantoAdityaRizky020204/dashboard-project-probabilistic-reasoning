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
    page_title="Model Penalaran Probabilistik",
    page_icon="📊",
    layout="wide"
)

data_nbc = st.session_state.data_normalization.copy()


st.write("Dataset")
# Step awal ambil data
st.dataframe(data_nbc.head())

# Definisikan target
target_variables = ['GPA_Disc', 'GradeClass']

# Ambil semua fitur selain target
feature_variables = [col for col in data_nbc.columns if col not in target_variables]
st.write(feature_variables)
st.session_state.feature_variables = feature_variables

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


# Periksa jumlah kolom pada data latih dan uji
st.write(f"Dimensi data latih: {X_train_nbc.shape}")
st.write(f"Dimensi data uji: {X_test_nbc.shape}")

# Pastikan kolom yang digunakan untuk prediksi sama antara latih dan uji
assert X_train_nbc.shape[1] == X_test_nbc.shape[1], "Jumlah fitur pada data latih dan uji tidak sesuai!"


# Model untuk GPA_Disc
model_gpa_disc_nbc = CategoricalNB()
model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

# Model untuk GradeClass
model_grade_class_nbc = CategoricalNB()
model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

st.write(model_gpa_disc_nbc.fit)


# Model untuk GPA_Disc
model_gpa_disc_nbc = CategoricalNB()
model_gpa_disc_nbc.fit(X_train_nbc, y_gpa_disc_train_nbc)

# Model untuk GradeClass
model_grade_class_nbc = CategoricalNB()
model_grade_class_nbc.fit(X_train_nbc, y_grade_class_train_nbc)

# CPT untuk GPA_Disc
print("Tabel Probabilitas Fitur untuk GPA_Disc:")
for class_idx, class_log_prob in enumerate(model_gpa_disc_nbc.feature_log_prob_):
    print(f"\nKelas {class_idx}:")
    probs = np.exp(class_log_prob)  # balik dari log-prob ke prob
    for idx, prob in enumerate(probs):
        if isinstance(prob, np.ndarray):
            for cat_idx, p in enumerate(prob):
                print(f"  Feature {idx} - Category {cat_idx}: Probabilitas: {p:.4f}")
        else:
            print(f"  Feature {idx}: Probabilitas: {prob:.4f}")

# CPT untuk GradeClass
print("\nTabel Probabilitas Fitur untuk GradeClass:")
for class_idx, class_log_prob in enumerate(model_grade_class_nbc.feature_log_prob_):
    print(f"\nKelas {class_idx}:")
    probs = np.exp(class_log_prob)
    for idx, prob in enumerate(probs):
        if isinstance(prob, np.ndarray):
            for cat_idx, p in enumerate(prob):
                print(f"  Feature {idx} - Category {cat_idx}: Probabilitas: {p:.4f}")
        else:
            print(f"  Feature {idx}: Probabilitas: {prob:.4f}")


st.subheader("Inferensi Probabilitas")
# Hapus StudentID
X_train_nbc = X_train_nbc.drop(columns=['StudentID'])
X_test_nbc = X_test_nbc.drop(columns=['StudentID'])

# Gunakan KBinsDiscretizer untuk mendiskritisasi (opsional sebenarnya, kalau mau)
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

# Evaluasi model
st.write("\nAkurasi GPA_Disc:", accuracy_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc))
st.text(classification_report(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc))

st.write("\nAkurasi GradeClass:", accuracy_score(y_grade_class_test_nbc, y_pred_grade_class_nbc))
st.text(classification_report(y_grade_class_test_nbc, y_pred_grade_class_nbc))

st.markdown("---")

plt.figure(figsize=(12,8))
sns.heatmap(data_nbc.corr()[['GPA_Disc']].sort_values(by='GPA_Disc', ascending=False), annot=True, cmap="coolwarm")
plt.title("Sensitivity Analysis: Korelasi Variabel terhadap GPA_Disc", fontsize=16)
st.pyplot(plt)

# Contoh eksperimen dengan berbagai prior
prior_types = ['uniform', 'empirical']
for prior in prior_types:
    model = CategoricalNB(alpha=1.0, fit_prior=True)
    model.fit(X_train_nbc_transformed, y_gpa_disc_train_nbc)
    y_pred = model.predict(X_test_nbc_transformed)
    accuracy = accuracy_score(y_gpa_disc_test_nbc, y_pred)
    st.write(f"Akurasi dengan prior {prior}: {accuracy:.4f}")


st.title("Evaluasi Model")
st.subheader("Mean Absolute Error (MAE) atau Root Mean Squared Error (RMSE)")
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- MAE dan RMSE untuk GPA_Disc ---
mae_gpa_nbc = mean_absolute_error(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc)
rmse_gpa_nbc = np.sqrt(mean_squared_error(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc))

st.write(f"MAE GPA_Disc (NBC): {mae_gpa_nbc:.2f}")
st.write(f"RMSE GPA_Disc (NBC): {rmse_gpa_nbc:.2f}")

# --- MAE dan RMSE untuk GradeClass ---
# Menghitung MAE dan RMSE menggunakan prediksi dan nilai target untuk GradeClass
mae_grade_class_nbc = mean_absolute_error(y_grade_class_test_nbc, y_pred_grade_class_nbc)
rmse_grade_class_nbc = np.sqrt(mean_squared_error(y_grade_class_test_nbc, y_pred_grade_class_nbc))

st.write(f"MAE GradeClass (NBC): {mae_grade_class_nbc:.2f}")
st.write(f"RMSE GradeClass (NBC): {rmse_grade_class_nbc:.2f}")

st.subheader("Evaluasi untuk GPA_Disc: Akurasi, Presisi, Recall")
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score
import numpy as np

# --- Akurasi, Presisi, dan Recall untuk GPA_Disc ---
accuracy_gpa_disc_nbc = accuracy_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc)
precision_gpa_disc_nbc = precision_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc, average='weighted', labels=np.unique(y_pred_gpa_disc_nbc))
recall_gpa_disc_nbc = recall_score(y_gpa_disc_test_nbc, y_pred_gpa_disc_nbc, average='weighted', labels=np.unique(y_pred_gpa_disc_nbc))

st.write(f"Akurasi untuk GPA_Disc: {accuracy_gpa_disc_nbc:.4f}")
st.write(f"Presisi untuk GPA_Disc: {precision_gpa_disc_nbc:.4f}")
st.write(f"Recall untuk GPA_Disc: {recall_gpa_disc_nbc:.4f}")

st.subheader("Evaluasi untuk GradeClass: Akurasi, Presisi, Recall, AUC")
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# --- Akurasi, Presisi, dan Recall untuk GradeClass ---
accuracy_grade_class_nbc = accuracy_score(y_grade_class_test_nbc, y_pred_grade_class_nbc)
precision_grade_class_nbc = precision_score(y_grade_class_test_nbc, y_pred_grade_class_nbc, average='weighted')
recall_grade_class_nbc = recall_score(y_grade_class_test_nbc, y_pred_grade_class_nbc, average='weighted')

st.write(f"Akurasi untuk GradeClass: {accuracy_grade_class_nbc:.4f}")
st.write(f"Presisi untuk GradeClass: {precision_grade_class_nbc:.4f}")
st.write(f"Recall untuk GradeClass: {recall_grade_class_nbc:.4f}")

# --- AUC untuk GradeClass ---
y_prob_grade_class_nbc = model_grade_class_nbc_inf.predict_proba(X_test_nbc_transformed)

# Binarisasi label target untuk multi-class AUC
y_true_bin_grade_class_nbc = label_binarize(y_grade_class_test_nbc, classes=np.unique(y_grade_class_test_nbc))

# Hitung AUC untuk multi-class
auc_grade_class_nbc = roc_auc_score(y_true_bin_grade_class_nbc, y_prob_grade_class_nbc, average='macro', multi_class='ovr')
st.write(f"AUC GradeClass (NBC): {auc_grade_class_nbc:.2f}")

st.subheader("Kalibrasi Probabilistik")
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# --- Kalibrasi Probabilistik ---
plt.figure(figsize=(10, 6))
for i in range(y_prob_grade_class_nbc.shape[1]):
    prob_true, prob_pred = calibration_curve(y_true_bin_grade_class_nbc[:, i], y_prob_grade_class_nbc[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}', color=plt.cm.jet(i / y_prob_grade_class_nbc.shape[1]))

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Calibration Curve Naive Bayes - GradeClass')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.grid()
st.pyplot(plt)

# Prediksi probabilitas untuk GPA_Disc
y_prob_gpa_disc_nbc = model_gpa_disc_nbc_inf.predict_proba(X_test_nbc_transformed)
# Binarize true labels for GPA_Disc
y_true_bin_gpa_disc_nbc = label_binarize(y_gpa_disc_test_nbc, classes=np.unique(y_gpa_disc_test_nbc))

# --- Kalibrasi Probabilistik untuk GPA_Disc ---
plt.figure(figsize=(10, 6))

# Untuk setiap kelas pada GPA_Disc, lakukan kalibrasi
for i in range(y_prob_gpa_disc_nbc.shape[1]):
    prob_true, prob_pred = calibration_curve(y_true_bin_gpa_disc_nbc[:, i], y_prob_gpa_disc_nbc[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}', color=plt.cm.jet(i / y_prob_gpa_disc_nbc.shape[1]))

# Plot diagonal referensi
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Calibration Curve Naive Bayes - GPA_Disc')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.grid()
st.pyplot(plt)

