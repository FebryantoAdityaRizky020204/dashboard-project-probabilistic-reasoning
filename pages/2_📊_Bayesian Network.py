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
    page_title="Bayesian Network",
    page_icon="📊",
    layout="wide"
)

data_bn = pd.read_csv("./data/data_normalization.csv")
data_bn.head()

with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Struktur BN",
                 "Feature dan Split",
                 "CPT",
                 "Inferensi Probabilistik",
                 "Analisis Kausal",
                 "Analisis Intervensi",
                 "Evaluasi Model",
                 "Validasi Model"]
    )


if selected == "Struktur BN":
    st.title("Struktur Bayesian Network")
    st.write("Edges")

    with st.echo():
        edges_bn = [
            ('ParentalEducation', 'StudyTimeWeekly_Disc'),
            ('StudyTimeWeekly_Disc', 'GPA_Disc'),
            ('ParentalSupport', 'GPA_Disc'),
            ('Tutoring', 'GPA_Disc'),
            ('Extracurricular', 'GPA_Disc'),
            ('Absences', 'GPA_Disc'),
            ('GPA_Disc', 'GradeClass')
        ]
    st.markdown("---")
    with open("./data/data_bn_split_info.json", "r") as f:
        loaded_info_bn_fs = json.load(f)
    
    st.write("### Kolom Yang Dipakai")
    st.write(loaded_info_bn_fs["column_use"])
    
    st.markdown("---")
    st.write("Visualisasi Struktur Bayesian Network")
    # Membuat Directed Graph dari edges yang sudah didefinisikan
    model = nx.DiGraph()
    model.add_edges_from(edges_bn)

    # 4. Visualisasi struktur jaringan
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(model)  # Posisi node menggunakan layout spring
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, node_color='skyblue',
            font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Bayesian Network Structure for GPA Prediction (with Continuous Variables)", fontsize=14)
    st.pyplot(plt)


if selected == "Feature dan Split":
    st.title("Feature dan Split")

    with open("./data/data_bn_split_info.json", "r") as f:
        loaded_info_bn_fs = json.load(f)
    
    st.write("### Kolom Yang Dipakai")
    st.write(loaded_info_bn_fs["column_use"])

    st.markdown("---")
    st.write("### Split Data")
    code_fs = '''
        # --- Split data menjadi training dan testing ---
        train_features, test_features, train_target, test_target = train_test_split(
            features_bn,
            target_data,
            test_size=0.2,
            random_state=42,
            stratify=data_bn['GradeClass']
        )

        # Gabungkan fitur dan target jadi satu dataframe
        train_data_bn = train_features.copy()
        train_data_bn['GPA_Disc'] = train_target['GPA_Disc']
        train_data_bn['GradeClass'] = train_target['GradeClass']

        test_data_bn = test_features.copy()
        test_data_bn['GPA_Disc'] = test_target['GPA_Disc']
        test_data_bn['GradeClass'] = test_target['GradeClass']
    '''
    st.code(code_fs, language='python')
    st.markdown("---")

    st.write("### Check Result")
    st.write(f"Training Set Size : `{loaded_info_bn_fs["train_size"]}`")
    st.write(f"Testing Set Size : `{loaded_info_bn_fs["test_size"]}`")
    st.write(f"Fitur yang digunakan : `{loaded_info_bn_fs["feature_columns"]}` ")
    st.write(f"Target : `{loaded_info_bn_fs["target_columns"]}` ")


if selected == "CPT":
    st.write("### Conditional Probability Table (CPT)")

    with open('./models/BN/model_bn.pkl', 'rb') as model_file:
        model_bn = pickle.load(model_file)

    code_model_bn = '''
    # Membuat model jaringan
    model_bn = BayesianNetwork(edges_bn)

    # Estimasi parameter (CPT) dengan menggunakan data training
    # Menggunakan estimator BayesianEstimator
    model_bn.fit(train_data_bn, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)
    '''
    st.code(code_model_bn, language='python')
    st.markdown("---")
    
    # --- Menampilkan CPT untuk setiap variabel dalam jaringan ---
    for cpd in model_bn.get_cpds():
        st.write(f"\nCPT untuk {cpd.variable}:")
        st.write(cpd)
        st.write("")

if selected == "Inferensi Probabilistik":
    st.write("### Inferensi Probabilistik")

    with open("./data/inference_bn_info.json", "r") as f:
        loaded_inference_info = json.load(f)
    
    code_inference = '''
    # Langkah 5: Lakukan inferensi dengan data test
    inference = VariableElimination(model_bn)

    # Simpan hasil prediksi
    predictions_gpa_disc = []
    predictions_grade_class = []

    # Lakukan prediksi untuk setiap baris data di test_data_bn
    for i in range(len(test_data_bn)):
        # Pastikan hanya menggunakan fitur relevan (tanpa target)
        evidence = test_data_bn.iloc[i].drop(['GPA_Disc', 'GradeClass'])

        # Prediksi distribusi probabilitas untuk 'GPA_Disc'
        gpa_disc_prob = inference.query(variables=['GPA_Disc'], evidence=evidence)
        predictions_gpa_disc.append(gpa_disc_prob.values.argmax())  # Ambil nilai dengan probabilitas tertinggi

        # Prediksi distribusi probabilitas untuk 'GradeClass'
        grade_class_prob = inference.query(variables=['GradeClass'], evidence=evidence)
        predictions_grade_class.append(grade_class_prob.values.argmax())  # Ambil nilai dengan probabilitas tertinggi

    # Ubah GradeClass jadi angka
    true_gpa_disc = test_data_bn['GPA_Disc'].astype(int)
    true_grade_class = test_data_bn['GradeClass'].astype(int)

    # Sekarang baru bisa hitung
    gpa_disc_accuracy = accuracy_score(true_gpa_disc, predictions_gpa_disc)
    grade_class_accuracy = accuracy_score(true_grade_class, predictions_grade_class)
    '''

    st.code(code_inference, language='python')
    st.markdown("---")
    
    st.write(f"Akurasi Prediksi GPA_Disc: `{loaded_inference_info["gpa_acc"] * 100:.2f}%`")
    st.write(f"Akurasi Prediksi GradeClass: `{loaded_inference_info["gradeclass_acc"] * 100:.2f}%`")

if selected == "Analisis Kausal":
    st.write("### Analisis Kausal")


if selected == "Analisis Intervensi":
    st.write("### Analisis Intervensi (what-if analysis)")

    with st.expander("Code"):
        code_bn_anaint = '''
        # Fungsi untuk What-If Analysis
        def what_if_analysis(inference, evidence, target_variable, var_to_change, new_value, model):
            modified_evidence = evidence.copy()

            # Mendapatkan status valid untuk variabel dari model
            valid_states = model.get_cpds(var_to_change).state_names[var_to_change]  # Menggunakan state_names untuk status variabel

            # Periksa jika new_value valid
            if new_value not in valid_states:
                raise ValueError(f"Nilai {new_value} tidak valid untuk variabel {var_to_change}. Valid states: {valid_states}")

            modified_evidence[var_to_change] = new_value
            modified_prob = inference.query(variables=[target_variable], evidence=modified_evidence)
            return modified_prob

        # Contoh: What-if Analysis untuk Absences (menggunakan angka 4 untuk status "High")
        what_if_result = what_if_analysis(inference, evidence, 'GPA_Disc', 'Absences', 4, model_bn)  # Menggunakan 4 sebagai status "High"
        print("Hasil What-If Analysis untuk Absences = 4 (High):", what_if_result)
        '''
        st.code(code_bn_anaint, language='python')

    st.markdown("---")

    st.title("📊 What-If Analysis dengan Bayesian Network")

    # Load model dari file lokal
    with open('./models/BN/model_bn.pkl', 'rb') as model_file:
        model_bn = pickle.load(model_file)

    with open('./models/BN/inference_bn.pkl', 'rb') as inference_file:
        inference_bn = pickle.load(inference_file)

    with open('./models/BN/features_bn.pkl', 'rb') as f:
        features_bn = pickle.load(f)

    # Fungsi What-If
    def what_if_analysis(inference, evidence, target_variable, var_to_change, new_value, model):
        modified_evidence = evidence.copy()
        valid_states = model.get_cpds(var_to_change).state_names[var_to_change]
        if new_value not in valid_states:
            raise ValueError(f"Nilai {new_value} tidak valid untuk variabel {var_to_change}. Valid states: {valid_states}")
        modified_evidence[var_to_change] = new_value
        modified_prob = inference.query(variables=[target_variable], evidence=modified_evidence)
        return modified_prob

    # Ambil nama fitur dan state dari model
    feature_columns = features_bn.columns.tolist()
    state_names = {cpd.variable: cpd.state_names[cpd.variable] for cpd in model_bn.get_cpds()}

    st.success("✅ Semua file berhasil dimuat!")

    st.markdown("### Input Evidence Awal")

    # Input evidence awal dari user
    evidence_input = {}
    for col in feature_columns:
        options = state_names.get(col, [0, 1])
        evidence_input[col] = st.selectbox(
            f"{col}", 
            options, 
            index=0, 
            format_func=lambda x: f"Status {x}"
        )

    st.markdown("### What-If Analysis")

    # Pilih variabel dan nilai baru
    var_to_change = st.selectbox("Variabel yang ingin diubah", feature_columns)
    new_val_options = state_names[var_to_change]
    new_val = st.selectbox(f"Ubah nilai '{var_to_change}' menjadi:", new_val_options)

    target_variable = st.selectbox("Target yang dianalisis", ['GPA_Disc', 'GradeClass'])

    if st.button("🔍 Jalankan What-If Analysis"):
        try:
            result = what_if_analysis(
                inference=inference_bn,
                evidence=evidence_input,
                target_variable=target_variable,
                var_to_change=var_to_change,
                new_value=new_val,
                model=model_bn
            )
            st.success(f"Hasil probabilitas {target_variable} jika `{var_to_change} = {new_val}`:")
            st.json({k: float(v) for k, v in zip(result.state_names[target_variable], result.values)})
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")



# if selected == "Clustering Mahasiswa":
#     st.write("### Clustering Mahasiswa")

#     data_clean_1 = pd.read_csv("./data/data_normalization.csv")
#     data_clean_2 = pd.read_csv("./data/data_normalization.csv")
#     features_bn_cls = ['Age', 'StudyTimeWeekly_Disc', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']

#     plt.figure(figsize=(8,6))
#     plt.scatter(data_clean_1['StudyTimeWeekly_Disc'], data_clean_1['Absences'], c=data_clean_1['Cluster'], cmap='viridis')
#     plt.xlabel('StudyTimeWeekly_Disc')
#     plt.ylabel('Absences')
#     plt.title('Clustering Mahasiswa Berdasarkan Profil Performa')
#     plt.grid(True)
#     st.pyplot(plt)
#     st.markdown("---")

#     st.write("Rata-rata tiap cluster berdasarkan fitur:")
#     st.write(data_clean_2.groupby('Cluster')[features_bn_cls].mean())


if selected == "Evaluasi Model":
    st.write("### Evaluasi Model")

    with st.sidebar:
        sub_selected = option_menu(
            menu_title="",
            options=["GPA (MAE, RMSE), GradeClass (Akurasi, Presisi, Recall), AUC",
                     "Kalibrasi Probabilistik"],
        )
    
    if sub_selected == "GPA (MAE, RMSE), GradeClass (Akurasi, Presisi, Recall), AUC":
        with open("./data/evaluasi_model_bn_wosh.json", "r") as f:
            em_wosh = json.load(f)
        
        st.markdown("---")
        st.write("#### Evaluasi Model untuk GPA (MAE, RMSE)")
        st.write(f"MAE: `{em_wosh["gpa_mae"] * 100:.2f}%`")
        st.write(f"RMSE: `{em_wosh["gpa_rmse"] * 100:.2f}%`")
        st.write("Nilai Mean Absolute Error (MAE) sebesar 0,39 menunjukkan bahwa rata-rata kesalahan prediksi terhadap GPA_Disc relatif rendah. Sementara itu, Root Mean Squared Error (RMSE) sebesar 0,82 mengindikasikan bahwa secara keseluruhan prediksi model cukup akurat, meskipun terdapat beberapa kesalahan prediksi yang lebih besar.")

        st.markdown("---")
        st.write("#### Evaluasi Model untuk GradeClass (Akurasi, Presisi, Recall, AUC)")
        st.write(f"Akurasi: `{em_wosh["gradeclass_akurasi"] * 100:.2f}%`")
        st.write(f"Presisi: `{em_wosh["gradeclass_presisi"] * 100:.2f}%`")
        st.write(f"Recall: `{em_wosh["gradeclass_recall"] * 100:.2f}%`")
        st.write("Berdasarkan hasil evaluasi, model berhasil memprediksi GradeClass dengan akurasi sebesar `70,98%`, yang berarti sekitar 71 dari 100 prediksi yang dilakukan oleh model sudah tepat. Presisi sebesar 70,10% menunjukkan bahwa dari semua prediksi yang diklasifikasikan ke dalam suatu kelas, sekitar 70% di antaranya benar, mencerminkan konsistensi model dalam memberikan label yang tepat. Recall sebesar `70,98%` mengindikasikan bahwa model mampu menangkap sebagian besar data yang relevan dengan cukup baik.")

        st.markdown("---")
        st.write("Area Under ROC Curve (AUC) untuk klasifikasi performa")
        st.write(f"GPA AUC: `{em_wosh["gpa_auc"] * 100:.2f}%`")
        st.write(f"GradeClass AUC: `{em_wosh["gradeclass_auc"] * 100:.2f}%`")
        st.write("Berdasarkan persentase tersebut, AUC GPA `89%` dan AUC GradeClass `90%` menandakan performa klasifikasi model cukup baik untuk GradeClass dan Sangat Baik untuk GPA dalam membedakan kelas.")


    if sub_selected == "Kalibrasi Probabilistik":
        st.markdown("---")
        st.write("### Kalibrasi Probabilistik GradeClass")

        # Load kembali
        prob_pred = joblib.load("./data/prob_pred_gradeclass.pkl")
        test_gradeclass = pd.read_csv("./data/test_gradeclass.csv").squeeze()  # Squeeze untuk Series
        lb = joblib.load("./data/label_binarizer_gradeclass.pkl")

        # Ubah label asli ke bentuk biner
        test_grade_class_bin = lb.transform(test_gradeclass)

        # --- Kalibrasi Multi-Kelas ---
        plt.figure(figsize=(10, 6))
        for i in range(prob_pred.shape[1]):
            prob_true, prob_pred_class = calibration_curve(test_grade_class_bin[:, i], prob_pred[:, i], n_bins=10)
            plt.plot(prob_pred_class, prob_true, marker='o', label=f'Class {i}', color=plt.cm.jet(i / prob_pred.shape[1]))

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Kalibrasi Probabilistik untuk GradeClass (Multi-Kelas)')
        plt.legend()
        st.pyplot(plt)

        # --- Kalibrasi Kelas Tertentu (misal kelas 1) ---
        y_true_bin = (test_gradeclass == 1).astype(int)
        prob_pred_class1 = prob_pred[:, 1]

        prob_true, prob_pred_class1_calib = calibration_curve(y_true_bin, prob_pred_class1, n_bins=10)

        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred_class1_calib, prob_true, marker='o', label='Model', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Kalibrasi Probabilistik untuk GradeClass (Kelas 1 vs Lainnya)')
        plt.legend()
        st.pyplot(plt)


        st.write("Grafik-grafik kurva kalibrasi probabilistik ini digunakan untuk model prediksi GradeClass, baik dalam skenario multi-kelas maupun biner. Pada grafik multi-kelas, masing-masing kelas dibandingkan dengan garis kalibrasi sempurna; semakin dekat ke garis diagonal, semakin baik model mengestimasi probabilitas kelas tersebut. Pada grafik biner (Kelas 1 vs Lainnya), model tampak sedikit under-confident pada probabilitas rendah dan over-confident pada probabilitas tinggi, menunjukkan kalibrasi yang kurang ideal terutama pada prediksi tinggi.")


        st.markdown("---")
        st.write("### Kalibrasi Probabilistik GPA")

        # Load kembali
        prob_pred_gpa = joblib.load("./data/prob_pred_gpa.pkl")
        test_gpa_disc = pd.read_csv("./data/test_gpa_disc.csv").squeeze()
        lb = joblib.load("./data/label_binarizer_gpa.pkl")

        # Binarisasi ulang label
        test_gpa_disc_bin = lb.transform(test_gpa_disc)

        # --- Kalibrasi Multi-Kelas ---
        plt.figure(figsize=(10, 6))

        for i in range(prob_pred_gpa.shape[1]):
            prob_true, prob_pred_class = calibration_curve(test_gpa_disc_bin[:, i], prob_pred_gpa[:, i], n_bins=10)
            plt.plot(prob_pred_class, prob_true, marker='o', label=f'Class {i}', color=plt.cm.jet(i / prob_pred_gpa.shape[1]))

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Kalibrasi Probabilistik untuk GPA_Disc (Multi-Kelas)')
        plt.legend()
        st.pyplot(plt)


        # --- Kalibrasi Kelas Tertentu (misal kelas 1) ---
        y_true_bin = (test_gpa_disc == 1).astype(int)
        prob_pred_class1_gpa = prob_pred_gpa[:, 1]

        prob_true, prob_pred_class1_gpa_calib = calibration_curve(y_true_bin, prob_pred_class1_gpa, n_bins=10)

        plt.figure(figsize=(10, 6))
        plt.plot(prob_pred_class1_gpa_calib, prob_true, marker='o', label='Model', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Kalibrasi Probabilistik untuk GPA_Disc (Kelas 1 vs Lainnya)')
        plt.legend()
        st.pyplot(plt)

        st.write("Grafik-grafik kurva kalibrasi probabilistik ini digunakan untuk model prediksi GPA_Disc, baik dalam skenario multi-kelas maupun biner. Pada kalibrasi multi-kelas, model menunjukkan variasi performa antar kelas, dengan beberapa kelas seperti Class 1 cukup terkalibrasi di probabilitas rendah-menengah, sementara kelas lain seperti Class 0 dan Class 4 mengalami over-confidence atau under-confidence di berbagai rentang. Pada kalibrasi biner (Class 1 vs Lainnya), model tampak cukup baik di rentang probabilitas 0.3–0.75, namun mengalami under-confidence pada prediksi rendah dan over-confidence signifikan pada prediksi tinggi.")


if selected == "Validasi Model":
    st.write("### Validasi Model")
    

    with st.sidebar:
        sub_selected = option_menu(
            menu_title="",
            options=["K-fold Cross-Validation",
                     "Sensitivity Analysis",
                    #  "Memberi Rekomendasi",
                     "Comparison Dengan Baseline Model"],
        )
    
    if sub_selected == "K-fold Cross-Validation":
        st.markdown("---")
        st.write("#### K-fold Cross-Validation")
        
        with open("./data/k_fold_bn_info.json", "r") as k_fold_bn:
            k_fold_bn = json.load(k_fold_bn)
    

        st.markdown("---")
        st.write("### Hasil evaluasi model menggunakan K-fold cross-validation (GPA):")
        st.write(f"Cross-validation MAE untuk GPA_Disc: `{k_fold_bn["mean_mae_gpa"]:.3f}` ± `{k_fold_bn["std_mae_gpa"]:.3f}`")
        st.write(f"Cross-validation RMSE untuk GPA_Disc: `{k_fold_bn["mean_rmse_gpa"]:.3f}` ± `{k_fold_bn["std_rmse_gpa"]:.3f}`")

        st.write("Hasil cross-validation menunjukkan bahwa model memiliki MAE rata-rata sebesar 0.035 dengan variasi ± 0.007, dan RMSE rata-rata sebesar 0.209 dengan variasi ± 0.032 untuk prediksi GPA_Disc. Nilai ini mengindikasikan bahwa kesalahan prediksi model relatif kecil dan stabil di berbagai lipatan data, dengan variasi yang juga cukup rendah. Hal ini menunjukkan bahwa model cukup andal dalam memberikan prediksi yang konsisten dengan kesalahan yang minim.")

        st.markdown("---")
        st.write("### Hasil evaluasi model menggunakan K-fold cross-validation (GradeClass):")
        st.write(f"Cross-validation MAE untuk GradeClass: `{k_fold_bn["mean_mae_grade_class"]:.3f}` ± `{k_fold_bn["std_mae_grade_class"]:.3f}`")
        st.write(f"Cross-validation RMSE untuk GradeClass: `{k_fold_bn["mean_rmse_grade_class"]:.3f}` ± `{k_fold_bn["std_rmse_grade_class"]:.3f}`")

        st.write("Hasil cross-validation menunjukkan MAE sebesar 0.000 dan RMSE sebesar 0.000 untuk prediksi GradeClass, yang berarti tidak ada kesalahan prediksi yang terdeteksi dalam lipatan data yang diuji.")


    if sub_selected == "Sensitivity Analysis":
        st.markdown("---")
        st.write("#### Sensitivity Analysis")

        # Baca hasil kembali
        loaded_sensitivity_df = pd.read_csv("./data/sensitivity_analysis_results_bn.csv")

        with st.expander("Code"):
            code_sa_bn = '''
            # Variabel untuk menyimpan hasil sensitivitas
            sensitivity_results = []

            # Sensitivity Analysis terhadap Parameter Model
            prior_types = ['BDeu', 'K2']
            equivalent_sample_sizes = [5, 10, 20]

            for prior in prior_types:
                for ess in equivalent_sample_sizes:
                    model_val = BayesianNetwork(edges_bn)
                    model_val.fit(train_data_bn, estimator=BayesianEstimator, prior_type=prior, equivalent_sample_size=ess)

                    inference = VariableElimination(model_val)

                    predictions_gpa_disc_sens = []
                    for i in range(len(test_data_bn)):
                        evidence = test_data_bn.iloc[i].drop(['GPA_Disc', 'GradeClass']).to_dict()
                        gpa_disc_prob = inference.query(variables=['GPA_Disc'], evidence=evidence)
                        predictions_gpa_disc_sens.append(gpa_disc_prob.values.argmax())

                    mae_sens = mean_absolute_error(true_gpa_disc, predictions_gpa_disc_sens)
                    print(f"MAE dengan prior_type={prior}, equivalent_sample_size={ess}: {mae_sens:.2f}")

                    # Simpan hasil untuk analisis sensitivitas
                    sensitivity_results.append((f'prior={prior}, ess={ess}', mae_sens))
            '''
            st.code(code_sa_bn, language='python')
            
        st.markdown("---")

        # Buat plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=loaded_sensitivity_df, x="Parameter", y="MAE", palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Sensitivity Analysis: MAE untuk Setiap Parameter")
        plt.ylabel("Mean Absolute Error")
        plt.xlabel("Parameter Setting")

        # Tambahkan label nilai di atas tiap bar
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=height + 0.01,
                s=f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        plt.tight_layout()
        st.pyplot(plt)


        st.write("Hasil menunjukkan bahwa perubahan prior_type (BDeu atau K2) dan nilai equivalent_sample_size (5, 10, 20) tidak mempengaruhi MAE, yang tetap stabil di angka 0.39. Hal ini menandakan bahwa model cukup robust terhadap variasi parameter ini, dengan tingkat kesalahan yang konsisten meskipun terjadi perubahan pada konfigurasi prior dan ukuran sampel.")


    # if sub_selected == "Memberi Rekomendasi":
    #     st.markdown("---")
    #     st.write("#### Memberi Rekomendasi")
    #     loaded_sensitivity_df = pd.read_csv("./data/sensitivity_analysis_results_bn.csv")

    #     with st.expander("Code"):
    #         code_m_r_bn = '''
    #         # Fungsi untuk memberikan rekomendasi berdasarkan hasil sensitivitas
    #         def give_recommendations(sensitivity_results):
    #             # Menentukan nilai MAE terbaik
    #             best_mae = min(sensitivity_results, key=lambda x: x[1])
    #             print(f"Rekomendasi: Menggunakan kombinasi {best_mae[0]} memberikan MAE terbaik dengan nilai {best_mae[1]:.2f}.")

    #         # Panggil fungsi untuk memberikan rekomendasi
    #         give_recommendations(sensitivity_results)
    #         '''
    #         st.code(code_m_r_bn, language='python')

    #     st.markdown("---")
    #     # Fungsi untuk memberikan rekomendasi berdasarkan hasil sensitivitas
    #     def give_recommendations(sensitivity_results):
    #         # Menentukan nilai MAE terbaik
    #         best_mae = min(sensitivity_results, key=lambda x: x[1])
    #         st.write(f"Rekomendasi: Menggunakan kombinasi `{best_mae[0]}` memberikan MAE terbaik dengan nilai `{best_mae[1]:.2f}`.")

    #     # Ubah DataFrame menjadi list of tuples
    #     sensitivity_results = list(loaded_sensitivity_df.itertuples(index=False, name=None))

    #     # Panggil fungsi rekomendasi
    #     give_recommendations(sensitivity_results)


    if sub_selected == "Comparison Dengan Baseline Model":
        st.markdown("---")
        st.write("#### Comparison Dengan Baseline Model")
        st.markdown("---")

        # Load data dari file JSON
        with open("./data/comparison_bn.json", "r") as f:
            load_comp_bn = json.load(f)

        with open("./data/evaluasi_model_bn_wosh.json", "r") as f:
            em_wosh = json.load(f)

        # --- 1. Perbandingan Regresi (MAE & RMSE) --- #
        st.subheader("📊 Perbandingan Model Regresi: GPA_Disc")

        # Data
        models_reg = ["Bayesian Network", "Linear Regression"]
        mae_scores = [em_wosh["gpa_mae"] * 100, load_comp_bn["lr_mae"] * 100]
        rmse_scores = [em_wosh["gpa_rmse"] * 100, load_comp_bn["lr_rmse"] * 100]

        # Plot MAE dan RMSE
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        bar_colors = ['#1f77b4', '#ff7f0e']

        # MAE
        ax[0].bar(models_reg, mae_scores, color=bar_colors)
        for i, val in enumerate(mae_scores):
            ax[0].text(i, val + 0.5, f'{val:.2f}%', ha='center')
        ax[0].set_title("MAE (%)")
        ax[0].set_ylim(0, max(mae_scores)*1.2)

        # RMSE
        ax[1].bar(models_reg, rmse_scores, color=bar_colors)
        for i, val in enumerate(rmse_scores):
            ax[1].text(i, val + 0.5, f'{val:.2f}%', ha='center')
        ax[1].set_title("RMSE (%)")
        ax[1].set_ylim(0, max(rmse_scores)*1.2)

        st.pyplot(fig)

        st.markdown("---")

        # --- 2. Perbandingan Klasifikasi (Akurasi & AUC) --- #
        st.subheader("📊 Perbandingan Model Klasifikasi: GradeClass")

        models_cls = ["Bayesian Network", "Random Forest"]
        accuracy_scores = [em_wosh["gradeclass_akurasi"] * 100 if "gradeclass_akurasi" in em_wosh else None,
                        load_comp_bn["rf_accuracy"] * 100]

        auc_scores = [em_wosh["gradeclass_auc"] * 100, load_comp_bn["rf_auc"] * 100]

        # Plot Akurasi dan AUC
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))

        # Accuracy
        ax2[0].bar(models_cls, accuracy_scores, color=bar_colors)
        for i, val in enumerate(accuracy_scores):
            ax2[0].text(i, val + 0.5, f'{val:.2f}%', ha='center')
        ax2[0].set_title("Akurasi (%)")
        ax2[0].set_ylim(0, max(accuracy_scores)*1.2)

        # AUC
        ax2[1].bar(models_cls, auc_scores, color=bar_colors)
        for i, val in enumerate(auc_scores):
            ax2[1].text(i, val + 0.5, f'{val:.2f}%', ha='center')
        ax2[1].set_title("AUC (%)")
        ax2[1].set_ylim(0, max(auc_scores)*1.2)

        st.pyplot(fig2)

        # Penjelasan Klasifikasi
