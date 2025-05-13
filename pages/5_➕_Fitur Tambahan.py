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
    page_title="Fitur Tambahan",
    page_icon="➕",
    layout="wide"
)

with st.sidebar:
    selected = option_menu(
        menu_title="MODEL",
        options=["Analisis Intervensi (what-if analysis)",
                 "Structure Learning",]
    )

if selected == "Analisis Intervensi (what-if analysis)":
    st.write("### Analisis Intervensi (what-if analysis)")
    st.write("Pada bagian ini, dilakukan analisis what-if untuk mengeksplorasi pengaruh waktu belajar dan dukungan orang tua terhadap prediksi GPA mahasiswa. Analisis ini dilakukan dengan menggunakan pendekatan inferensi pada model Bayesian Network yang telah dilatih sebelumnya. Tujuan dari analisis ini adalah untuk memahami dampak dari variasi faktor-faktor tersebut terhadap nilai GPA yang diprediksi.")

    # === 3. Visualisasi What-if ===
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    labels = ['GPA_Disc(0)', 'GPA_Disc(1)', 'GPA_Disc(2)', 'GPA_Disc(3)', 'GPA_Disc(4)']
    distrib_awal = [0.0336, 0.0259, 0.2159, 0.1552, 0.5694]
    distrib_whatif = [0.1019, 0.1487, 0.1498, 0.2321, 0.3674]

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].bar(labels, distrib_awal, color='orange')
    ax[0].set_title("StudyTime: Low + Support: High")
    ax[0].set_ylim(0, 1)

    ax[1].bar(labels, distrib_whatif, color='green')
    ax[1].set_title("StudyTime: High + Support: High")
    ax[1].set_ylim(0, 1)

    fig.suptitle("Perbandingan Prediksi GPA (What-if Analysis)", fontsize=14)
    fig.tight_layout()

    # Tampilkan plot di Streamlit
    st.pyplot(fig)

    st.markdown("""
        Dalam simulasi ini, dua kondisi yang berbeda diuji, yaitu: pertama, kondisi dengan waktu belajar rendah dan dukungan orang tua tinggi, dan kedua, kondisi dengan waktu belajar tinggi dan dukungan orang tua tetap tinggi.

        1. Kondisi Awal (`StudyTime` = Low, `ParentalSupport` = High):
            - Hasil Distribusi GPA:
                - `GPA_Disc(0)` = `3.36%`
                - `GPA_Disc(1)` = `2.59%`
                - `GPA_Disc(2)` = `21.59%`
                - `GPA_Disc(3)` = `15.52%`
                - `GPA_Disc(4)` = `56.94%`
            - Interpretasi:
                - Dalam kondisi ini, meskipun waktu belajar rendah, dukungan orang tua tinggi memberikan probabilitas terbesar pada GPA yang lebih tinggi `(GPA_Disc 4)`. Dukungan orang tua berperan besar dalam membantu mahasiswa mencapai performa yang baik meskipun waktu belajar terbatas.
        
                
        2. Kondisi What-if (`StudyTime` = High, `ParentalSupport` = High):
            - Hasil Distribusi GPA:
                - `GPA_Disc(0)` = `10.19%`
                - `GPA_Disc(1)` = `14.87%`
                - `GPA_Disc(2)` = `14.98%`
                - `GPA_Disc(3)` = `23.21%`
                - `GPA_Disc(4)` = `36.74%`
            - Interpretasi:
                - Pada kondisi ini, meskipun waktu belajar tinggi dan dukungan orang tua tetap tinggi, distribusi nilai GPA menjadi lebih merata. Meskipun kategori GPA_Disc(4) masih memiliki probabilitas tertinggi, nilainya lebih rendah dibandingkan kondisi pertama. Ini menunjukkan bahwa faktor lain, seperti efisiensi waktu belajar, dapat mempengaruhi hasil akhir meskipun waktu belajar lebih banyak.
        
                
        Kesimpulan:
        - Dukungan orang tua yang tinggi berpengaruh besar terhadap performa GPA, bahkan ketika waktu belajar terbatas. Dalam kondisi waktu belajar rendah, dukungan orang tua dapat memberikan dorongan yang signifikan terhadap hasil yang lebih baik.
        - Dengan waktu belajar yang lebih tinggi, distribusi nilai GPA menjadi lebih bervariasi, menunjukkan bahwa kualitas waktu belajar dan faktor lainnya juga perlu diperhatikan agar hasilnya optimal.
    """)

if selected == "Structure Learning":
    st.write("### Structure Learning")
    st.write("Kode ini menggunakan algoritma Hill Climbing dan skor BIC untuk secara otomatis mempelajari struktur terbaik dari sebuah Bayesian Network berdasarkan data (data_normalization). Hasil struktur ini kemudian divisualisasikan sebagai grafik berarah menggunakan library networkx dan matplotlib untuk menunjukkan hubungan ketergantungan antar variabel.")

    # Edges dari model Bayesian Network
    edges_from_model = [('Absences', 'GPA_Disc'), ('GPA_Disc', 'GradeClass'), ('GPA_Disc', 'Tutoring')]

    # Buat directed graph dari edges
    G = nx.DiGraph(edges_from_model)

    # Visualisasi
    fig, ax = plt.subplots(figsize=(5, 3))
    pos = nx.spring_layout(G, seed=42)  # seed agar posisi tetap

    nx.draw(
        G, pos, with_labels=True, node_size=1200, node_color='lightblue',
        font_size=4, font_weight='bold', arrowsize=11,
        connectionstyle='arc3,rad=0.1', ax=ax
    )

    plt.title("Struktur Bayesian Network Hasil Hill Climbing", fontsize=16)
    st.pyplot(fig)

    st.markdown("""
    Pada tahap ini, dilakukan proses pembelajaran struktur Bayesian Network menggunakan algoritma Hill Climbing yang dikombinasikan dengan metode penilaian BIC (Bayesian Information Criterion). Tujuan dari proses ini adalah untuk memperoleh struktur jaringan probabilistik terbaik yang merepresentasikan hubungan ketergantungan antar variabel berdasarkan data yang telah dinormalisasi.

    Hasil pembelajaran struktur menghasilkan jaringan dengan 4 node dan 3 edge, yaitu:
    - `Absences` → `GradeClass`
    - `GradeClass` → `GPA_Disc`
    - `GPA_Disc` → `Tutoring`
    
    Struktur ini menunjukkan adanya pengaruh langsung antara variabel-variabel tersebut. Secara khusus, variabel Absences (jumlah ketidakhadiran) berpengaruh terhadap GradeClass (kelas nilai akhir), yang selanjutnya memengaruhi GPA_Disc (diskretisasi GPA). Kemudian, GPA_Disc juga memengaruhi kemungkinan siswa mengikuti bimbingan belajar (Tutoring).
    """)