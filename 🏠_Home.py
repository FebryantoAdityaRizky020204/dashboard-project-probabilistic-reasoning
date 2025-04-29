import streamlit as st
import pandas as pd
import base64

from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="HOME",
    page_icon="🏠",
    layout="wide"
)

with st.sidebar:
    selected = option_menu(
        menu_title="HOME",
        options=["TEAMS","ABOUT PROJECT"],
    )


if selected == "TEAMS":
    st.title("KELOMPOK 6")

    namaKelompok = pd.DataFrame({
        "Nama" : ["Yunisa Nur Safa", "Willy Azrieel", "Aditya Rizky Febryanto", "Novita Maria", "Milky Gratia Br Sitorus", "Melda Nia Yuliani", "Dectrixie Theodore Mayumi S."],
        "NIM" : ["223020503078", "223020503101", "223020503108", "223020503109", "223020503116", "223020503119", "223020503140"]
    })

    # Reset index dan mulai dari 1
    namaKelompok = namaKelompok.reset_index(drop=True)
    namaKelompok.index = namaKelompok.index + 1

    st.table(namaKelompok)

if selected == "ABOUT PROJECT":
    st.title("Project Probabilistic Reasoning (Penalaran Probabilistik): Prediksi Performa Mahasiswa Berbasis Bayesian Network")
    pdf_path = "./data/Project 3.pdf"
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
