import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Prediksi Risiko Penyakit Jantung", layout="wide")

st.sidebar.title("Navigasi")
page = st.sidebar.radio("", ["Beranda", "Persiapan Data", "Evaluasi Model", "Prediksi"])

@st.cache_data
def load_data():
    try:
        return pd.read_csv("heart.csv")
    except FileNotFoundError:
        return None

df = load_data()

@st.cache_resource
def prepare_models_and_scaler(df_raw):
    df_clean = df_raw.drop_duplicates()
    
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    return scaler, dt_model, knn_model, X_train_scaled, X_test_scaled, y_train, y_test

if df is None:
    st.warning("File 'heart.csv' tidak ditemukan.")
else:
    if page == "Beranda":
        st.header("Prediksi Risiko Penyakit Jantung")
        
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            st.metric("Total Baris", df.shape[0])
        with col2:
            st.metric("Total Kolom", df.shape[1])
            
        st.dataframe(df, use_container_width=True)

    elif page == "Persiapan Data":
        st.header("Persiapan Data")
        
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum Pembersihan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Baris", df.shape[0])
            c2.metric("Missing Values", missing_values)
            c3.metric("Duplikat", duplicates)
            
        df_clean = df.drop_duplicates()
        
        with col2:
            st.subheader("Setelah Pembersihan")
            st.metric("Total Baris Akhir", df_clean.shape[0])
            
        st.markdown("---")
        st.subheader("Pembagian Data (Train/Test 80:20)")
        
        X = df_clean.drop('target', axis=1)
        y = df_clean['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        col3, col4, _ = st.columns([1, 1, 2])
        with col3:
            st.metric("Data Latih (Train)", X_train.shape[0])
        with col4:
            st.metric("Data Uji (Test)", X_test.shape[0])
            
        # Standard Scaler dijalankan di backend tanpa penjelasan teks

    elif page == "Evaluasi Model":
        st.header("Evaluasi Model")
        
        scaler, dt_model, knn_model, X_train_scaled, X_test_scaled, y_train, y_test = prepare_models_and_scaler(df)
        
        y_pred_dt = dt_model.predict(X_test_scaled)
        y_pred_knn = knn_model.predict(X_test_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Decision Tree")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Akurasi", f"{accuracy_score(y_test, y_pred_dt):.4f}")
            c2.metric("Presisi", f"{precision_score(y_test, y_pred_dt):.4f}")
            c3.metric("Recall", f"{recall_score(y_test, y_pred_dt):.4f}")
            c4.metric("F1-Score", f"{f1_score(y_test, y_pred_dt):.4f}")
            
        with col2:
            st.subheader("K-Nearest Neighbors")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Akurasi", f"{accuracy_score(y_test, y_pred_knn):.4f}")
            k2.metric("Presisi", f"{precision_score(y_test, y_pred_knn):.4f}")
            k3.metric("Recall", f"{recall_score(y_test, y_pred_knn):.4f}")
            k4.metric("F1-Score", f"{f1_score(y_test, y_pred_knn):.4f}")
            
        st.markdown("---")
        st.subheader("Confusion Matrix")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
        ax[0].set_title("Decision Tree")
        ax[0].set_xlabel("Prediksi")
        ax[0].set_ylabel("Aktual")
        
        sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
        ax[1].set_title("K-Nearest Neighbors")
        ax[1].set_xlabel("Prediksi")
        ax[1].set_ylabel("Aktual")
        
        st.pyplot(fig)
        
        st.markdown("---")
        dt_acc = accuracy_score(y_test, y_pred_dt)
        knn_acc = accuracy_score(y_test, y_pred_knn)
        best_model = "Decision Tree" if dt_acc > knn_acc else "K-Nearest Neighbors"
        if dt_acc == knn_acc:
            best_model = "Decision Tree & K-Nearest Neighbors"
            
        st.markdown(f"**Model Terbaik: {best_model}**")

    elif page == "Prediksi":
        st.header("Prediksi")
        
        scaler, dt_model, knn_model, _, X_test_scaled, _, y_test = prepare_models_and_scaler(df)
        
        dt_acc = accuracy_score(y_test, dt_model.predict(X_test_scaled))
        knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))
        best_model = dt_model if dt_acc > knn_acc else knn_model
        
        with st.form("form_prediksi"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Umur", min_value=1, max_value=120, value=50)
                sex = st.selectbox("Jenis Kelamin", ["Wanita", "Pria"])
                cp = st.selectbox("Tipe Nyeri Dada", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
                trestbps = st.number_input("Tekanan Darah (mm Hg)", min_value=50, max_value=250, value=120)
                chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
                
            with col2:
                fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak", "Ya"])
                restecg = st.selectbox("Hasil ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
                thalach = st.number_input("Detak Jantung Maksimal", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Angina Akibat Olahraga", ["Tidak", "Ya"])
                oldpeak = st.number_input("Depresi ST", min_value=0.0, max_value=10.0, value=1.0)
                
            with col3:
                slope = st.selectbox("Kemiringan Segmen ST", ["Upsloping", "Flat", "Downsloping"])
                ca = st.number_input("Jumlah Pembuluh Darah Utama", min_value=0, max_value=4, value=0)
                thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
                st.markdown("<br><br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Prediksi", use_container_width=True)
                
        if submit:
            sex_val = 1 if sex == "Pria" else 0
            cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp_val = cp_dict[cp]
            fbs_val = 1 if fbs == "Ya" else 0
            restecg_dict = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg_val = restecg_dict[restecg]
            exang_val = 1 if exang == "Ya" else 0
            slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_val = slope_dict[slope]
            thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
            thal_val = thal_dict[thal]
            
            input_data = pd.DataFrame([[
                age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                thalach, exang_val, oldpeak, slope_val, ca, thal_val
            ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            input_scaled = scaler.transform(input_data)
            prediction = best_model.predict(input_scaled)[0]
            
            st.markdown("---")
            if prediction == 0:
                st.success("Tidak Berisiko")
            else:
                st.error("Berisiko")
