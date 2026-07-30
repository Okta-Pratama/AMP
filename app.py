import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="AMP - Predictive Analytics", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS menggunakan variabel native Streamlit agar kompatibel dengan Dark/Light mode
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
<style>
    /* Styling Typography & Headers tanpa merusak Dark Mode */
    .title-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid var(--secondary-background-color);
        padding-bottom: 1rem;
    }
    .title-header i {
        color: var(--primary-color);
        font-size: 2.2rem;
    }
    .sub-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sub-title i {
        color: var(--primary-color);
    }
    
    /* Tombol yang lebih luwes */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar dengan Streamlit Option Menu
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px; padding-top: 10px;">
        <h1 style="color: var(--primary-color); font-size: 3rem; font-weight: 800; margin: 0; letter-spacing: 2px;">AMP</h1>
        <p style="font-size: 0.9rem; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7;">Predictive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = option_menu(
        menu_title=None,
        options=["Beranda", "Persiapan Data", "Evaluasi Model", "Prediksi Risiko"],
        icons=["house", "clipboard-data", "cpu", "heart-pulse"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "1.1rem"},
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin": "8px 0", 
                "font-weight": "500",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "font-weight": "600"
            },
        }
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 Beralih ke menu **Prediksi Risiko** untuk analisis parameter pasien.")

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
    st.error("File 'heart.csv' tidak ditemukan. Pastikan dataset berada di direktori yang sama.")
else:
    if page == "Beranda":
        st.markdown("""
        <div class="title-header">
            <i class="bi bi-grid-1x2"></i>
            Tinjauan Data (Dashboard)
        </div>
        """, unsafe_allow_html=True)
        st.write("Sistem prediksi risiko kardiovaskular menggunakan pendekatan *Machine Learning*. Platform ini dirancang secara minimalis untuk memberikan analisis prediktif berbasis data klinis.")
        
        st.markdown('<div class="sub-title"><i class="bi bi-bar-chart"></i> Statistik Dataset Terkini</div>', unsafe_allow_html=True)
        
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Total Sampel", value=f"{df.shape[0]:,}")
            with col2:
                st.metric(label="Total Fitur Klinis", value=df.shape[1] - 1)
            with col3:
                st.metric(label="Kasus Berisiko", value=df[df['target'] == 1].shape[0])
            with col4:
                st.metric(label="Kasus Normal", value=df[df['target'] == 0].shape[0])
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Menghilangkan background gradient yang bisa bentrok dengan dark mode, gunakan default dataframe
        st.dataframe(df, use_container_width=True, height=400)

    elif page == "Persiapan Data":
        st.markdown("""
        <div class="title-header">
            <i class="bi bi-layers"></i>
            Pemrosesan Data
        </div>
        """, unsafe_allow_html=True)
        st.write("Tahapan pembersihan data (*Data Cleaning*) dan penyesuaian skala sebelum algoritma pelatihan berjalan.")
        
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        df_clean = df.drop_duplicates()
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('#### Sebelum Pembersihan')
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Baris", df.shape[0])
                c2.metric("Nilai Kosong", missing_values)
                c3.metric("Data Ganda", duplicates)
            
        with col2:
            with st.container(border=True):
                st.markdown('#### Setelah Pembersihan')
                c1, c2 = st.columns(2)
                c1.metric("Baris Bersih", df_clean.shape[0])
                c2.metric("Integritas Data", "100%")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sub-title"><i class="bi bi-pie-chart"></i> Proporsi Pembagian Data (80% Train : 20% Test)</div>', unsafe_allow_html=True)
        
        X = df_clean.drop('target', axis=1)
        y = df_clean['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with st.container(border=True):
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Data Latih (Train)", X_train.shape[0])
            with col4:
                st.metric("Data Uji (Test)", X_test.shape[0])

    elif page == "Evaluasi Model":
        st.markdown("""
        <div class="title-header">
            <i class="bi bi-cpu"></i>
            Evaluasi Performa Model
        </div>
        """, unsafe_allow_html=True)
        
        scaler, dt_model, knn_model, X_train_scaled, X_test_scaled, y_train, y_test = prepare_models_and_scaler(df)
        
        y_pred_dt = dt_model.predict(X_test_scaled)
        y_pred_knn = knn_model.predict(X_test_scaled)
        
        st.write("Komparasi metrik evaluasi antara algoritma Decision Tree dan K-Nearest Neighbors.")
        
        col1, col2 = st.columns(2)
        
        def display_metrics(y_true, y_pred, model_name):
            with st.container(border=True):
                st.markdown(f'#### {model_name}')
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c1.metric("Akurasi", f"{accuracy_score(y_true, y_pred):.4f}")
                c2.metric("Presisi", f"{precision_score(y_true, y_pred):.4f}")
                c3.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
                c4.metric("F1-Score", f"{f1_score(y_true, y_pred):.4f}")

        with col1:
            display_metrics(y_test, y_pred_dt, "Decision Tree")
            
        with col2:
            display_metrics(y_test, y_pred_knn, "K-Nearest Neighbors")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sub-title"><i class="bi bi-table"></i> Confusion Matrix</div>', unsafe_allow_html=True)
        
        with st.container(border=True):
            # Membuat confusion matrix dengan matplotlib standar agar sesuai dengan tema terang/gelap
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_alpha(0.0) # Transparan background
            
            sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
            ax[0].set_title("Decision Tree")
            ax[0].set_xlabel("Prediksi")
            ax[0].set_ylabel("Aktual")
            
            sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
            ax[1].set_title("K-Nearest Neighbors")
            ax[1].set_xlabel("Prediksi")
            ax[1].set_ylabel("Aktual")
            
            st.pyplot(fig)
        
        dt_acc = accuracy_score(y_test, y_pred_dt)
        knn_acc = accuracy_score(y_test, y_pred_knn)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if dt_acc > knn_acc:
            st.success(f"**Rekomendasi:** Model Decision Tree memiliki kinerja lebih optimal (Akurasi: {dt_acc:.2%})")
        else:
            st.success(f"**Rekomendasi:** Model K-Nearest Neighbors memiliki kinerja lebih optimal (Akurasi: {knn_acc:.2%})")

    elif page == "Prediksi Risiko":
        st.markdown("""
        <div class="title-header">
            <i class="bi bi-heart-pulse"></i>
            Prediksi Risiko Klinis
        </div>
        """, unsafe_allow_html=True)
        st.write("Masukkan parameter fisiologis pasien di bawah ini untuk mengevaluasi tingkat risiko kardiovaskular secara instan.")
        
        scaler, dt_model, knn_model, _, X_test_scaled, _, y_test = prepare_models_and_scaler(df)
        dt_acc = accuracy_score(y_test, dt_model.predict(X_test_scaled))
        knn_acc = accuracy_score(y_test, knn_model.predict(X_test_scaled))
        best_model = dt_model if dt_acc > knn_acc else knn_model
        
        with st.container(border=True):
            with st.form("form_prediksi", border=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Umur (Tahun)", min_value=1, max_value=120, value=50)
                    sex = st.selectbox("Jenis Kelamin", ["Wanita", "Pria"])
                    cp = st.selectbox("Tipe Nyeri Dada", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
                    trestbps = st.number_input("Tekanan Darah Sistolik (mm Hg)", min_value=50, max_value=250, value=120)
                    chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=200)
                    
                with col2:
                    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak", "Ya"])
                    restecg = st.selectbox("Hasil ECG Istirahat", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
                    thalach = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=150)
                    exang = st.selectbox("Angina Terinduksi Olahraga", ["Tidak", "Ya"])
                    oldpeak = st.number_input("Depresi ST", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                    
                with col3:
                    slope = st.selectbox("Kemiringan ST (Puncak Olahraga)", ["Upsloping", "Flat", "Downsloping"])
                    ca = st.number_input("Pembuluh Darah Utama (0-3)", min_value=0, max_value=4, value=0)
                    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    submit = st.form_submit_button("Jalankan Analisis", use_container_width=True)
                
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
            
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 0:
                st.success("### ✅ Risiko Rendah\n\nParameter indikator pasien saat ini berada dalam batas aman. Probabilitas kondisi kardiovaskular akut terdeteksi rendah.")
            else:
                st.error("### ⚠️ Peringatan Risiko Tinggi\n\nParameter mengindikasikan probabilitas tinggi terhadap risiko kardiovaskular. Diperlukan tindakan medis lanjutan.")
