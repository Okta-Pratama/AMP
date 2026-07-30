import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data():
    try:
        return pd.read_csv("heart.csv")
    except FileNotFoundError:
        print("Error: File 'heart.csv' tidak ditemukan.")
        return None
# ---
def main():
    print("="*50)
    print("PREDIKSI RISIKO PENYAKIT JANTUNG (CLI VERSION)")
    print("="*50)
    
    df = load_data()
    if df is None:
        return
        
    print(f"\n[INFO] Data berhasil dimuat: {df.shape[0]} Baris, {df.shape[1]} Kolom")
    
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"[INFO] Missing Values: {missing_values}")
    print(f"[INFO] Duplikat: {duplicates}")
    
    df_clean = df.drop_duplicates()
    print(f"[INFO] Data setelah pembersihan: {df_clean.shape[0]} Baris")
    
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Data Latih (Train): {X_train.shape[0]} baris")
    print(f"[INFO] Data Uji (Test): {X_test.shape[0]} baris")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    y_pred_dt = dt_model.predict(X_test_scaled)
    y_pred_knn = knn_model.predict(X_test_scaled)

    dt_acc = accuracy_score(y_test, y_pred_dt)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    
    print("\n" + "-"*50)
    print("EVALUASI MODEL")
    print("-"*50)
    
    print("1. Decision Tree")
    print(f"   Akurasi  : {dt_acc:.4f}")
    print(f"   Presisi  : {precision_score(y_test, y_pred_dt):.4f}")
    print(f"   Recall   : {recall_score(y_test, y_pred_dt):.4f}")
    print(f"   F1-Score : {f1_score(y_test, y_pred_dt):.4f}")
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print(f"   Confusion Matrix:")
    print(f"      [{cm_dt[0][0]:>3} {cm_dt[0][1]:>3}]")
    print(f"      [{cm_dt[1][0]:>3} {cm_dt[1][1]:>3}]")
    
    print("\n2. K-Nearest Neighbors")
    print(f"   Akurasi  : {knn_acc:.4f}")
    print(f"   Presisi  : {precision_score(y_test, y_pred_knn):.4f}")
    print(f"   Recall   : {recall_score(y_test, y_pred_knn):.4f}")
    print(f"   F1-Score : {f1_score(y_test, y_pred_knn):.4f}")
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print(f"   Confusion Matrix:")
    print(f"      [{cm_knn[0][0]:>3} {cm_knn[0][1]:>3}]")
    print(f"      [{cm_knn[1][0]:>3} {cm_knn[1][1]:>3}]")
    
    best_model = dt_model if dt_acc > knn_acc else knn_model
    best_model_name = "Decision Tree" if dt_acc > knn_acc else "K-Nearest Neighbors"
    if dt_acc == knn_acc:
        best_model_name = "Decision Tree & K-Nearest Neighbors"
# ---
    print("\n" + "-"*50)
    print(f"MODEL TERBAIK: {best_model_name}")
    print("-"*50)
    
    while True:
        print("\n\n--- PREDIKSI PASIEN BARU ---")
        pilihan = input("Apakah Anda ingin memprediksi data pasien baru? (y/n): ")
        if pilihan.lower() != 'y':
            print("Program selesai. Terima kasih.")
            break
            
        try:
            print("\nMasukkan data pasien:")
            age = float(input("1. Umur (contoh: 50): "))
            
            sex_str = input("2. Jenis Kelamin (Pria/Wanita): ")
            sex = 1 if sex_str.lower() == 'pria' else 0
            
            print("3. Tipe Nyeri Dada (Chest Pain):")
            print("   0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
            cp = int(input("   Pilih (0-3): "))
            
            trestbps = float(input("4. Tekanan Darah Istirahat (mm Hg, contoh: 120): "))
            chol = float(input("5. Kolesterol Serum (mg/dl, contoh: 200): "))
            
            fbs_str = input("6. Gula Darah Puasa > 120 mg/dl? (Ya/Tidak): ")
            fbs = 1 if fbs_str.lower() == 'ya' else 0
            
            print("7. Hasil ECG Istirahat:")
            print("   0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
            restecg = int(input("   Pilih (0-2): "))
            
            thalach = float(input("8. Detak Jantung Maksimal (BPM, contoh: 150): "))
            
            exang_str = input("9. Angina Akibat Olahraga? (Ya/Tidak): ")
            exang = 1 if exang_str.lower() == 'ya' else 0
            
            oldpeak = float(input("10. Depresi ST (contoh: 1.0): "))
            
            print("11. Kemiringan Segmen ST:")
            print("    0: Upsloping, 1: Flat, 2: Downsloping")
            slope = int(input("    Pilih (0-2): "))
            
            ca = int(input("12. Jumlah Pembuluh Darah Utama (0-4): "))
            
            print("13. Thalassemia:")
            print("    1: Normal, 2: Fixed Defect, 3: Reversable Defect")
            thal = int(input("    Pilih (1-3): "))
            
            input_data = pd.DataFrame([[
                age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal
            ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            input_scaled = scaler.transform(input_data)
            prediction = best_model.predict(input_scaled)[0]
            
            print("\n=> HASIL PREDIKSI:")
            if prediction == 0:
                print("   [AMAN] Pasien diprediksi TIDAK BERISIKO penyakit jantung.")
            else:
                print("   [PERINGATAN] Pasien diprediksi BERISIKO penyakit jantung!")
                
        except ValueError:
            print("\n[ERROR] Input tidak valid! Pastikan Anda memasukkan angka/format yang benar.")

if __name__ == "__main__":
    main()