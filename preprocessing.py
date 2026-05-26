import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath='data/dataset.csv'):
    """Memuat dataset."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Membersihkan data dari missing values (jika ada)."""
    # Mengisi missing values dengan rata-rata untuk kolom numerik
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def preprocess_features(df):
    """Memisahkan fitur dan target, lalu melakukan split dan scaling."""
    # Anggap 'target' adalah nama kolom label (ubah sesuai dataset jika berbeda)
    if 'target' not in df.columns:
        # Jika menggunakan breast cancer bawaan sklearn, kolom target biasanya bernama 'target'
        pass
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert kembali ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir='data/'):
    """Menyimpan data yang sudah diproses dan scaler."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Simpan scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print("Data dan scaler berhasil disimpan di:", output_dir)

def run_preprocessing_pipeline():
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    print("Memulai pipeline preprocessing...")
    df = load_data()
    print(f"Data dimuat dengan shape: {df.shape}")
    
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_features(df_clean)
    
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
    print("Pipeline preprocessing selesai tanpa error.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
