import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# URL ke tracking server lokal MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Breast_Cancer_Basic_Experiment")

def load_data():
    # Anggap dataset.csv sudah berada di folder proyek (dari Kriteria 1)
    df = pd.read_csv('../data/dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_basic_model():
    X_train, X_test, y_train, y_test = load_data()
    
    # Aktifkan MLflow autologging untuk scikit-learn
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Basic_RandomForest_Autolog"):
        print("Training model dengan MLflow Autolog...")
        # Definisikan model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        # Fit model (autolog akan otomatis mencatat parameter, metrik training, dan artefak model)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi model: {acc:.4f}")
        
        print("Training selesai, artefak dan log telah disimpan ke MLflow.")

if __name__ == "__main__":
    train_basic_model()
