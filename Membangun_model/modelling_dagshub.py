import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# =========================================================
# TODO: Ganti USERNAME dan REPO_NAME dengan milik Anda
# =========================================================
DAGSHUB_USERNAME = "USERNAME_ANDA"
DAGSHUB_REPO = "NAMA_REPO_DAGSHUB_ANDA"

def init_dagshub():
    # Inisialisasi DagsHub tracking
    # Ini akan otomatis setup MLFLOW_TRACKING_URI ke DagsHub
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("Breast_Cancer_DagsHub_Experiment")

def load_data():
    df = pd.read_csv('../data/dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_advanced_model():
    # Setup DagsHub (Hapus komentar di bawah ini jika ingin push beneran ke Dagshub)
    # init_dagshub()
    
    # Untuk testing lokal sementara:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Breast_Cancer_Advanced")
    
    X_train, X_test, y_train, y_test = load_data()
    mlflow.sklearn.autolog(disable=True)
    
    with mlflow.start_run(run_name="RandomForest_DagsHub_Log"):
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # 1. Menghitung Metrik Wajib
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 2. Confusion Matrix & Classification Report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        print(cr)
        
        # Logging Parameters & Metrics
        mlflow.log_params({"n_estimators": 100, "max_depth": 10})
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # 3. Artefak 1: Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_plot_path = "confusion_matrix.png"
        plt.savefig(cm_plot_path)
        plt.close()
        
        mlflow.log_artifact(cm_plot_path, artifact_path="evaluation_plots")
        
        # 4. Artefak 2: Classification Report Text
        cr_path = "classification_report.txt"
        with open(cr_path, "w") as f:
            f.write(cr)
            
        mlflow.log_artifact(cr_path, artifact_path="evaluation_texts")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("Berhasil melog metrik, parameter, dan artefak tambahan ke MLflow/DagsHub.")

if __name__ == "__main__":
    train_advanced_model()
