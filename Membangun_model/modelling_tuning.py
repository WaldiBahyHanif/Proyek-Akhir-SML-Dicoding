import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Konfigurasi MLflow ke lokal
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Breast_Cancer_Tuning_Experiment")

def load_data():
    df = pd.read_csv('../data/dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_tuning_model():
    X_train, X_test, y_train, y_test = load_data()
    
    # Menonaktifkan autolog karena kita akan manual log sesuai Kriteria 2
    mlflow.sklearn.autolog(disable=True)
    
    print("Memulai Hyperparameter Tuning dengan GridSearchCV...")
    
    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best Parameters: {best_params}")
    
    # Memulai MLflow run
    with mlflow.start_run(run_name="RandomForest_Tuned_ManualLog"):
        y_pred = best_model.predict(X_test)
        
        # Hitung metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Metrics -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Logging parameter terbaik
        mlflow.log_params(best_params)
        
        # Logging metrik
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # Logging model
        mlflow.sklearn.log_model(best_model, "tuned_rf_model")
        
        print("Model dan hasil tuning berhasil di-log ke MLflow!")

if __name__ == "__main__":
    train_tuning_model()
