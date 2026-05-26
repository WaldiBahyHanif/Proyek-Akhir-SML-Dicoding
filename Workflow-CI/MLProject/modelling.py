import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil
from sklearn.datasets import load_breast_cancer

def load_data():
    # Menghasilkan dataset dummy untuk keperluan run di CI
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_experiment("Breast_Cancer_CI_Pipeline")
    
    X_train, X_test, y_train, y_test = load_data()
    
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained with accuracy: {acc}")

if __name__ == "__main__":
    main()
