import requests
import json
import time
import random
import pandas as pd

# Targetkan port 5001 yang di-serve oleh MLflow
# Command untuk run MLflow serve: mlflow models serve -m "models:/NAMA_MODEL/Staging" -p 5001
SCORING_URI = 'http://localhost:5001/invocations'

def simulate_inference():
    print(f"Memulai simulasi inference ke {SCORING_URI}...")
    
    # Load sedikit data untuk disimulasikan (hanya ambil 5 baris pertama, drop target)
    df = pd.read_csv('../data/dataset.csv').drop('target', axis=1).head(5)
    
    # Format data sesuai standar MLflow model serving (Split-oriented)
    # MLflow 2.0+ merekomendasikan dataframe_split format
    payload = {
        "dataframe_split": df.to_dict(orient="split")
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    while True:
        try:
            start_time = time.time()
            response = requests.post(SCORING_URI, json=payload, headers=headers)
            
            latency = time.time() - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Request latency: {latency:.3f}s | Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"Prediksi: {response.json()}")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Gagal connect. Pastikan model sudah di-serve di port 5001!")
        
        # Jeda acak antara 2 hingga 10 detik
        time.sleep(random.uniform(2, 10))

if __name__ == "__main__":
    simulate_inference()
