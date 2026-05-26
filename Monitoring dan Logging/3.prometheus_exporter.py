from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import time
import random

# Definisi Metrik (Minimal 10 untuk Advanced)
MODEL_ACCURACY = Gauge('model_accuracy', 'Akurasi model')
MODEL_PRECISION = Gauge('model_precision', 'Presisi model')
MODEL_RECALL = Gauge('model_recall', 'Recall model')
MODEL_F1 = Gauge('model_f1', 'F1-score model')

PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Waktu yang dibutuhkan untuk memproses prediksi')
PREDICTION_REQUESTS = Counter('prediction_requests_total', 'Total jumlah request prediksi')
INFERENCE_ERRORS = Counter('inference_errors_total', 'Total error saat inferensi')

MEMORY_USAGE = Gauge('memory_usage_percent', 'Persentase penggunaan memori (simulasi)')
CPU_USAGE = Gauge('cpu_usage_percent', 'Persentase penggunaan CPU (simulasi)')

MODEL_DRIFT_DETECTED = Gauge('model_drift_detected', 'Deteksi model drift (0 = Tidak, 1 = Ya)')
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Skor data drift')

ACTIVE_MODEL_VERSION = Info('active_model_version', 'Versi model yang sedang aktif')

def simulate_metrics():
    """Mengupdate metrik dengan nilai simulasi secara berkala."""
    ACTIVE_MODEL_VERSION.info({'version': '1.0.0', 'framework': 'scikit-learn'})
    
    while True:
        # Simulasi fluktuasi metrik
        acc = random.uniform(0.78, 0.95)
        MODEL_ACCURACY.set(acc)
        MODEL_PRECISION.set(random.uniform(0.80, 0.96))
        MODEL_RECALL.set(random.uniform(0.75, 0.94))
        MODEL_F1.set(random.uniform(0.77, 0.95))
        
        # Simulasi sistem metrics
        MEMORY_USAGE.set(random.uniform(40.0, 85.0))
        CPU_USAGE.set(random.uniform(20.0, 70.0))
        
        # Simulasi drift
        drift_prob = random.random()
        if drift_prob > 0.95:
            MODEL_DRIFT_DETECTED.set(1)
        else:
            MODEL_DRIFT_DETECTED.set(0)
            
        DATA_DRIFT_SCORE.set(random.uniform(0.01, 0.2))
        
        # Simulasi request
        requests_in_interval = random.randint(1, 10)
        PREDICTION_REQUESTS.inc(requests_in_interval)
        
        # Simulasi latency dan error
        for _ in range(requests_in_interval):
            latency = random.uniform(0.1, 2.5)  # bisa lebih dari 2s untuk trigger alert
            PREDICTION_LATENCY.observe(latency)
            
            # 5% chance of error
            if random.random() < 0.05:
                INFERENCE_ERRORS.inc()
                
        time.sleep(15) # update setiap 15 detik mengikuti scrape interval

if __name__ == '__main__':
    # Start server metrik di port 8000
    print("Memulai Prometheus Exporter di http://localhost:8000")
    start_http_server(8000)
    simulate_metrics()
