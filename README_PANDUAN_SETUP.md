# Panduan Setup & Run Proyek Akhir Machine Learning Dicoding

Panduan ini berisi tahapan lengkap untuk me-run semua skrip yang telah dibuat dan mempersiapkan _submission_ tugas akhir (Advanced/5 Bintang).

## Prasyarat
- Python 3.12.7 (Direkomendasikan menggunakan Conda atau venv)
- MLflow 2.19.0
- Docker (Untuk Kriteria 3 & 4)
- Akun GitHub (Untuk repo Kriteria 1 dan 3)
- Akun DagsHub (Untuk tracking Kriteria 2)
- Akun Docker Hub (Untuk publish image Kriteria 3)

---

## Tahap 1: Setup Environment
Buka terminal/command prompt, dan jalankan perintah berikut:

```bash
# 1. Buat virtual environment (opsional tapi sangat disarankan)
python -m venv venv
# Aktifkan venv (Windows)
.\venv\Scripts\activate

# 2. Install dependencies
cd Membangun_model
pip install -r requirements.txt
cd ..
```

---

## Tahap 2: Kriteria 1 (Data & Preprocessing)

1. Jalankan pipeline preprocessing:
   ```bash
   python preprocessing.py
   ```
   *Skrip ini akan mengisi nilai kosong, melakukan Train-Test Split, fitur scaling, dan menyimpan datanya ke folder `data/` beserta scalernya.*
2. (Opsional) Buka `eksperimen.ipynb` di Jupyter Notebook atau VS Code untuk melihat tahapan EDA.
3. Buat repositori **PUBLIC** di GitHub (misal: `SMSML-Kriteria1`). Push folder `data/`, file `preprocessing.py`, dan `eksperimen.ipynb` ke repo tersebut.
4. Copy link repositori tersebut dan tempelkan ke dalam file `Eksperimen_SML_Waldi_Bahy_Hanif_Ramadhani.txt`.

---

## Tahap 3: Kriteria 2 (Membangun Model)

1. Buka terminal baru, jalankan Tracking Server MLflow lokal:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
2. Buka terminal lain, masuk ke folder `Membangun_model`:
   ```bash
   cd Membangun_model
   
   # Run model basic (Autolog)
   python modelling.py
   
   # Run model skilled (Hyperparameter tuning + manual log)
   python modelling_tuning.py
   ```
3. Buka browser dan akses `http://localhost:5000`. 
   - **TUGAS ANDA:** Ambil screenshot halaman utama Dashboard MLflow UI dan simpan sebagai `Membangun_model/screenshot_dashboard.jpg`.
   - **TUGAS ANDA:** Buka salah satu Run, klik tab "Artifacts" dan ambil screenshot sebagai `Membangun_model/screenshot_artifak.jpg`.

**Untuk Advanced (DagsHub):**
1. Buat repository di [DagsHub](https://dagshub.com).
2. Edit file `modelling_dagshub.py` baris 13 & 14, masukkan Username dan Nama Repository DagsHub Anda. Hapus komentar `# init_dagshub()` pada baris 26.
3. Jalankan: `python modelling_dagshub.py`. Skrip ini akan melog 2 artifak tambahan (Confusion Matrix plot & Classification report txt).
4. Copy link repo DagsHub Anda ke dalam file `DagsHub.txt`.

---

## Tahap 4: Kriteria 3 (CI/CD)

1. Buat repository GitHub **PUBLIC** baru (misal: `SMSML-Kriteria3`).
2. Masukkan *Secrets* di setting repository GitHub:
   - `DOCKER_USERNAME`: Username Docker Hub Anda
   - `DOCKER_PASSWORD`: Password/Access Token Docker Hub Anda
3. Push isi folder `Workflow-CI` (termasuk folder `.github`) ke repositori GitHub tersebut.
   *(Proses push ini akan men-trigger GitHub Actions untuk men-train model via MLflow Project, lalu membuild dan men-push Docker Image-nya ke Docker Hub Anda).*
4. Copy link repository GitHub ini dan tempelkan ke `Workflow-CI.txt`.

---

## Tahap 5: Kriteria 4 (Monitoring & Logging)

Anda perlu menggunakan Grafana dan Prometheus untuk bagian ini.
Langkah-langkah:
1. Menjalankan Model Serving:
   ```bash
   mlflow models serve -m "models:/NAMA_MODEL_YANG_TERDAFTAR/Production" -p 5001
   ```
   Atau jika Anda belum meregister model di MLflow registry, Anda bisa load langsung dari run ID (Cek di MLflow UI localhost:5000):
   ```bash
   mlflow models serve -m "runs:/ID_RUN_ANDA/model" -p 5001
   ```
   **TUGAS ANDA:** Ambil screenshot Command Prompt/Terminal yang menunjukkan model sedang diserve (Listening at: http://127.0.0.1:5001). Simpan di folder `Monitoring dan Logging/1.bukti_serving/`.

2. Menjalankan Prometheus Exporter:
   Buka terminal baru, jalankan:
   ```bash
   cd "Monitoring dan Logging"
   python 3.prometheus_exporter.py
   ```
   *(Jangan di-close, biarkan metrics terekspos di port 8000).*

3. Menjalankan Prometheus & Grafana (Gunakan Docker):
   ```bash
   # Jalankan Prometheus (Mount file prometheus.yml yang sudah kita siapkan)
   docker run -d --name prometheus -p 9090:9090 -v "%cd%\Monitoring dan Logging\2.prometheus.yml":/etc/prometheus/prometheus.yml prom/prometheus
   
   # Jalankan Grafana
   docker run -d --name grafana -p 3000:3000 grafana/grafana
   ```
   *(Catatan: %cd% pada Windows CMD, gunakan ${PWD} untuk PowerShell/Linux).*

4. Akses `http://localhost:9090` (Prometheus).
   Cek menu **Status > Targets**. Pastikan `prometheus-exporter` berstatus UP.
   **TUGAS ANDA:** Ambil screenshot target yang UP, dan screenshot graph metrics, simpan ke `Monitoring dan Logging/4.bukti_monitoring_prometheus/`.

5. Akses `http://localhost:3000` (Grafana). Login dengan `admin` / `admin`.
   - Add Data Source: Pilih Prometheus. Set URL: `http://host.docker.internal:9090`. Save & Test.
   - Buat Dashboard baru. Judul Dashboard: **Waldi Bahy Hanif Ramadhani - ML Monitoring**.
   - Tambahkan minimal 10 Panel (sesuai metric dari file `3.prometheus_exporter.py`: `model_accuracy`, `prediction_latency_seconds`, dll).
   - Buat 3 Alert Rules (Accuracy < 0.8, Latency > 2s, Memory > 80%).
   **TUGAS ANDA:** Ambil screenshot lengkap dashboard Grafana dan screenshot notifikasi alerting (Slack/Email), simpan di folder `5.bukti_monitoring_grafana/` dan `6.bukti_alerting_grafana/`.

6. Menyimulasikan trafik Inference:
   Agar grafik Grafana/Prometheus berubah (dinamis), jalankan script inference:
   ```bash
   cd "Monitoring dan Logging"
   python 7.inference.py
   ```

---

## Tahap Akhir (Packaging Zip)
Setelah semua kode berjalan, dan **SEMBILAN BELAS (19+) SCREENSHOT** lengkap terisi di dalam folder masing-masing, Anda bisa *compress* seluruh folder proyek ini (kecuali folder `venv`, `__pycache__`, `.git`, atau `mlruns`) menjadi satu file ZIP:
**SMSML_Waldi_Bahy_Hanif_Ramadhani.zip**

Submit file ZIP ini ke platform Dicoding. Semoga sukses mendapatkan 5 Bintang!

## Troubleshooting Umum
- **"Port 5000 is already in use"**: Pastikan Anda mematikan proses MLflow server sebelumnya (Ctrl+C).
- **"host.docker.internal tidak ditemukan"**: Ini biasa terjadi di Linux. Pada config `2.prometheus.yml`, ganti `host.docker.internal` menjadi IP lokal jaringan Anda (misal `192.168.1.5`).
- **"requests.exceptions.ConnectionError saat inference"**: Pastikan Anda sudah menjalankan perintah `mlflow models serve` di terminal lain.
