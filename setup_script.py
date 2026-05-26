import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
import json

# 1. Bikin folder data
os.makedirs("data", exist_ok=True)

# 2. Load dataset breast cancer dan simpan ke CSV
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv("data/dataset.csv", index=False)

# 3. Bikin struktur folder screenshot dll
folders_to_create = [
    "Membangun_model",
    "Membangun_model/heart_disease_preprocessing",
    "Workflow-CI",
    "Workflow-CI/.github/workflows",
    "Workflow-CI/MLProject/data_preprocessing",
    "Monitoring dan Logging/1.bukti_serving",
    "Monitoring dan Logging/4.bukti_monitoring_prometheus",
    "Monitoring dan Logging/5.bukti_monitoring_grafana",
    "Monitoring dan Logging/6.bukti_alerting_grafana"
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

# 4. Generate Eksperimen_SML_Waldi_Bahy_Hanif_Ramadhani.txt
with open("Eksperimen_SML_Waldi_Bahy_Hanif_Ramadhani.txt", "w") as f:
    f.write("Link Repository GitHub untuk Kriteria 1:\nhttps://github.com/USERNAME/REPO-KRITERIA-1\n")

with open("Workflow-CI.txt", "w") as f:
    f.write("Link Repository GitHub untuk Kriteria 3 (CI/CD):\nhttps://github.com/USERNAME/REPO-KRITERIA-3\n")

# 5. Generate eksperimen.ipynb (Notebook basic EDA)
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Eksperimen Data dan Preprocessing\n",
    "Notebook ini berisi tahapan Eksplorasi Data (EDA) dan Preprocessing untuk dataset Breast Cancer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set style\n",
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv('data/dataset.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cek Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Distribusi Target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.countplot(x='target', data=df)\n",
    "plt.title('Distribusi Kelas Target')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Menjalankan Pipeline Preprocessing dari preprocessing.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from preprocessing import run_preprocessing_pipeline\n",
    "run_preprocessing_pipeline()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("eksperimen.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Setup awal dataset, folder, dan notebook berhasil dibuat!")
