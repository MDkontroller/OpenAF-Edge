# 🚀 OpenAF-Edge: A Trustworthy On-Device AF Monitoring Prototype

**OpenAF-Edge** is a rapid prototyping project focused on **real-time atrial fibrillation (AF) detection** using lightweight deep learning models deployed on edge devices.

This repository demonstrates how clinically relevant cardiac anomaly detection can be:
- 💡 Accurate enough for triage-level alerts
- 🔒 Privacy-preserving (runs fully offline)
- ⚡ Efficient for wearable or mobile deployment
- 🔍 Transparent with interpretable outputs

> Built in under 3 days to showcase AI readiness for medical-grade monitoring — and support innovation calls like **ENFIELD's IDH.1** or MDR pre-certification pilots.

---

## 🫀 What It Does

- 📥 Ingests raw ECG signals (MIT-BIH AFDB) (https://physionet.org/content/afdb/1.0.0/)
- 🧠 Classifies 3 rhythms: **Atrial Fibrillation (AF)**, **Normal Sinus Rhythm (NSR)**, and **Other**
- 🧮 Runs inference using a quantized or compact CNN/xLSTM/Tranformer
- 💻 Executes on Raspberry Pi pico (or emulated via TFLite/ONNX on desktop)
- 📈 Visualizes results with real-time waveform + rhythm class (if possible)

---

## 📊 📊 Prototype Specs (Planned Benchmark Goals)

| Metric        | Value               |
|---------------|---------------------|
| Accuracy      | ~97.5% (AFDB, inter-patient) |
| Latency       | < 1s (on Pi pico 4)       |
| Model size    | < 15 KB (quantized CNN or hybrid CNN/xLSTM) |
| Power usage   | < 1uW       |

---

## 📂 Folder Structure


📦 OpenAF-Edge
 ┣ 📁 data/               # Preprocessed ECG samples
 ┣ 📁 edge/               # Raspberry Pi-pico / ARM64 deployment scripts
 ┣ 📁 models/             # 
 ┣ 📁 notebooks/          # Exploratory tests & signal diagnostics
 ┣ 📁 scripts/            # Utility scripts
 ┣ 📁 src/                # Core pipeline: preprocessing, inference, visualization
 ┣ 📄 .gitignore          # Git ignore rules
 ┗ 📄 README.md           # Project documentation

```bash

python3.10 -m venv .venv
source .venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

```

to download the file run

```bash
python /scripts/download_afdb.py
```

first data visualization

```bash
notebooks/01_afdb_visualize.ipynb
```