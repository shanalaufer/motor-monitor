# Motor Health Monitor
### AI-powered predictive maintenance system for electric motors

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motor-monitor-gtmdausuctqgflvd5xteux.streamlit.app/)

---

## What this is

An end-to-end system that detects motor bearing faults before they cause failure. It collects vibration data from an accelerometer, extracts signal features using FFT-based processing, and runs a machine learning model to classify motor health in real time — displayed on a live web dashboard.

This is the same concept used in industrial predictive maintenance systems at manufacturing plants, wind farms, and data centers — where unplanned motor failures can cost thousands of dollars per minute of downtime.

---

## Live demo

👉 [Open the live dashboard](YOUR_STREAMLIT_URL_HERE)

Use the sidebar to switch between healthy and faulty motor modes and watch the FFT, health score, and status banner update in real time.

---

## System architecture
```
Vibration sensor (MPU-6050)
        ↓
Edge node (ESP32) — samples at 1000Hz over I2C
        ↓
Feature extraction — FFT, RMS, crest factor, fault frequency energy
        ↓
ML model (Random Forest) — classifies healthy vs faulty
        ↓
REST API (FastAPI) — serves predictions
        ↓
Live dashboard (Streamlit) — health score, signal plots, alerts
```

---

## Key results

- 100% accuracy on test set (200 samples) — zero missed faults, zero false alarms
- Fault detection latency: < 2 seconds from signal to dashboard alert
- Feature importance: RMS vibration (40%), energy at fault frequency (37%)

---

## How it works

**Signal processing:**
Motor vibration is sampled at 1000Hz. The FFT decomposes the raw signal into frequency components — a healthy motor shows one dominant peak at its rotation frequency (50Hz). A bearing fault creates a characteristic peak at a calculable fault frequency (120Hz in this system). Five features are extracted from each signal window: RMS amplitude, peak value, crest factor, and energy at 50Hz and 120Hz.

**Machine learning:**
A Random Forest classifier (100 trees) was trained on 1000 labeled samples — 500 healthy, 500 faulty — with an 80/20 train/test split. The model outputs a fault probability which is converted to a 0-100% health score. The model was chosen for its interpretability and robustness on small tabular datasets.

**Why crest factor matters:**
Bearing faults create sharp impact spikes in the vibration signal. Crest factor (peak ÷ RMS) is sensitive to these spikes — a healthy motor has a crest factor around 1.4, while an impact-type fault can push it above 3.0. This is a standard diagnostic metric in industrial vibration analysis.

---

## Tech stack

| Layer | Technology |
|---|---|
| Signal processing | Python, NumPy, SciPy (FFT) |
| Machine learning | scikit-learn (Random Forest) |
| Edge firmware | MicroPython on ESP32 |
| Sensor | MPU-6050 accelerometer over I2C |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## Project structure
```
motor-monitor/
├── signal_sim.py          # Signal simulation and FFT visualization
├── generate_dataset.py    # Dataset generation (1000 labeled samples)
├── train_model.py         # Model training and evaluation
├── dashboard.py           # Live Streamlit dashboard
├── motor_dataset.csv      # Generated training dataset
├── motor_model.pkl        # Trained Random Forest model
└── requirements.txt       # Python dependencies
```

---

## What's next

- [ ] Custom PCB design in KiCad — replacing breadboard with designed hardware
- [ ] Deep learning upgrade — PyTorch autoencoder for unsupervised fault detection
- [ ] Multiple fault types — bearing defect, imbalance, misalignment
- [ ] FPGA acceleration — Verilog FFT implementation for microsecond latency
- [ ] LLM interface — natural language queries via Anthropic API

---

## Background

Built as a portfolio project to demonstrate the intersection of electrical engineering and machine learning. Predictive maintenance is one of the most widely deployed industrial AI applications — the same signal processing and anomaly detection concepts used here are applied at scale in manufacturing, energy, and aerospace.