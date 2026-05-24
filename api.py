from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import sqlite3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
rf_model = joblib.load('motor_model_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')
threshold = joblib.load('autoencoder_threshold_v2.pkl')

class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, bottleneck),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoder_v2.pth', weights_only=True))
autoencoder.eval()

# In-memory storage (unchanged)
latest_reading = {}
history = deque(maxlen=100)

# --- SQLite setup ---
DB_PATH = "motor_readings.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            rf_health REAL,
            rf_status TEXT,
            ae_health REAL,
            ae_status TEXT,
            recon_error REAL,
            rms REAL,
            crest_factor REAL,
            energy_100hz REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_reading(r: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO readings
        (timestamp, rf_health, rf_status, ae_health, ae_status,
         recon_error, rms, crest_factor, energy_100hz)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        r['timestamp'], r['rf_health'], r['rf_status'],
        r['ae_health'], r['ae_status'], r['recon_error'],
        r['rms'], r['crest_factor'], r['energy_100hz']
    ))
    conn.commit()
    conn.close()

init_db()

# --- Existing endpoints (unchanged) ---

@app.get("/")
def root():
    return {"message": "Motor Monitor API running"}

@app.get("/health")
def get_health():
    if not latest_reading:
        return {"status": "No data yet"}
    return latest_reading

@app.get("/history")
def get_history():
    return list(history)
@app.get("/db-check")
def db_check():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM readings")
    count = c.fetchone()[0]
    conn.close()
    return {"rows_in_db": count}
@app.get("/faults")
def get_faults():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT * FROM readings 
        WHERE rf_status = 'FAULT DETECTED' OR ae_status = 'FAULT DETECTED'
        ORDER BY timestamp DESC
    """)
    rows = c.fetchall()
    conn.close()
    columns = ['id', 'timestamp', 'rf_health', 'rf_status', 'ae_health',
               'ae_status', 'recon_error', 'rms', 'crest_factor', 'energy_100hz']
    return [dict(zip(columns, row)) for row in rows]

@app.get("/trends")
def get_trends(hours: int = 24):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT AVG(rf_health), AVG(ae_health), AVG(recon_error), COUNT(*)
        FROM readings
        WHERE timestamp >= datetime('now', '-' || ? || ' hours')
    """, (hours,))
    row = c.fetchone()
    conn.close()
    return {
        "hours": hours,
        "avg_rf_health": round(row[0], 1) if row[0] else None,
        "avg_ae_health": round(row[1], 1) if row[1] else None,
        "avg_recon_error": round(row[2], 4) if row[2] else None,
        "total_readings": row[3]
    }

@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM readings")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM readings WHERE rf_status = 'FAULT DETECTED'")
    rf_faults = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM readings WHERE ae_status = 'FAULT DETECTED'")
    ae_faults = c.fetchone()[0]
    c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM readings")
    time_range = c.fetchone()
    conn.close()
    return {
        "total_readings": total,
        "rf_fault_count": rf_faults,
        "ae_fault_count": ae_faults,
        "first_reading": time_range[0],
        "last_reading": time_range[1]
    }

@app.post("/predict")
def predict(data: dict):
    features = data['features']
    features_df = pd.DataFrame([features])

    # Random Forest
    rf_pred = rf_model.predict(features_df)[0]
    rf_prob = rf_model.predict_proba(features_df)[0][1]
    rf_health = round((1 - rf_prob) * 100, 1)
    rf_status = "HEALTHY" if rf_pred == 0 else "FAULT DETECTED"

    # Autoencoder
    features_scaled = scaler.transform(features_df)
    features_tensor = torch.FloatTensor(features_scaled)
    with torch.no_grad():
        reconstructed = autoencoder(features_tensor)
        recon_error = nn.MSELoss()(reconstructed, features_tensor).item()
    ae_status = "FAULT DETECTED" if recon_error > threshold else "HEALTHY"
    ae_health = round(max(0, 100 - (recon_error / threshold) * 100), 1)

    reading = {
        'timestamp': datetime.now().isoformat(),
        'rf_health': rf_health,
        'rf_status': rf_status,
        'ae_health': ae_health,
        'ae_status': ae_status,
        'recon_error': round(recon_error, 4),
        'threshold': round(threshold, 4),
        'rms': round(features['rms'], 3),
        'crest_factor': round(features['crest_factor'], 3),
        'energy_100hz': round(features['energy_100hz'], 3),
    }

    latest_reading.update(reading)
    history.append(reading)
    insert_reading(reading)  # <-- persist to SQLite

    print(f"RF: {rf_status} {rf_health}% | AE: {ae_status} {ae_health}% | Error: {recon_error:.4f}")

    return reading