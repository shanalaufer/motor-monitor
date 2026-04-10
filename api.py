from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

app = FastAPI()

# Allow dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
rf_model = joblib.load('motor_model.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('autoencoder_threshold.pkl')

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
autoencoder.load_state_dict(torch.load('autoencoder.pth', weights_only=True))
autoencoder.eval()

# In-memory storage for latest reading and history
latest_reading = {}
history = deque(maxlen=100)

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

    print(f"RF: {rf_status} {rf_health}% | AE: {ae_status} {ae_health}% | Error: {recon_error:.4f}")

    return reading