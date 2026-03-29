import socket
import json
import numpy as np
from scipy.fft import fft, fftfreq
import joblib
import pandas as pd

model = joblib.load('motor_model.pkl')
fs = 1000

def extract_features(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)  # remove DC offset / gravity
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    N = len(signal)
    yf = np.abs(fft(signal)[:N//2]) * 2/N
    xf = fftfreq(N, 1/fs)[:N//2]
    def energy_at(hz, bw=5):
        mask = (xf >= hz - bw) & (xf <= hz + bw)
        return np.sum(yf[mask])
    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'energy_50hz': energy_at(50),
        'energy_120hz': energy_at(120),
    }

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 5001))
server.listen(5)
print('Listening for ESP32 on port 5001...')

while True:
    try:
        conn, addr = server.accept()
        data = b''
        conn.settimeout(5)
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
        except:
            pass
        conn.close()
        
        if data:
            payload = json.loads(data.decode())
            samples = payload['samples']
            features = extract_features(samples)
            features_df = pd.DataFrame([features])
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0][1]
            health_score = round((1 - probability) * 100, 1)
            status = "HEALTHY" if prediction == 0 else "FAULT DETECTED"
            print(f"Status: {status} | Health: {health_score}% | RMS: {features['rms']:.3f} | E120Hz: {features['energy_120hz']:.3f}")
    except Exception as e:
        print('Error:', e)