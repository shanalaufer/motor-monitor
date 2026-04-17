import csv
import os
import socket
import json
import threading
import numpy as np
from scipy.fft import fft, fftfreq
import joblib
import pandas as pd
import requests

LABEL = "faulty"  # "healthy" or "faulty"
API_URL = "https://motor-fault-api-f7r8.onrender.com"

model = joblib.load('motor_model.pkl')
fs = 500
burst_count = 0

def extract_features(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)
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
        'rms': float(rms),
        'peak': float(peak),
        'crest_factor': float(crest_factor),
        'energy_50hz': float(energy_at(50)),
        'energy_100hz': float(energy_at(100)),
        'energy_150hz': float(energy_at(150)),
    }

def _post_to_api(f):
    try:
        requests.post(f"{API_URL}/predict", json={'features': f}, timeout=5)
    except Exception as e:
        print(f"API error: {e}")

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
        current_label = 0
        if data:
            payload = json.loads(data.decode())

            # New 3-axis format; fall back to legacy single-axis key
            if 'samples_z' in payload:
                samples_x = np.array(payload['samples_x'], dtype=np.float32)
                samples_y = np.array(payload['samples_y'], dtype=np.float32)
                samples_z = np.array(payload['samples_z'], dtype=np.float32)
                pwm_duty  = payload.get('pwm_duty')
            else:
                samples_z = np.array(payload['samples'], dtype=np.float32)
                samples_x = samples_y = samples_z
                pwm_duty  = None

            rms_check = np.sqrt(np.mean(samples_z**2))
            if len(samples_z) == 0 or rms_check < 0.05:
                print("Skipping low-energy burst")
                continue

            features = extract_features(samples_z)  # Z axis drives existing pipeline
            features_df = pd.DataFrame([features])
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0][1]
            health_score = round((1 - probability) * 100, 1)
            status = "HEALTHY" if prediction == 0 else "FAULT DETECTED"
            print(f"Status: {status} | Health: {health_score}% | RMS: {features['rms']:.3f} | E100Hz: {features['energy_100hz']:.3f}")

            threading.Thread(target=_post_to_api, args=(features,), daemon=True).start()

            csv_path = 'real_data_v2.csv'
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz', 'label'])
                writer.writerow([
                    features['rms'],
                    features['peak'],
                    features['crest_factor'],
                    features['energy_50hz'],
                    features['energy_100hz'],
                    features['energy_150hz'],
                    current_label
                ])

            npy_path = 'raw_healthy.npy' if LABEL == 'healthy' else 'raw_faulty.npy'
            burst = np.stack([samples_x, samples_y, samples_z])  # shape (3, N)
            if os.path.exists(npy_path):
                existing = np.load(npy_path)
                np.save(npy_path, np.concatenate([existing, burst[np.newaxis]], axis=0))
            else:
                np.save(npy_path, burst[np.newaxis])  # shape (1, 3, N)

            burst_count += 1
            print(f"Saved burst {burst_count} ({LABEL})")

    except Exception as e:
        print('Error:', e)