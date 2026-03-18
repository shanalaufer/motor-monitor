import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from scipy.fft import fft, fftfreq

# Load the trained model
model = joblib.load('motor_model.pkl')

# Page config
st.set_page_config(page_title="Motor Monitor", page_icon="⚙", layout="wide")
st.title("Motor Health Monitor")
st.caption("Real-time bearing fault detection system")

# Sidebar controls
st.sidebar.header("Simulation controls")
mode = st.sidebar.radio("Motor mode", ["Healthy", "Faulty"])
fault_amplitude = st.sidebar.slider("Fault severity", 0.0, 1.0, 0.6)
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 5, 2)
run = st.sidebar.toggle("Run monitor", value=True)

# Signal parameters
fs = 1000
duration = 2
t = np.linspace(0, duration, fs * duration)

def generate_signal(faulty=False, amplitude=0.6):
    normal = np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.3, len(t))
    signal = normal + noise
    if faulty:
        fault = amplitude * np.sin(2 * np.pi * 120 * t)
        signal = signal + fault
    return signal

def extract_features(signal):
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
    }, yf, xf

# Main layout
col1, col2, col3 = st.columns(3)
status_box = st.empty()
plot_area = st.empty()
history_area = st.empty()

# Health score history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main loop
while run:
    signal = generate_signal(faulty=(mode == "Faulty"), amplitude=fault_amplitude)
    features, yf, xf = extract_features(signal)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]
    health_score = round((1 - probability) * 100, 1)

    # Update history
    st.session_state.history.append(health_score)
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

    # Status banner
    if prediction == 0:
        status_box.success(f"Status: HEALTHY — Health score: {health_score}%")
    else:
        status_box.error(f"WARNING: FAULT DETECTED — Health score: {health_score}%")

    # Metrics
    with col1:
        st.metric("Health score", f"{health_score}%")
    with col2:
        st.metric("RMS vibration", f"{features['rms']:.3f}")
    with col3:
        st.metric("Energy @ 120Hz", f"{features['energy_120hz']:.3f}")

    # Plots
    with plot_area.container():
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        axes[0].plot(t[:300], signal[:300],
                     color='steelblue' if prediction == 0 else 'crimson',
                     linewidth=0.8)
        axes[0].set_title('Vibration signal (time domain)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xf, yf,
                     color='steelblue' if prediction == 0 else 'crimson',
                     linewidth=1)
        axes[1].axvline(x=120, color='red', linestyle='--',
                        alpha=0.5, label='Fault frequency (120Hz)')
        axes[1].set_title('FFT — frequency content')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlim(0, 200)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Health score history chart
    with history_area.container():
        st.subheader("Health score history")
        history_df = pd.DataFrame({'Health score': st.session_state.history})
        st.line_chart(history_df, height=150)

    time.sleep(refresh_rate)