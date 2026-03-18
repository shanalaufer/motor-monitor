import numpy as np
import pandas as pd

# Signal parameters
fs = 1000
duration = 2
t = np.linspace(0, duration, fs * duration)

def generate_signal(fault=False, fault_amplitude=0.6):
    normal = np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.3, len(t))
    signal = normal + noise
    if fault:
        bearing_fault = fault_amplitude * np.sin(2 * np.pi * 120 * t)
        signal = signal + bearing_fault
    return signal

def extract_features(signal):
    # Time domain features
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms

    # Frequency domain features
    N = len(signal)
    yf = np.abs(np.fft.fft(signal)[:N//2]) * 2/N
    xf = np.fft.fftfreq(N, 1/fs)[:N//2]

    # Energy at key frequencies
    freq_resolution = xf[1] - xf[0]
    
    def energy_at(target_hz, bandwidth=5):
        mask = (xf >= target_hz - bandwidth) & (xf <= target_hz + bandwidth)
        return np.sum(yf[mask])

    energy_50hz  = energy_at(50)
    energy_120hz = energy_at(120)

    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'energy_50hz': energy_50hz,
        'energy_120hz': energy_120hz,
    }

# Generate dataset
rows = []
n_samples = 500  # 500 healthy + 500 faulty = 1000 total

print("Generating healthy signals...")
for i in range(n_samples):
    signal = generate_signal(fault=False)
    features = extract_features(signal)
    features['label'] = 0  # 0 = healthy
    rows.append(features)

print("Generating faulty signals...")
for i in range(n_samples):
    signal = generate_signal(fault=True)
    features = extract_features(signal)
    features['label'] = 1  # 1 = faulty
    rows.append(features)

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv('motor_dataset.csv', index=False)

print(f"Done! Dataset saved as motor_dataset.csv")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLabel counts:")
print(df['label'].value_counts())