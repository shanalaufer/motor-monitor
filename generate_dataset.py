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
        # Imbalance creates energy at harmonics of rotation frequency
        # plus random impacts (high crest factor)
        harmonic2 = fault_amplitude * np.sin(2 * np.pi * 100 * t)  # 2x rotation
        harmonic3 = (fault_amplitude * 0.5) * np.sin(2 * np.pi * 150 * t)  # 3x rotation
        # Add random impacts to raise crest factor
        impacts = np.zeros(len(t))
        impact_times = np.random.choice(len(t), size=20, replace=False)
        impacts[impact_times] = fault_amplitude * 3
        signal = signal + harmonic2 + harmonic3 + impacts
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
    energy_100hz = energy_at(100)
    energy_150hz = energy_at(150)

    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'energy_50hz': energy_50hz,
        'energy_100hz': energy_100hz,
        'energy_150hz': energy_150hz,
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