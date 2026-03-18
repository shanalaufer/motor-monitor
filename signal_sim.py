import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Signal parameters
fs = 1000
duration = 2
t = np.linspace(0, duration, fs * duration)

# Normal motor signal: 50Hz
normal = np.sin(2 * np.pi * 50 * t)
noise = np.random.normal(0, 0.3, len(t))

# Fault signal: bearing defect at 120Hz
fault = 0.6 * np.sin(2 * np.pi * 120 * t)

# Two signals: healthy vs faulty motor
healthy = normal + noise
faulty = normal + noise + fault

def compute_fft(signal, fs):
    N = len(signal)
    yf = np.abs(fft(signal)[:N//2]) * 2/N
    xf = fftfreq(N, 1/fs)[:N//2]
    return xf, yf

xf_h, yf_h = compute_fft(healthy, fs)
xf_f, yf_f = compute_fft(faulty, fs)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

axes[0,0].plot(t[:500], healthy[:500], color='steelblue', linewidth=0.8)
axes[0,0].set_title('Healthy motor — time domain')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Amplitude')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t[:500], faulty[:500], color='crimson', linewidth=0.8)
axes[0,1].set_title('Faulty motor — time domain')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Amplitude')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(xf_h, yf_h, color='steelblue', linewidth=1)
axes[1,0].set_title('Healthy motor — FFT')
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].set_ylabel('Amplitude')
axes[1,0].set_xlim(0, 200)
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(xf_f, yf_f, color='crimson', linewidth=1)
axes[1,1].set_title('Faulty motor — FFT')
axes[1,1].set_xlabel('Frequency (Hz)')
axes[1,1].set_ylabel('Amplitude')
axes[1,1].set_xlim(0, 200)
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Healthy vs faulty motor comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('healthy_vs_faulty.png')
plt.show()
print("Done! Saved as healthy_vs_faulty.png")