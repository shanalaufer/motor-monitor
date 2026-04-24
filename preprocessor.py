import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch
from scipy.fft import fft, fftfreq

FS = 500
BANDPASS_LOW = 10
BANDPASS_HIGH = 200
FILTER_ORDER = 4


def _bandpass_filter(signal):
    sos = butter(FILTER_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', fs=FS, output='sos')
    return sosfilt(sos, signal)


def _best_axis_index(axes):
    """Return index of the axis with the highest Welch PSD peak power."""
    peak_powers = []
    for ax in axes:
        freqs, psd = welch(ax, fs=FS, nperseg=min(256, len(ax)))
        peak_powers.append(psd.max())
    return int(np.argmax(peak_powers))


def extract_features(signal):
    N = len(signal)
    yf = np.abs(fft(signal)[:N // 2]) * 2 / N
    xf = fftfreq(N, 1 / FS)[:N // 2]

    def energy_at(hz, bw=5):
        mask = (xf >= hz - bw) & (xf <= hz + bw)
        return float(np.sum(yf[mask]))

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = float(peak / rms) if rms > 0 else 0.0
    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'energy_50hz': energy_at(50),
        'energy_100hz': energy_at(100),
        'energy_150hz': energy_at(150),
    }


def process_dataset(raw, label):
    rows = []
    for burst in raw:
        # burst shape: (3, 256) — axes are X, Y, Z
        axes = [burst[i].astype(np.float64) for i in range(3)]

        # Remove DC offset
        axes = [ax - ax.mean() for ax in axes]

        # Bandpass filter
        axes = [_bandpass_filter(ax) for ax in axes]

        # Select best axis by peak PSD power
        best_idx = _best_axis_index(axes)
        best_axis = axes[best_idx]

        features = extract_features(best_axis)
        features['label'] = label
        rows.append(features)
    return rows


def main():
    healthy_raw = np.load('raw_healthy.npy')  # (202, 3, 256)
    faulty_raw = np.load('raw_faulty.npy')    # (95, 3, 256)

    healthy_rows = process_dataset(healthy_raw, label=0)
    faulty_rows = process_dataset(faulty_raw, label=1)

    all_rows = healthy_rows + faulty_rows
    df = pd.DataFrame(all_rows, columns=[
        'rms', 'peak', 'crest_factor',
        'energy_50hz', 'energy_100hz', 'energy_150hz',
        'label',
    ])
    df.to_csv('real_data_v2.csv', index=False)

    print(f"Done. Healthy samples: {len(healthy_rows)}, Faulty samples: {len(faulty_rows)}, Total: {len(all_rows)}")
    print(f"Saved to real_data_v2.csv")


if __name__ == '__main__':
    main()
