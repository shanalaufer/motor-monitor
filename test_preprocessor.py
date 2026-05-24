"""
test_preprocessor.py — Preprocessing pipeline validation tests
Runs locally against real data files.

Usage:
    pytest test_preprocessor.py -v

Requires in the same directory:
    preprocessor.py, raw_healthy.npy, raw_faulty.npy, real_data_v2.csv
"""

import pytest
import numpy as np
import pandas as pd
from preprocessor import (
    _bandpass_filter,
    _best_axis_index,
    extract_features,
    process_dataset,
)

FS = 500
BURST_SAMPLES = 256
N_AXES = 3
EXPECTED_FEATURES = ['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz']


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def healthy_raw():
    return np.load('raw_healthy.npy')

@pytest.fixture(scope="module")
def faulty_raw():
    return np.load('raw_faulty.npy')

@pytest.fixture(scope="module")
def csv_data():
    return pd.read_csv('real_data_v2.csv')

@pytest.fixture(scope="module")
def single_healthy_burst(healthy_raw):
    """One real burst, DC removed, as float64 — ready for pipeline."""
    burst = healthy_raw[0].astype(np.float64)
    return [ax - ax.mean() for ax in burst]

@pytest.fixture(scope="module")
def synthetic_clean_signal():
    """Pure 100Hz sine wave — known ground truth for filter and feature tests."""
    t = np.linspace(0, BURST_SAMPLES / FS, BURST_SAMPLES, endpoint=False)
    return np.sin(2 * np.pi * 100 * t)

@pytest.fixture(scope="module")
def synthetic_dc_signal():
    """Sine wave with a large DC offset."""
    t = np.linspace(0, BURST_SAMPLES / FS, BURST_SAMPLES, endpoint=False)
    return np.sin(2 * np.pi * 100 * t) + 5.0  # +5g DC offset


# ── 1. Raw data shape and integrity ──────────────────────────────────────────

class TestRawDataShape:
    def test_healthy_is_3d(self, healthy_raw):
        assert healthy_raw.ndim == 3, f"Expected 3D array, got {healthy_raw.ndim}D"

    def test_faulty_is_3d(self, faulty_raw):
        assert faulty_raw.ndim == 3

    def test_healthy_has_3_axes(self, healthy_raw):
        assert healthy_raw.shape[1] == N_AXES, (
            f"Expected {N_AXES} axes, got {healthy_raw.shape[1]}"
        )

    def test_faulty_has_3_axes(self, faulty_raw):
        assert faulty_raw.shape[1] == N_AXES

    def test_healthy_burst_length(self, healthy_raw):
        assert healthy_raw.shape[2] == BURST_SAMPLES, (
            f"Expected {BURST_SAMPLES} samples per burst, got {healthy_raw.shape[2]}"
        )

    def test_faulty_burst_length(self, faulty_raw):
        assert faulty_raw.shape[2] == BURST_SAMPLES

    def test_healthy_sample_count(self, healthy_raw):
        assert healthy_raw.shape[0] == 202, (
            f"Expected 202 healthy bursts, got {healthy_raw.shape[0]}"
        )

    def test_faulty_sample_count(self, faulty_raw):
        assert faulty_raw.shape[0] == 95, (
            f"Expected 95 faulty bursts, got {faulty_raw.shape[0]}"
        )

    def test_no_nans_in_healthy(self, healthy_raw):
        assert not np.isnan(healthy_raw).any(), "NaNs found in raw_healthy.npy"

    def test_no_nans_in_faulty(self, faulty_raw):
        assert not np.isnan(faulty_raw).any(), "NaNs found in raw_faulty.npy"

    def test_no_infs_in_healthy(self, healthy_raw):
        assert not np.isinf(healthy_raw).any(), "Infs found in raw_healthy.npy"

    def test_no_infs_in_faulty(self, faulty_raw):
        assert not np.isinf(faulty_raw).any(), "Infs found in raw_faulty.npy"


# ── 2. DC offset removal ──────────────────────────────────────────────────────

class TestDCRemoval:
    def test_mean_near_zero_after_removal(self, synthetic_dc_signal):
        """After subtracting mean, signal mean should be ~0."""
        removed = synthetic_dc_signal - synthetic_dc_signal.mean()
        assert abs(removed.mean()) < 1e-10, (
            f"Mean after DC removal: {removed.mean():.2e} (expected ~0)"
        )

    def test_dc_removal_preserves_length(self, synthetic_dc_signal):
        removed = synthetic_dc_signal - synthetic_dc_signal.mean()
        assert len(removed) == len(synthetic_dc_signal)

    def test_dc_removal_preserves_ac_amplitude(self, synthetic_dc_signal):
        """Removing DC offset should not change the AC amplitude."""
        removed = synthetic_dc_signal - synthetic_dc_signal.mean()
        # Peak of pure sine should still be ~1.0
        assert abs(np.max(np.abs(removed)) - 1.0) < 0.05, (
            f"AC amplitude changed after DC removal: {np.max(np.abs(removed)):.3f}"
        )

    def test_real_bursts_have_low_mean_after_dc_removal(self, healthy_raw):
        """All real healthy bursts should have near-zero mean after DC removal."""
        for i, burst in enumerate(healthy_raw[:10]):  # check first 10
            for ax_idx in range(N_AXES):
                ax = burst[ax_idx].astype(np.float64)
                removed = ax - ax.mean()
                assert abs(removed.mean()) < 1e-8, (
                    f"Burst {i}, axis {ax_idx}: mean after DC removal = {removed.mean():.2e}"
                )


# ── 3. Bandpass filter ────────────────────────────────────────────────────────

class TestBandpassFilter:
    def test_output_length_unchanged(self, synthetic_clean_signal):
        filtered = _bandpass_filter(synthetic_clean_signal)
        assert len(filtered) == len(synthetic_clean_signal), (
            f"Filter changed signal length: {len(filtered)} vs {len(synthetic_clean_signal)}"
        )

    def test_no_nans_in_output(self, synthetic_clean_signal):
        filtered = _bandpass_filter(synthetic_clean_signal)
        assert not np.isnan(filtered).any(), "NaNs in filter output"

    def test_no_infs_in_output(self, synthetic_clean_signal):
        filtered = _bandpass_filter(synthetic_clean_signal)
        assert not np.isinf(filtered).any(), "Infs in filter output"

    def test_passband_signal_survives(self, synthetic_clean_signal):
        """100Hz sine is inside the 10-200Hz passband — should survive filtering."""
        filtered = _bandpass_filter(synthetic_clean_signal)
        # RMS should be close to original (allow for edge effects)
        rms_before = np.sqrt(np.mean(synthetic_clean_signal ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        ratio = rms_after / rms_before
        assert 0.5 < ratio < 1.5, (
            f"100Hz signal heavily attenuated by bandpass filter: ratio={ratio:.3f}"
        )

    def test_stopband_signal_attenuated(self):
        """5Hz sine is below the 10Hz cutoff — should be heavily attenuated."""
        t = np.linspace(0, BURST_SAMPLES / FS, BURST_SAMPLES, endpoint=False)
        low_freq = np.sin(2 * np.pi * 5 * t)
        filtered = _bandpass_filter(low_freq)
        rms_before = np.sqrt(np.mean(low_freq ** 2))
        rms_after = np.sqrt(np.mean(filtered ** 2))
        ratio = rms_after / rms_before
        assert ratio < 0.3, (
            f"5Hz signal not attenuated enough: ratio={ratio:.3f} (expected < 0.3)"
        )

    def test_filter_stable_on_real_data(self, healthy_raw):
        """Filter must not blow up on any real burst."""
        for i, burst in enumerate(healthy_raw):
            for ax_idx in range(N_AXES):
                ax = burst[ax_idx].astype(np.float64)
                ax = ax - ax.mean()
                filtered = _bandpass_filter(ax)
                assert not np.isnan(filtered).any(), f"NaN in burst {i} axis {ax_idx}"
                assert not np.isinf(filtered).any(), f"Inf in burst {i} axis {ax_idx}"


# ── 4. Axis selection ─────────────────────────────────────────────────────────

class TestAxisSelection:
    def test_returns_valid_index(self, single_healthy_burst):
        idx = _best_axis_index(single_healthy_burst)
        assert idx in (0, 1, 2), f"Invalid axis index: {idx}"

    def test_returns_int(self, single_healthy_burst):
        idx = _best_axis_index(single_healthy_burst)
        assert isinstance(idx, int), f"Expected int, got {type(idx)}"

    def test_selects_highest_power_axis(self):
        """Axis with injected high-power signal should always be selected."""
        t = np.linspace(0, BURST_SAMPLES / FS, BURST_SAMPLES, endpoint=False)
        low_noise = np.random.normal(0, 0.01, BURST_SAMPLES)
        high_power = np.sin(2 * np.pi * 100 * t) * 10  # 10x amplitude
        axes = [low_noise, low_noise.copy(), high_power]  # Z axis is dominant
        idx = _best_axis_index(axes)
        assert idx == 2, f"Expected axis 2 (high power), got axis {idx}"

    def test_consistent_on_repeated_calls(self, single_healthy_burst):
        """Same input should always return the same axis."""
        idx1 = _best_axis_index(single_healthy_burst)
        idx2 = _best_axis_index(single_healthy_burst)
        assert idx1 == idx2, "Axis selection is non-deterministic"


# ── 5. Feature extraction ─────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_returns_all_feature_keys(self, synthetic_clean_signal):
        features = extract_features(synthetic_clean_signal)
        missing = set(EXPECTED_FEATURES) - set(features.keys())
        assert not missing, f"Missing features: {missing}"

    def test_all_values_are_floats(self, synthetic_clean_signal):
        features = extract_features(synthetic_clean_signal)
        for k, v in features.items():
            assert isinstance(v, float), f"Feature '{k}' is {type(v)}, expected float"

    def test_no_nans_in_features(self, synthetic_clean_signal):
        features = extract_features(synthetic_clean_signal)
        for k, v in features.items():
            assert not np.isnan(v), f"NaN in feature '{k}'"

    def test_no_negative_energy_values(self, synthetic_clean_signal):
        features = extract_features(synthetic_clean_signal)
        for k in ['energy_50hz', 'energy_100hz', 'energy_150hz']:
            assert features[k] >= 0, f"Negative energy in '{k}': {features[k]}"

    def test_rms_is_positive(self, synthetic_clean_signal):
        features = extract_features(synthetic_clean_signal)
        assert features['rms'] > 0

    def test_peak_geq_rms(self, synthetic_clean_signal):
        """Peak must always be >= RMS — basic signal property."""
        features = extract_features(synthetic_clean_signal)
        assert features['peak'] >= features['rms'], (
            f"peak ({features['peak']:.4f}) < rms ({features['rms']:.4f})"
        )

    def test_crest_factor_geq_1(self, synthetic_clean_signal):
        """Crest factor = peak/RMS. For any real signal this is always >= 1."""
        features = extract_features(synthetic_clean_signal)
        assert features['crest_factor'] >= 1.0, (
            f"Crest factor < 1: {features['crest_factor']:.4f}"
        )

    def test_100hz_energy_high_for_100hz_sine(self, synthetic_clean_signal):
        """A pure 100Hz sine should have high energy_100hz and near-zero elsewhere."""
        features = extract_features(synthetic_clean_signal)
        assert features['energy_100hz'] > features['energy_50hz'], (
            "100Hz sine should have more energy at 100Hz than 50Hz"
        )
        assert features['energy_100hz'] > features['energy_150hz'], (
            "100Hz sine should have more energy at 100Hz than 150Hz"
        )

    def test_zero_division_safe(self):
        """extract_features must not crash or produce NaN on a flat (zero) signal."""
        signal = np.zeros(BURST_SAMPLES)
        features = extract_features(signal)
        for k, v in features.items():
            assert not np.isnan(v), f"NaN on zero input for feature '{k}'"
        assert features['crest_factor'] == 0.0  # defined as 0 when rms=0

    def test_rms_correct_for_known_signal(self):
        """RMS of a unit sine wave is 1/sqrt(2) ≈ 0.707."""
        t = np.linspace(0, BURST_SAMPLES / FS, BURST_SAMPLES, endpoint=False)
        sine = np.sin(2 * np.pi * 100 * t)
        features = extract_features(sine)
        expected_rms = 1 / np.sqrt(2)
        assert abs(features['rms'] - expected_rms) < 0.02, (
            f"RMS={features['rms']:.4f}, expected ~{expected_rms:.4f}"
        )


# ── 6. Full pipeline (process_dataset) ───────────────────────────────────────

class TestProcessDataset:
    def test_output_length_matches_input(self, healthy_raw, faulty_raw):
        healthy_rows = process_dataset(healthy_raw, label=0)
        faulty_rows = process_dataset(faulty_raw, label=1)
        assert len(healthy_rows) == 202
        assert len(faulty_rows) == 95

    def test_every_row_has_all_features(self, healthy_raw):
        rows = process_dataset(healthy_raw[:5], label=0)
        for i, row in enumerate(rows):
            missing = set(EXPECTED_FEATURES) - set(row.keys())
            assert not missing, f"Row {i} missing features: {missing}"

    def test_labels_assigned_correctly(self, healthy_raw, faulty_raw):
        healthy_rows = process_dataset(healthy_raw[:3], label=0)
        faulty_rows = process_dataset(faulty_raw[:3], label=1)
        assert all(r['label'] == 0 for r in healthy_rows)
        assert all(r['label'] == 1 for r in faulty_rows)

    def test_no_nans_in_any_row(self, healthy_raw, faulty_raw):
        for label, raw in [(0, healthy_raw), (1, faulty_raw)]:
            rows = process_dataset(raw, label=label)
            for i, row in enumerate(rows):
                for k, v in row.items():
                    if k == 'label':
                        continue
                    assert not np.isnan(v), (
                        f"NaN in label={label} row {i} feature '{k}'"
                    )

    def test_crest_factor_always_geq_1(self, healthy_raw, faulty_raw):
        for label, raw in [(0, healthy_raw), (1, faulty_raw)]:
            rows = process_dataset(raw, label=label)
            for i, row in enumerate(rows):
                assert row['crest_factor'] >= 1.0, (
                    f"label={label} row {i}: crest_factor={row['crest_factor']:.4f} < 1"
                )


# ── 7. CSV output validation ──────────────────────────────────────────────────

class TestCSVOutput:
    def test_csv_row_count(self, csv_data):
        """297 total samples: 202 healthy + 95 faulty."""
        assert len(csv_data) == 297, f"Expected 297 rows, got {len(csv_data)}"

    def test_csv_has_all_columns(self, csv_data):
        expected = set(EXPECTED_FEATURES + ['label'])
        missing = expected - set(csv_data.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_csv_label_distribution(self, csv_data):
        counts = csv_data['label'].value_counts()
        assert counts[0] == 202, f"Expected 202 healthy, got {counts[0]}"
        assert counts[1] == 95, f"Expected 95 faulty, got {counts[1]}"

    def test_csv_no_nans(self, csv_data):
        assert not csv_data.isnull().any().any(), (
            f"NaNs found in CSV:\n{csv_data.isnull().sum()}"
        )

    def test_csv_crest_factor_all_geq_1(self, csv_data):
        bad = csv_data[csv_data['crest_factor'] < 1.0]
        assert len(bad) == 0, f"{len(bad)} rows with crest_factor < 1"

    def test_csv_rms_all_positive(self, csv_data):
        bad = csv_data[csv_data['rms'] <= 0]
        assert len(bad) == 0, f"{len(bad)} rows with rms <= 0"

    def test_csv_energy_all_non_negative(self, csv_data):
        for col in ['energy_50hz', 'energy_100hz', 'energy_150hz']:
            bad = csv_data[csv_data[col] < 0]
            assert len(bad) == 0, f"{len(bad)} rows with {col} < 0"

    def test_faulty_higher_crest_factor_on_average(self, csv_data):
        """Real imbalance faults should show higher crest factor — key insight."""
        healthy_cf = csv_data[csv_data['label'] == 0]['crest_factor'].mean()
        faulty_cf = csv_data[csv_data['label'] == 1]['crest_factor'].mean()
        assert faulty_cf > healthy_cf, (
            f"Expected faulty crest_factor ({faulty_cf:.3f}) > "
            f"healthy ({healthy_cf:.3f})"
        )