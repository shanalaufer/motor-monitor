"""
test_models.py — Model regression tests for Motor Monitor
Runs locally against v2 model files and real_data_v2.csv.

Usage:
    pytest test_models.py -v

Requires in the same directory:
    motor_model_v2.pkl, scaler_v2.pkl,
    autoencoder_v2.pth, autoencoder_threshold_v2.pkl,
    real_data_v2.csv
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

FEATURE_COLS = ['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz']


# ── Model definitions (must match api.py exactly) ────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12), nn.ReLU(),
            nn.Linear(12, bottleneck), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 12), nn.ReLU(),
            nn.Linear(12, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rf_model():
    return joblib.load('motor_model_v2.pkl')

@pytest.fixture(scope="module")
def scaler():
    return joblib.load('scaler_v2.pkl')

@pytest.fixture(scope="module")
def threshold():
    return joblib.load('autoencoder_threshold_v2.pkl')

@pytest.fixture(scope="module")
def autoencoder(threshold):
    model = Autoencoder()
    model.load_state_dict(torch.load('autoencoder_v2.pth', weights_only=True))
    model.eval()
    return model

@pytest.fixture(scope="module")
def csv_data():
    return pd.read_csv('real_data_v2.csv')

@pytest.fixture(scope="module")
def healthy_samples(csv_data):
    return csv_data[csv_data['label'] == 0][FEATURE_COLS].reset_index(drop=True)

@pytest.fixture(scope="module")
def faulty_samples(csv_data):
    return csv_data[csv_data['label'] == 1][FEATURE_COLS].reset_index(drop=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def ae_recon_error(autoencoder, scaler, features_df):
    scaled = scaler.transform(features_df)
    tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        reconstructed = autoencoder(tensor)
        errors = nn.MSELoss(reduction='none')(reconstructed, tensor).mean(dim=1)
    return errors.numpy()


# ── 1. Model loading ──────────────────────────────────────────────────────────

class TestModelLoading:
    def test_rf_loads(self, rf_model):
        assert rf_model is not None

    def test_scaler_loads(self, scaler):
        assert scaler is not None

    def test_threshold_loads(self, threshold):
        assert threshold is not None

    def test_autoencoder_loads(self, autoencoder):
        assert autoencoder is not None

    def test_threshold_is_positive(self, threshold):
        assert float(threshold) > 0, f"Threshold must be positive, got {threshold}"

    def test_rf_has_expected_feature_count(self, rf_model):
        assert rf_model.n_features_in_ == 6, (
            f"RF expects {rf_model.n_features_in_} features, expected 6"
        )

    def test_scaler_has_expected_feature_count(self, scaler):
        assert scaler.n_features_in_ == 6

    def test_autoencoder_in_eval_mode(self, autoencoder):
        assert not autoencoder.training, "Autoencoder should be in eval mode"


# ── 2. Random Forest — output format ─────────────────────────────────────────

class TestRFOutputFormat:
    def test_predict_returns_array(self, rf_model, healthy_samples):
        preds = rf_model.predict(healthy_samples.iloc[:5])
        assert isinstance(preds, np.ndarray)

    def test_predict_proba_shape(self, rf_model, healthy_samples):
        proba = rf_model.predict_proba(healthy_samples.iloc[:5])
        assert proba.shape == (5, 2), f"Expected (5,2), got {proba.shape}"

    def test_probabilities_sum_to_1(self, rf_model, healthy_samples):
        proba = rf_model.predict_proba(healthy_samples.iloc[:10])
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Probabilities don't sum to 1: {sums}"

    def test_predictions_are_binary(self, rf_model, csv_data):
        preds = rf_model.predict(csv_data[FEATURE_COLS])
        unique = set(preds)
        assert unique <= {0, 1}, f"Unexpected prediction values: {unique}"


# ── 3. Random Forest — accuracy regression ───────────────────────────────────

class TestRFAccuracyRegression:
    def test_overall_accuracy_above_threshold(self, rf_model, csv_data):
        """RF v2 achieved 96.7% on real data — must stay above 90%."""
        X = csv_data[FEATURE_COLS]
        y = csv_data['label']
        preds = rf_model.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy >= 0.90, (
            f"RF accuracy dropped below 90%: {accuracy:.1%}"
        )

    def test_healthy_recall(self, rf_model, healthy_samples, csv_data):
        """RF must correctly identify at least 85% of healthy samples."""
        true_labels = csv_data[csv_data['label'] == 0]['label'].values
        preds = rf_model.predict(healthy_samples)
        recall = (preds == 0).mean()
        assert recall >= 0.85, (
            f"Healthy recall too low: {recall:.1%} (expected >= 85%)"
        )

    def test_fault_recall(self, rf_model, faulty_samples, csv_data):
        """RF must correctly identify at least 85% of faulty samples."""
        preds = rf_model.predict(faulty_samples)
        recall = (preds == 1).mean()
        assert recall >= 0.85, (
            f"Fault recall too low: {recall:.1%} (expected >= 85%)"
        )

    def test_healthy_not_all_predicted_faulty(self, rf_model, healthy_samples):
        """Sanity check — model hasn't collapsed to always predicting fault."""
        preds = rf_model.predict(healthy_samples)
        fault_rate = (preds == 1).mean()
        assert fault_rate < 0.5, (
            f"RF flagging {fault_rate:.1%} of healthy samples as faults"
        )

    def test_faulty_not_all_predicted_healthy(self, rf_model, faulty_samples):
        """Sanity check — model hasn't collapsed to always predicting healthy."""
        preds = rf_model.predict(faulty_samples)
        healthy_rate = (preds == 0).mean()
        assert healthy_rate < 0.5, (
            f"RF classifying {healthy_rate:.1%} of faulty samples as healthy"
        )


# ── 4. Random Forest — feature importance ────────────────────────────────────

class TestRFFeatureImportance:
    def test_feature_importances_sum_to_1(self, rf_model):
        total = rf_model.feature_importances_.sum()
        assert abs(total - 1.0) < 1e-6, f"Feature importances sum to {total}"

    def test_crest_factor_or_rms_is_top_feature(self, rf_model):
        """Key insight from real data: crest_factor and rms dominate.
        If a retrain demotes both, something changed in the data."""
        importances = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
        top_feature = max(importances, key=importances.get)
        assert top_feature in ('crest_factor', 'rms', 'peak'), (
            f"Unexpected top feature: {top_feature} ({importances[top_feature]:.3f}). "
            f"Expected crest_factor, rms, or peak to dominate. "
            f"All importances: {importances}"
        )


# ── 5. Autoencoder — output format ───────────────────────────────────────────

class TestAEOutputFormat:
    def test_forward_pass_output_shape(self, autoencoder, scaler, healthy_samples):
        scaled = scaler.transform(healthy_samples.iloc[:5])
        tensor = torch.FloatTensor(scaled)
        with torch.no_grad():
            output = autoencoder(tensor)
        assert output.shape == (5, 6), f"Expected (5,6), got {output.shape}"

    def test_recon_error_is_scalar_per_sample(self, autoencoder, scaler, healthy_samples):
        errors = ae_recon_error(autoencoder, scaler, healthy_samples.iloc[:10])
        assert errors.shape == (10,), f"Expected (10,), got {errors.shape}"

    def test_recon_error_all_positive(self, autoencoder, scaler, healthy_samples):
        errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        assert (errors >= 0).all(), f"Negative reconstruction errors found: {errors[errors < 0]}"

    def test_no_nans_in_output(self, autoencoder, scaler, healthy_samples):
        errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        assert not np.isnan(errors).any(), "NaNs in autoencoder reconstruction error"

    def test_no_infs_in_output(self, autoencoder, scaler, healthy_samples):
        errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        assert not np.isinf(errors).any(), "Infs in autoencoder reconstruction error"


# ── 6. Autoencoder — accuracy regression ─────────────────────────────────────

class TestAEAccuracyRegression:
    def test_healthy_error_below_threshold_majority(self, autoencoder, scaler, threshold, healthy_samples):
        """Majority of healthy samples must reconstruct below threshold."""
        errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        below = (errors <= float(threshold)).mean()
        assert below >= 0.70, (
            f"Only {below:.1%} of healthy samples below threshold "
            f"(expected >= 70%). Threshold may need recalibration."
        )

    def test_faulty_error_above_threshold_majority(self, autoencoder, scaler, threshold, faulty_samples):
        """Majority of faulty samples must exceed the threshold."""
        errors = ae_recon_error(autoencoder, scaler, faulty_samples)
        above = (errors > float(threshold)).mean()
        assert above >= 0.60, (
            f"Only {above:.1%} of faulty samples above threshold "
            f"(expected >= 60%). AE may not be discriminating faults."
        )

    def test_mean_faulty_error_exceeds_mean_healthy_error(self, autoencoder, scaler, healthy_samples, faulty_samples):
        """Mean reconstruction error must be higher for faulty than healthy."""
        healthy_errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        faulty_errors = ae_recon_error(autoencoder, scaler, faulty_samples)
        assert faulty_errors.mean() > healthy_errors.mean(), (
            f"Faulty mean error ({faulty_errors.mean():.4f}) not greater than "
            f"healthy mean error ({healthy_errors.mean():.4f})"
        )

    def test_error_ratio_meaningful(self, autoencoder, scaler, healthy_samples, faulty_samples):
        """Faulty/healthy error ratio should be > 1.5 — model has real separation."""
        healthy_errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        faulty_errors = ae_recon_error(autoencoder, scaler, faulty_samples)
        ratio = faulty_errors.mean() / healthy_errors.mean()
        assert ratio > 1.5, (
            f"Error ratio too low: {ratio:.2f}x (expected > 1.5x). "
            f"AE may not have meaningful separation between classes."
        )


# ── 7. Scaler ─────────────────────────────────────────────────────────────────

class TestScaler:
    def test_transform_output_shape(self, scaler, healthy_samples):
        scaled = scaler.transform(healthy_samples.iloc[:10])
        assert scaled.shape == (10, 6)

    def test_scaled_mean_near_zero(self, scaler, csv_data):
        """StandardScaler fitted on training data — transform of that data should be ~0 mean."""
        scaled = scaler.transform(csv_data[FEATURE_COLS])
        means = np.abs(scaled.mean(axis=0))
        assert (means < 0.5).all(), (
            f"Scaled means far from zero: {means}. Scaler may be mismatched to data."
        )

    def test_no_nans_after_scaling(self, scaler, csv_data):
        scaled = scaler.transform(csv_data[FEATURE_COLS])
        assert not np.isnan(scaled).any()


# ── 8. Model consistency — RF and AE agree on clear cases ────────────────────

class TestModelConsistency:
    def test_both_models_agree_on_most_healthy(self, rf_model, autoencoder, scaler, threshold, healthy_samples):
        """Both models should agree on healthy for most clearly healthy samples."""
        rf_preds = rf_model.predict(healthy_samples)
        ae_errors = ae_recon_error(autoencoder, scaler, healthy_samples)
        ae_preds = (ae_errors > float(threshold)).astype(int)
        agreement = (rf_preds == ae_preds).mean()
        assert agreement >= 0.60, (
            f"RF and AE agree on only {agreement:.1%} of healthy samples "
            f"(expected >= 60%)"
        )

    def test_both_models_agree_on_most_faulty(self, rf_model, autoencoder, scaler, threshold, faulty_samples):
        """Both models should agree on fault for most clearly faulty samples."""
        rf_preds = rf_model.predict(faulty_samples)
        ae_errors = ae_recon_error(autoencoder, scaler, faulty_samples)
        ae_preds = (ae_errors > float(threshold)).astype(int)
        agreement = (rf_preds == ae_preds).mean()
        assert agreement >= 0.55, (
            f"RF and AE agree on only {agreement:.1%} of faulty samples "
            f"(expected >= 55%)"
        )