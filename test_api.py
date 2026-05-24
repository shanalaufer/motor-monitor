"""
test_api.py — Automated API tests for Motor Monitor backend
Runs against the live Render API.

Usage:
    pip install pytest requests
    pytest test_api.py -v

All tests are independent — order doesn't matter.
"""

import pytest
import requests

# ── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://motor-fault-api-f7r8.onrender.com"
TIMEOUT = 60  # Render cold starts can take ~30s

# Representative feature vectors from real motor data
HEALTHY_FEATURES = {
    "rms": 0.593,
    "peak": 1.076,
    "crest_factor": 1.812,
    "energy_50hz": 0.138,
    "energy_100hz": 0.385,
    "energy_150hz": 0.158,
}

FAULTY_FEATURES = {
    "rms": 0.31,
    "peak": 1.85,
    "crest_factor": 5.9,
    "energy_50hz": 0.0008,
    "energy_100hz": 0.0021,
    "energy_150hz": 0.0009,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get(path, params=None):
    return requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)

def post(path, json):
    return requests.post(f"{BASE_URL}{path}", json=json, timeout=TIMEOUT)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_status_200(self):
        r = get("/")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_returns_message(self):
        r = get("/")
        body = r.json()
        assert "message" in body, f"Missing 'message' key: {body}"
        assert isinstance(body["message"], str)


class TestHealth:
    def test_status_200(self):
        r = get("/health")
        assert r.status_code == 200

    def test_returns_json(self):
        r = get("/health")
        body = r.json()
        assert isinstance(body, dict), f"Expected dict, got {type(body)}"

    def test_no_data_response_is_safe(self):
        """If no readings yet, must return a dict — not a crash."""
        r = get("/health")
        assert r.status_code == 200
        body = r.json()
        # Either "No data yet" or a full reading — both are valid
        assert isinstance(body, dict)


class TestHistory:
    def test_status_200(self):
        r = get("/history")
        assert r.status_code == 200

    def test_returns_list(self):
        r = get("/history")
        body = r.json()
        assert isinstance(body, list), f"Expected list, got {type(body)}"

    def test_list_items_have_expected_keys(self):
        r = get("/history")
        body = r.json()
        if len(body) == 0:
            pytest.skip("No history yet — run with live motor to populate")
        item = body[0]
        required_keys = {"timestamp", "rf_health", "rf_status", "ae_health", "ae_status", "recon_error"}
        missing = required_keys - set(item.keys())
        assert not missing, f"Missing keys in history item: {missing}"


class TestPredict:
    def test_healthy_input_returns_200(self):
        r = post("/predict", {"features": HEALTHY_FEATURES})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_faulty_input_returns_200(self):
        r = post("/predict", {"features": FAULTY_FEATURES})
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        required = {
            "timestamp", "rf_health", "rf_status",
            "ae_health", "ae_status", "recon_error",
            "threshold", "rms", "crest_factor", "energy_100hz"
        }
        missing = required - set(body.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_health_scores_in_range(self):
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        assert 0 <= body["rf_health"] <= 100, f"rf_health out of range: {body['rf_health']}"
        assert 0 <= body["ae_health"] <= 100, f"ae_health out of range: {body['ae_health']}"

    def test_status_values_are_valid_strings(self):
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        valid = {"HEALTHY", "FAULT DETECTED"}
        assert body["rf_status"] in valid, f"Invalid rf_status: {body['rf_status']}"
        assert body["ae_status"] in valid, f"Invalid ae_status: {body['ae_status']}"

    def test_recon_error_is_positive(self):
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        assert body["recon_error"] >= 0, f"recon_error is negative: {body['recon_error']}"

    def test_healthy_features_classify_correctly(self):
        """Known-healthy input should not trigger RF fault."""
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        assert body["rf_status"] == "HEALTHY", (
            f"RF misclassified known-healthy input as FAULT. "
            f"rf_health={body['rf_health']}"
        )

    def test_faulty_features_classify_correctly(self):
        """Known-faulty input should trigger at least one model."""
        r = post("/predict", {"features": FAULTY_FEATURES})
        body = r.json()
        either_fault = (
            body["rf_status"] == "FAULT DETECTED" or
            body["ae_status"] == "FAULT DETECTED"
        )
        assert either_fault, (
            f"Neither model detected known-faulty input. "
            f"RF={body['rf_status']} ({body['rf_health']}%), "
            f"AE={body['ae_status']} ({body['ae_health']}%)"
        )

    def test_missing_features_key_returns_error(self):
        """Malformed input — no 'features' wrapper — should not return 200."""
        r = post("/predict", HEALTHY_FEATURES)  # missing wrapper
        assert r.status_code != 200, (
            f"API accepted malformed input (missing 'features' key) and returned 200"
        )

    def test_empty_features_returns_error(self):
        """Empty features dict should fail, not crash silently."""
        r = post("/predict", {"features": {}})
        assert r.status_code != 200, "API accepted empty features dict"

    def test_rms_value_matches_input(self):
        """Returned rms should match what was sent (rounded to 3dp)."""
        r = post("/predict", {"features": HEALTHY_FEATURES})
        body = r.json()
        assert abs(body["rms"] - round(HEALTHY_FEATURES["rms"], 3)) < 0.001


class TestStats:
    def test_status_200(self):
        r = get("/stats")
        assert r.status_code == 200

    def test_required_keys_present(self):
        body = get("/stats").json()
        required = {"total_readings", "rf_fault_count", "ae_fault_count"}
        missing = required - set(body.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_counts_are_non_negative(self):
        body = get("/stats").json()
        assert body["total_readings"] >= 0
        assert body["rf_fault_count"] >= 0
        assert body["ae_fault_count"] >= 0

    def test_fault_counts_not_exceed_total(self):
        body = get("/stats").json()
        assert body["rf_fault_count"] <= body["total_readings"]
        assert body["ae_fault_count"] <= body["total_readings"]


class TestTrends:
    def test_default_returns_200(self):
        r = get("/trends")
        assert r.status_code == 200

    def test_custom_hours_param(self):
        r = get("/trends", params={"hours": 1})
        assert r.status_code == 200

    def test_required_keys_present(self):
        body = get("/trends").json()
        required = {"hours", "avg_rf_health", "avg_ae_health", "avg_recon_error", "total_readings"}
        missing = required - set(body.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_hours_reflects_param(self):
        body = get("/trends", params={"hours": 6}).json()
        assert body["hours"] == 6

    def test_health_averages_in_range_when_data_present(self):
        body = get("/trends").json()
        if body["avg_rf_health"] is not None:
            assert 0 <= body["avg_rf_health"] <= 100
        if body["avg_ae_health"] is not None:
            assert 0 <= body["avg_ae_health"] <= 100


class TestFaults:
    def test_status_200(self):
        r = get("/faults")
        assert r.status_code == 200

    def test_returns_list(self):
        body = get("/faults").json()
        assert isinstance(body, list)

    def test_fault_items_have_status_fields(self):
        body = get("/faults").json()
        if not body:
            pytest.skip("No faults recorded yet")
        item = body[0]
        assert "rf_status" in item or "ae_status" in item

    def test_all_items_are_actually_faults(self):
        """Every row in /faults must have at least one FAULT DETECTED status."""
        body = get("/faults").json()
        for item in body:
            is_fault = (
                item.get("rf_status") == "FAULT DETECTED" or
                item.get("ae_status") == "FAULT DETECTED"
            )
            assert is_fault, f"Non-fault row returned by /faults: {item}"


class TestExportCSV:
    def test_status_200(self):
        r = get("/export/csv")
        assert r.status_code == 200

    def test_content_type_is_csv(self):
        r = get("/export/csv")
        assert "text/csv" in r.headers.get("content-type", ""), (
            f"Expected text/csv, got: {r.headers.get('content-type')}"
        )

    def test_content_disposition_header(self):
        r = get("/export/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd, f"Missing attachment header: {cd}"
        assert ".csv" in cd

    def test_csv_has_header_row(self):
        r = get("/export/csv")
        first_line = r.text.strip().split("\n")[0]
        expected_cols = {"id", "timestamp", "rf_health", "rf_status"}
        cols = set(first_line.split(","))
        missing = expected_cols - cols
        assert not missing, f"CSV header missing columns: {missing}"


class TestDbCheck:
    def test_status_200(self):
        r = get("/db-check")
        assert r.status_code == 200

    def test_returns_row_count(self):
        body = get("/db-check").json()
        assert "rows_in_db" in body
        assert isinstance(body["rows_in_db"], int)
        assert body["rows_in_db"] >= 0


# ── Regression: predict + verify db count increments ─────────────────────────

class TestRegressionPredictPersists:
    def test_predict_increments_db_count(self):
        """Sending a /predict should increase the DB row count by 1."""
        before = get("/db-check").json()["rows_in_db"]
        post("/predict", {"features": HEALTHY_FEATURES})
        after = get("/db-check").json()["rows_in_db"]
        assert after == before + 1, (
            f"DB row count didn't increment: before={before}, after={after}"
        )