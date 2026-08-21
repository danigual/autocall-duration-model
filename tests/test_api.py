"""Integration tests for the FastAPI inference server.

Uses FastAPI's TestClient which sends real HTTP requests to the app
in-memory without starting an actual server. The lifespan context manager
fires normally, so the real model artifact is loaded from artifacts/.
Run train.py before running these tests to ensure the artifact exists.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "product_type": "Snowball",
    "basket_type": "worst_of",
    "underlyings": "CLNE|DRC",
    "autocall_barrier_pct": 1.0,
    "protection_barrier_pct": 0.7,
    "no_call_period_months": 6.0,
    "observation_frequency": "3M",
    "quoted_implied_vol": 0.35,
    "requested_date": "2021-06-01",
    "start_date": "2021-06-15",
    "end_date": "2024-06-15",
}


@pytest.fixture(scope="module")
def client():
    """TestClient that triggers the lifespan and loads the real model artifact.

    scope='module' loads the model once for all tests in this file.
    """
    with TestClient(app) as c:
        yield c


class TestHealth:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self, client):
        """Health check must return HTTP 200."""
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        """Response body must contain status='ok'."""
        assert client.get("/health").json()["status"] == "ok"

    def test_model_is_loaded(self, client):
        """model_loaded must be True after lifespan startup."""
        assert client.get("/health").json()["model_loaded"] is True


class TestPredict:
    """Tests for the POST /predict endpoint."""

    def test_valid_payload_returns_200(self, client):
        """A complete valid payload must return HTTP 200."""
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_contains_duration_key(self, client):
        """Response body must contain the avg_duration_months key."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "avg_duration_months" in data

    def test_prediction_is_positive(self, client):
        """Predicted duration must be strictly positive."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["avg_duration_months"] > 0

    def test_prediction_is_finite(self, client):
        """Predicted duration must be a finite number, not NaN or Inf."""
        import math
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert math.isfinite(data["avg_duration_months"])

    def test_single_underlying(self, client):
        """Single-asset basket must produce a valid prediction."""
        payload = {**VALID_PAYLOAD, "underlyings": "CLNE", "basket_type": "single"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["avg_duration_months"] > 0

    def test_missing_field_returns_422(self, client):
        """A payload missing a required field must return HTTP 422."""
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "underlyings"}
        assert client.post("/predict", json=incomplete).status_code == 422

    def test_wrong_type_returns_422(self, client):
        """A payload with wrong field type must return HTTP 422."""
        bad = {**VALID_PAYLOAD, "autocall_barrier_pct": "not_a_float"}
        assert client.post("/predict", json=bad).status_code == 422

    def test_empty_body_returns_422(self, client):
        """An empty request body must return HTTP 422."""
        assert client.post("/predict", json={}).status_code == 422
