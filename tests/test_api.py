from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.6, 0.4]] * X.shape[0])


with patch("app.model_loader.load_model", return_value=FakeModel()):
    with patch(
        "app.model_loader.get_model_info",
        return_value={
            "model_name": "OlistDeliveryModel",
            "alias": "champion",
            "version": "test",
        },
    ):
        from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "OlistDeliveryModel"
    assert data["alias"] == "champion"
    assert "version" in data


def test_predict():
    payload = {
        "total_items": 2,
        "total_price": 100.0,
        "total_freight_value": 20.0,
        "total_payment": 120.0,
        "max_installments": 3.0,
        "payment_count": 1.0,
        "distance_km": 50.0,
        "seller_state": "SP",
        "customer_state": "RJ",
        "month_name": "January",
        "day_name": "Monday",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "predictions" in data
    assert "probabilities_late" in data
    assert "model_version" in data
    assert "latency_ms" in data

    assert len(data["predictions"]) == 1
    assert len(data["probabilities_late"]) == 1


def test_predict_batch():
    payload = {
        "orders": [
            {
                "total_items": 2,
                "total_price": 100.0,
                "total_freight_value": 20.0,
                "total_payment": 120.0,
                "max_installments": 3.0,
                "payment_count": 1.0,
                "distance_km": 50.0,
                "seller_state": "SP",
                "customer_state": "RJ",
                "month_name": "January",
                "day_name": "Monday",
            },
            {
                "total_items": 1,
                "total_price": 250.0,
                "total_freight_value": 30.0,
                "total_payment": 280.0,
                "max_installments": 5.0,
                "payment_count": 1.0,
                "distance_km": 100.0,
                "seller_state": "MG",
                "customer_state": "SP",
                "month_name": "June",
                "day_name": "Friday",
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 2

    for result in data["results"]:
        assert "request_id" in result
        assert "predictions" in result
        assert "probabilities_late" in result
        assert "model_version" in result
        assert "latency_ms" in result


def test_predict_invalid_payload():
    payload = {
        "total_items": -5,
        "total_price": 100.0,
        "total_freight_value": 20.0,
        "total_payment": 120.0,
        "max_installments": 3.0,
        "payment_count": 1.0,
        "distance_km": 50.0,
        "seller_state": "SP",
        "customer_state": "RJ",
        "month_name": "January",
        "day_name": "Monday",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
