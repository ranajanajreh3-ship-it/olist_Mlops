import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model_loader import (
    get_model_info,
    load_model,
    load_transformer,
    load_threshold,
)
import csv
from pathlib import Path
from src.predictor import predict as run_prediction
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from prometheus_client import Gauge
import time

PREDICTION_LOG = Path("logs/predictions.csv")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
)
LATE_PREDICTION_RATE = Gauge(
    "api_late_prediction_rate",
    "Current proportion of predictions classified as late",
)

PREDICTION_DRIFT = Gauge(
    "api_prediction_drift",
    "Absolute difference between current late prediction rate and baseline",
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
)
ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
)
PREDICTION_COUNT = Counter(
    "api_predictions_total",
    "Total number of predictions",
    ["prediction"],
)

app = FastAPI(title="Olist Delivery Prediction API", version="1.0.0")
model = load_model()
transformer = load_transformer()
threshold = load_threshold()
model_version = "test"


@app.middleware("http")
async def count_requests(request, call_next):
    REQUEST_COUNT.inc()

    start_time = time.time()

    try:
        response = await call_next(request)
        return response
    except Exception:
        ERROR_COUNT.inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


class PredictionRequest(BaseModel):
    total_items: float = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)
    total_payment: float = Field(ge=0)
    max_installments: float = Field(ge=1)
    payment_count: float = Field(ge=1)
    distance_km: float = Field(ge=0)

    seller_state: str
    customer_state: str
    month_name: str
    day_name: str


class PredictionResponse(BaseModel):
    request_id: str
    predictions: list[str]
    probabilities_late: list[float]
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return get_model_info()


def log_prediction(result, request_data):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)

    file_exists = PREDICTION_LOG.exists()

    with PREDICTION_LOG.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "request_id",
                    "prediction",
                    "probability_late",
                    "model_version",
                    "latency_ms",
                ]
            )

        writer.writerow(
            [
                pd.Timestamp.now().isoformat(),
                result["request_id"],
                result["predictions"][0],
                result["probabilities_late"][0],
                result["model_version"],
                result["latency_ms"],
            ]
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    x_raw = pd.DataFrame([request.model_dump()])

    result = run_prediction(
        x_raw=x_raw,
        transformer=transformer,
        model=model,
        threshold=threshold,
        model_version=str(model_version),
    )

    PREDICTION_COUNT.labels(prediction=result["predictions"][0]).inc()

    total_predictions = sum(
        child._value.get() for child in PREDICTION_COUNT._metrics.values()
    )

    late_predictions = PREDICTION_COUNT.labels(prediction="late")._value.get()

    if total_predictions > 0:
        late_rate = late_predictions / total_predictions

        LATE_PREDICTION_RATE.set(late_rate)

        baseline_late_rate = 0.087
        PREDICTION_DRIFT.set(abs(late_rate - baseline_late_rate))

    log_prediction(result, request.model_dump())

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    results = []

    for order in request.orders:
        x_raw = pd.DataFrame([order.model_dump()])

        result = run_prediction(
            x_raw=x_raw,
            transformer=transformer,
            model=model,
            threshold=threshold,
            model_version=str(model_version),
        )

        results.append(result)

    return {"results": results}
