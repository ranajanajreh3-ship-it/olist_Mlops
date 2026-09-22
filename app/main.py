import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model_loader import(
    get_model_info,
    load_model,
    load_transformer,
    load_threshold,
)
from src.predictor import (
    predict as run_prediction,
    MODEL_FEATURES,
)

app = FastAPI(
    title="Olist Delivery Prediction API",
    version="1.0.0"
)
model = load_model()
transformer = load_transformer()
threshold = load_threshold()
model_version = get_model_info()["version"]

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