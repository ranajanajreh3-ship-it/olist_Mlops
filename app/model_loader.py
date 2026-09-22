import mlflow
import joblib
import os
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)
MODEL_NAME = "OlistDeliveryModel"
MODEL_ALIAS = "champion"


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    model_uri = model_version.source.replace(
        "run:/",
        "runs:/",
        1
    )

    model = mlflow.sklearn.load_model(model_uri)

    return model


def get_model_info():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": model_version.version,
    }


def load_transformer():
    return joblib.load("models/transformer.pkl")


def load_threshold():
    return joblib.load("models/final_threshold.pkl")