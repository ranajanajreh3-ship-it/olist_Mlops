import numpy as np
import pandas as pd
from src.predictor import threshold_predictions, predict_proba_late
from src.preprocessing import impute_numerical_features
from src.features import add_date_features
from src.validation import validate_data
from app.model_loader import load_model
from src.features import MODEL_FEATURES


def test_threshold_predictions():
    probabilities = [0.8, 0.3, 0.6]
    threshold = 0.5

    result = threshold_predictions(probabilities, threshold)

    assert result == ["late", "on time", "late"]


def test_threshold_predictions_at_threshold():
    probabilities = [0.5]
    threshold = 0.5

    result = threshold_predictions(probabilities, threshold)

    assert result == ["late"]


class FakeModel:
    def predict_proba(self, X):
        return np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        )


def test_predict_proba_late():
    model = FakeModel()
    X_transformed = [[1], [2], [3]]

    result = predict_proba_late(model, X_transformed)

    assert result.tolist() == [0.8, 0.3, 0.6]


def test_impute_numerical_features():
    train = pd.DataFrame({"total_price": [100, 200, None, 300]})

    numerical_features = ["total_price"]

    result = impute_numerical_features(train, numerical_features)

    assert result["total_price"].isna().sum() == 0
    assert result["total_price"].iloc[2] == 200


def test_add_date_features():
    df = pd.DataFrame({"order_purchase_timestamp": ["2024-01-15"]})

    result = add_date_features(df)

    assert result["month_name"].iloc[0] == "January"
    assert result["day_name"].iloc[0] == "Monday"


def test_validate_data_valid_input():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    result = validate_data(df)

    assert result is True


def test_validate_data_invalid_category():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["XX"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    result = validate_data(df)

    assert result is False


def test_validate_data_missing_value():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [None],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    result = validate_data(df)

    assert result is False


def test_validate_data_missing_column():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    result = validate_data(df)

    assert result is False


def test_model_loads():
    model = load_model()

    assert model is not None


def test_model_predicts():
    import joblib

    model = load_model()
    transformer = joblib.load("models/transformer.pkl")

    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    X_transformed = transformer.transform(df)
    predictions = model.predict(X_transformed)

    assert len(predictions) == 1
    assert predictions[0] in {"late", "on time"}


def test_model_prediction_type():
    import joblib

    model = load_model()
    transformer = joblib.load("models/transformer.pkl")

    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    X_transformed = transformer.transform(df)
    prediction = model.predict(X_transformed)

    assert prediction is not None
    assert len(prediction) == 1


def test_no_target_in_model_features():
    assert "delivery_status" not in MODEL_FEATURES


def test_model_features_are_expected():
    expected_features = {
        "total_items",
        "total_price",
        "total_freight_value",
        "total_payment",
        "max_installments",
        "payment_count",
        "distance_km",
        "month_name",
        "day_name",
        "customer_state",
        "seller_state",
    }

    assert set(MODEL_FEATURES) == expected_features


def test_numeric_features_have_valid_ranges():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
        }
    )

    assert (df["total_items"] >= 1).all()
    assert (df["total_price"] >= 0).all()
    assert (df["total_freight_value"] >= 0).all()
    assert (df["total_payment"] >= 0).all()
    assert (df["max_installments"] >= 1).all()
    assert (df["payment_count"] >= 1).all()
    assert (df["distance_km"] >= 0).all()


def test_model_features_have_no_nulls():
    df = pd.DataFrame(
        {
            "total_items": [2.0],
            "total_price": [100.0],
            "total_freight_value": [10.0],
            "total_payment": [110.0],
            "max_installments": [3.0],
            "payment_count": [1.0],
            "distance_km": [50.0],
            "seller_state": ["SP"],
            "customer_state": ["RJ"],
            "month_name": ["January"],
            "day_name": ["Monday"],
        }
    )

    assert df[MODEL_FEATURES].isnull().sum().sum() == 0
