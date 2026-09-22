import time
import uuid

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV

from src.logging_config import logger
from src.validation import validate_data


# ============================================================
# Serving Configuration
# ============================================================

NUMERICAL_FEATURES = [
    "total_items",
    "total_price",
    "total_freight_value",
    "total_payment",
    "max_installments",
    "payment_count",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "month_name",
    "day_name",
    "customer_state",
    "seller_state",
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class InvalidInputError(Exception):
    pass


# ============================================================
# Inference / Pretrained Objects
# ============================================================


def load_transformer(transformer_path: str):
    logger.info("Loading transformer from %s", transformer_path)
    return joblib.load(transformer_path)


def load_feature_names(features_name_path: str):
    logger.info("Loading feature names from %s", features_name_path)
    return joblib.load(features_name_path)


def transform_new_data(transformer, x):
    logger.info("Transforming new data")
    return transformer.transform(x)


def prepare_inference_dataframe(
    transformer,
    x,
    features_name,
):
    X_transformed = transform_new_data(
        transformer,
        x,
    )

    logger.info(
        "Inference data transformed successfully. Shape: %s",
        X_transformed.shape,
    )

    return pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=features_name,
    )


# ============================================================
# Prepare Data
# ============================================================


def prepare_split_features_targets(
    train,
    validation,
    test,
    features,
    target="delivery_status",
):
    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

    X_test = test[features]
    y_test = test[target]

    logger.info("Train, validation, and test features prepared")

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def transform_splits(
    transformer,
    X_train,
    X_validation,
    X_test,
):
    logger.info("Transforming train, validation, and test data")

    X_train_transformed = transformer.transform(X_train)
    X_validation_transformed = transformer.transform(X_validation)
    X_test_transformed = transformer.transform(X_test)

    logger.info("All data splits transformed successfully")

    return (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
    )


# ============================================================
# Baseline Model
# ============================================================


def train_baseline(
    X_train_transformed,
    y_train,
):
    logger.info("Training baseline model")

    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(
        X_train_transformed,
        y_train,
    )

    logger.info("Baseline model trained successfully")

    return baseline


# ============================================================
# Logistic Regression
# ============================================================


def train_logistic_regression(
    X_train_transformed,
    y_train,
):
    logger.info("Training logistic regression model")

    logistic_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    logistic_model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info("Logistic regression model trained successfully")

    return logistic_model


# ============================================================
# Random Forest
# ============================================================


def train_random_forest(
    X_train_transformed,
    y_train,
):
    logger.info("Training random forest model")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info("Random forest model trained successfully")

    return rf_model


# ============================================================
# Model Evaluation
# ============================================================


def evaluate_classification(
    y_true,
    y_pred,
    pos_label="late",
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        zero_division=0,
    )

    logger.info(
        "Evaluation completed | Accuracy: %.4f | "
        "Precision: %.4f | Recall: %.4f | F1: %.4f",
        accuracy,
        precision,
        recall,
        f1,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_confusion_matrix(
    name,
    y_true,
    y_pred,
    labels=None,
):
    logger.info(
        "Calculating confusion matrix for %s",
        name,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    logger.info(
        "Confusion matrix:\n%s",
        matrix,
    )

    return matrix


# ============================================================
# Random Forest Tuning
# ============================================================


def tune_random_forest(
    rf_model,
    X_train_transformed,
    y_train,
):
    logger.info("Starting random forest hyperparameter tuning")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }

    late_f1 = make_scorer(
        f1_score,
        pos_label="late",
    )

    rf_tuning = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring=late_f1,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    rf_tuning.fit(
        X_train_transformed,
        y_train,
    )

    logger.info(
        "Random forest tuning completed | Best parameters: %s",
        rf_tuning.best_params_,
    )

    return rf_tuning


# ============================================================
# Probability & Threshold
# ============================================================


def predict_proba_late(
    model,
    X_transformed,
):
    logger.info("Generating prediction probabilities")

    return model.predict_proba(X_transformed)[:, 0]


def threshold_predictions(
    prob,
    threshold,
):
    predictions = ["late" if p >= threshold else "on time" for p in prob]

    logger.info(
        "Predictions generated using threshold %.2f",
        threshold,
    )

    return predictions


def scan_thresholds(
    y_true,
    prob,
    thresholds=(0.2, 0.3, 0.4, 0.5),
):
    results = []

    for threshold in thresholds:
        pred = threshold_predictions(
            prob,
            threshold,
        )

        f1 = f1_score(
            y_true,
            pred,
            pos_label="late",
            zero_division=0,
        )

        logger.info(
            "Threshold: %.2f | F1: %.4f",
            threshold,
            f1,
        )

        results.append(
            {
                "threshold": threshold,
                "f1": f1,
            }
        )

    return results


def evaluate_at_threshold(
    y_true,
    prob,
    threshold,
    labels=("late", "on time"),
):
    pred = threshold_predictions(
        prob,
        threshold,
    )

    accuracy = accuracy_score(
        y_true,
        pred,
    )

    precision = precision_score(
        y_true,
        pred,
        pos_label="late",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        pred,
        pos_label="late",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        pred,
        pos_label="late",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        pred,
        labels=list(labels),
    )

    logger.info(
        "Threshold evaluation | Threshold: %.2f | "
        "Accuracy: %.4f | Precision: %.4f | "
        "Recall: %.4f | F1: %.4f",
        threshold,
        accuracy,
        precision,
        recall,
        f1,
    )

    logger.info(
        "Confusion matrix:\n%s",
        matrix,
    )

    return {
        "predictions": pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }


# ============================================================
# Final Model - Save / Load
# ============================================================


def save_final_model(
    model,
    path: str,
):
    joblib.dump(
        model,
        path,
    )

    logger.info(
        "Final model saved to %s",
        path,
    )

    return path


def save_final_threshold(
    threshold,
    path: str,
):
    joblib.dump(
        threshold,
        path,
    )

    logger.info(
        "Final threshold %.2f saved to %s",
        threshold,
        path,
    )

    return path


def load_final_model(
    path: str,
):
    logger.info(
        "Loading final model from %s",
        path,
    )

    return joblib.load(path)


def load_final_threshold(
    path: str,
):
    logger.info(
        "Loading final threshold from %s",
        path,
    )

    return joblib.load(path)


# ============================================================
# Final Results
# ============================================================


def build_final_results(
    model_name,
    threshold,
    y_true,
    y_pred,
):
    results = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_late": precision_score(
            y_true,
            y_pred,
            pos_label="late",
            zero_division=0,
        ),
        "recall_late": recall_score(
            y_true,
            y_pred,
            pos_label="late",
            zero_division=0,
        ),
        "f1_late": f1_score(
            y_true,
            y_pred,
            pos_label="late",
            zero_division=0,
        ),
    }

    logger.info(
        "Final results generated: %s",
        results,
    )

    return results


def save_final_results(
    final_results,
    path: str,
):
    joblib.dump(
        final_results,
        path,
    )

    logger.info(
        "Final results saved to %s",
        path,
    )

    return path


# ============================================================
# Serving - Input Validation
# ============================================================


def validate_input(
    x: pd.DataFrame,
    required_columns=MODEL_FEATURES,
):
    if x is None or len(x) == 0:
        raise InvalidInputError("Input is empty")

    missing = set(required_columns) - set(x.columns)

    if missing:
        raise InvalidInputError(f"Missing required columns: {sorted(missing)}")


# ============================================================
# Serving - Malformed Values
# ============================================================


def handle_malformed_values(
    x: pd.DataFrame,
) -> pd.DataFrame:

    x = x.copy()

    for col in NUMERICAL_FEATURES:

        if col in x.columns:

            before_na = x[col].isna().sum()

            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

            after_na = x[col].isna().sum()

            if after_na > before_na:
                logger.warning(
                    "Column '%s' had %d non-numeric value(s) "
                    "and they were converted to NaN.",
                    col,
                    after_na - before_na,
                )

    for col in CATEGORICAL_FEATURES:

        if col in x.columns and x[col].isna().any():

            n_missing = x[col].isna().sum()

            logger.warning(
                "Column '%s' had %d missing value(s). " "Filled with 'Unknown'.",
                col,
                n_missing,
            )

            x[col] = x[col].fillna("Unknown")

    return x


# ============================================================
# Serving - Prediction Request
# ============================================================


def predict(
    x_raw: pd.DataFrame,
    transformer,
    model,
    threshold: float,
    model_version: str,
    request_id: str = None,
):

    request_id = request_id or str(uuid.uuid4())

    start_time = time.time()

    n_rows = len(x_raw) if x_raw is not None else 0

    logger.info(
        "[%s] Prediction request received | " "rows=%d | model_version=%s",
        request_id,
        n_rows,
        model_version,
    )

    logger.info(
        "[%s] Raw input: %s",
        request_id,
        (x_raw.to_dict(orient="records") if x_raw is not None else None),
    )

    try:

        validate_input(x_raw)

        # Great Expectations validation
        validation_passed = validate_data(x_raw[MODEL_FEATURES])

        if not validation_passed:
            raise InvalidInputError("Data validation failed")

        x_clean = handle_malformed_values(x_raw)

        x_transformed = transform_new_data(
            transformer,
            x_clean[MODEL_FEATURES],
        )

        prob_late = predict_proba_late(
            model,
            x_transformed,
        )

        predictions = threshold_predictions(
            prob_late,
            threshold,
        )

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "[%s] Prediction success | "
            "output=%s | latency_ms=%.2f | "
            "model_version=%s",
            request_id,
            predictions,
            latency_ms,
            model_version,
        )

        return {
            "request_id": request_id,
            "predictions": predictions,
            "probabilities_late": prob_late.tolist(),
            "model_version": model_version,
            "latency_ms": latency_ms,
        }

    except InvalidInputError as e:

        latency_ms = (time.time() - start_time) * 1000

        logger.error(
            "[%s] Invalid input | " "error=%s | latency_ms=%.2f | " "model_version=%s",
            request_id,
            e,
            latency_ms,
            model_version,
        )

        return {
            "request_id": request_id,
            "error": str(e),
            "model_version": model_version,
            "latency_ms": latency_ms,
        }

    except Exception:

        latency_ms = (time.time() - start_time) * 1000

        logger.exception(
            "[%s] Unexpected error during prediction | "
            "latency_ms=%.2f | model_version=%s",
            request_id,
            latency_ms,
            model_version,
        )

        return {
            "request_id": request_id,
            "error": "internal_error",
            "model_version": model_version,
            "latency_ms": latency_ms,
        }
