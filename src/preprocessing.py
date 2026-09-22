import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def drop_faulty_column(path: str):
    df = pd.read_csv(path)

    if "oeder_status" in df.columns:
        df.drop(columns=["oeder_status"], inplace=True)

    df.to_csv(path, index=False)

    return df


def convert_date_columns(df):
    dates_col = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in dates_col:
        df[col] = pd.to_datetime(df[col])

    return df


def impute_numerical_features(train, numerical_features):
    train[numerical_features] = train[numerical_features].fillna(
        train[numerical_features].median()
    )

    return train


def fill_missing_seller_state(train):
    train["seller_state"] = train["seller_state"].fillna("Unknown")

    return train


def select_features_and_target(train, features):
    x = train[features]
    y = train["delivery_status"]

    return x, y


def build_preprocessing_transformer(
    categorical_features,
    numerical_features,
):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            ),
        ]
    )

    return transformer


def fit_transformer(transformer, x):
    X_train_transformed = transformer.fit_transform(x)
    features_name = transformer.get_feature_names_out()

    return X_train_transformed, features_name


def transform_new_data(transformer, x):
    return transformer.transform(x)


def to_dataframe(X_transformed, features_name):
    return pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=features_name,
    )


def save_transformer(
    transformer,
    features_name,
    transformer_path: str,
    features_name_path: str,
):
    joblib.dump(transformer, transformer_path)
    joblib.dump(features_name, features_name_path)


def load_transformer(transformer_path: str):
    return joblib.load(transformer_path)


def load_features_name(features_name_path: str):
    return joblib.load(features_name_path)
