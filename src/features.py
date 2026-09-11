import numpy as np
import pandas as pd


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


def aggregate_order_items(order_items):
    items_agg = order_items.groupby("order_id").agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    ).reset_index()

    return items_agg


def aggregate_order_payments(order_payments):
    payment_agg = order_payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"),
    ).reset_index()

    return payment_agg


def build_ml_table(orders, items_agg, payment_agg):
    ml_table = orders.merge(
        items_agg,
        on="order_id",
        how="left",
    )

    ml_table = ml_table.merge(
        payment_agg,
        on="order_id",
        how="left",
    )

    return ml_table


def create_delivery_status_label(ml_table):
    ml_table["delivery_status"] = (
        ml_table["order_delivered_customer_date"]
        <= ml_table["order_estimated_delivery_date"]
    ).map({
        True: "on time",
        False: "late",
    })

    return ml_table


def haversine(lat1, lon1, lat2, lon2):
    earth_radius = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c


def add_geo_features(
    df,
    orders,
    customers,
    order_items,
    sellers,
    gelocation,
):
    geo_grouped = gelocation.groupby(
        "geolocation_zip_code_prefix"
    )[["geolocation_lat", "geolocation_lng"]].mean().reset_index()

    temp_geo = (
        df[["order_id"]]
        .merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
        )
        .merge(
            customers[
                [
                    "customer_id",
                    "customer_state",
                    "customer_zip_code_prefix",
                ]
            ],
            on="customer_id",
            how="left",
        )
        .merge(
            order_items[["order_id", "seller_id"]],
            on="order_id",
            how="left",
        )
        .merge(
            sellers[
                [
                    "seller_id",
                    "seller_state",
                    "seller_zip_code_prefix",
                ]
            ],
            on="seller_id",
            how="left",
        )
    )

    temp_geo = temp_geo.merge(
        geo_grouped,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )

    temp_geo = temp_geo.merge(
        geo_grouped,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        }
    )

    df["customer_state"] = temp_geo["customer_state"]
    df["seller_state"] = temp_geo["seller_state"]

    df["distance_km"] = haversine(
        temp_geo["customer_lat"],
        temp_geo["customer_lng"],
        temp_geo["seller_lat"],
        temp_geo["seller_lng"],
    )

    return df


def add_date_features(df):
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    df["month_name"] = df["order_purchase_timestamp"].dt.month_name()
    df["day_name"] = df["order_purchase_timestamp"].dt.day_name()

    return df


def create_delivery_duration_feature(train_data):
    train_data["delivery_duration_days"] = (
        train_data["order_delivered_customer_date"]
        - train_data["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    return train_data


def create_estimated_vs_actual_feature(train_data):
    train_data["estimated_vs_actual_days"] = (
        train_data["order_estimated_delivery_date"]
        - train_data["order_delivered_customer_date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    return train_data


def build_geo_table(
    train_data,
    orders,
    customers,
    order_items,
    sellers,
    gelocation,
):
    geo = (
        train_data[["order_id", "delivery_status"]]
        .merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
        )
        .merge(
            customers[["customer_id", "customer_state"]],
            on="customer_id",
            how="left",
        )
    )

    order_seller = order_items[
        ["order_id", "seller_id"]
    ].drop_duplicates("order_id")

    geo = geo.merge(
        order_seller,
        on="order_id",
        how="left",
    )

    geo = geo.merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
    )

    geo_ave = gelocation.groupby(
        "geolocation_zip_code_prefix"
    )[["geolocation_lat", "geolocation_lng"]].mean().reset_index()

    customer_location = customers[
        ["customer_id", "customer_zip_code_prefix"]
    ].merge(
        geo_ave,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    customers_location = customer_location.rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )

    sellers_location = sellers[
        ["seller_id", "seller_zip_code_prefix"]
    ].merge(
        geo_ave,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )

    sellers_location = sellers_location.rename(
        columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        }
    )

    geo = geo.merge(
        customers_location[
            ["customer_id", "customer_lat", "customer_lng"]
        ],
        on="customer_id",
        how="left",
    )

    geo = geo.merge(
        sellers_location[
            ["seller_id", "seller_lat", "seller_lng"]
        ],
        on="seller_id",
        how="left",
    )

    return geo


def merge_geo_into_train(train_data, geo_data):
    train = train_data.merge(
        geo_data[
            [
                "customer_state",
                "order_id",
                "seller_id",
                "seller_state",
                "customer_lat",
                "customer_lng",
                "seller_lat",
                "seller_lng",
            ]
        ],
        on="order_id",
        how="left",
    )

    return train


def add_distance_feature(train):
    train["distance_km"] = haversine(
        train["customer_lat"],
        train["customer_lng"],
        train["seller_lat"],
        train["seller_lng"],
    )

    return train