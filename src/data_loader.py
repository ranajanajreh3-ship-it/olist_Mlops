import joblib
import pandas as pd
from sqlalchemy import create_engine


def get_engine(db_url: str):
    return create_engine(db_url)


def get_tables_list(engine):
    return pd.read_sql(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """,
        engine,
    )


def load_all_tables(engine):
    customers = pd.read_sql("SELECT * FROM customers", engine)
    gelocation = pd.read_sql("SELECT * FROM gelocation", engine)
    order_items = pd.read_sql("SELECT * FROM order_items", engine)
    order_payments = pd.read_sql("SELECT * FROM order_payments", engine)
    orders = pd.read_sql("SELECT * FROM orders", engine)
    order_reviews = pd.read_sql("SELECT * FROM order_reviews", engine)
    products = pd.read_sql("SELECT * FROM products", engine)
    product_category_name_translation = pd.read_sql(
        "SELECT * FROM product_category_name_translation",
        engine,
    )
    sellers = pd.read_sql("SELECT * FROM sellers", engine)
    return {
        "customers": customers,
        "gelocation": gelocation,
        "order_items": order_items,
        "order_payments": order_payments,
        "orders": orders,
        "order_reviews": order_reviews,
        "products": products,
        "product_category_name_translation": product_category_name_translation,
        "sellers": sellers,
    }


def load_ml_table(path: str):
    return pd.read_csv(path)


def load_labeled_table(path: str):
    return pd.read_csv(path)


def load_train_data(path: str):
    return pd.read_csv(path)


def load_train_and_geo(train_path: str, geo_path: str):
    train_data = pd.read_csv(train_path)
    geo_data = pd.read_csv(geo_path)
    return train_data, geo_data


def load_geo_feature_tables(engine):
    orders = pd.read_sql("SELECT * FROM orders", engine)
    customers = pd.read_sql("SELECT * FROM customers", engine)
    order_items = pd.read_sql("SELECT * FROM order_items", engine)
    sellers = pd.read_sql("SELECT * FROM sellers", engine)
    gelocation = pd.read_sql("SELECT * FROM gelocation", engine)
    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "sellers": sellers,
        "gelocation": gelocation,
    }


def save_labeled_table(ml_table, path: str):
    ml_table.to_csv(path, index=False)


def save_splits(
    train,
    validation,
    test,
    train_path: str,
    validation_path: str,
    test_path: str,
):
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)


def save_geo_data(geo, path: str):
    geo.to_csv(path, index=False)


def save_engineered_train(train, path: str):
    train.to_csv(path, index=False)


def save_transformed_train_artifact(
    X_train_transformed,
    path: str,
):
    joblib.dump(X_train_transformed, path)


def load_test_and_validation(
    test_path: str,
    validation_path: str,
):
    test = pd.read_csv(test_path)
    validation = pd.read_csv(validation_path)
    return test, validation


def load_engineered_train(path: str):
    return pd.read_csv(path)
