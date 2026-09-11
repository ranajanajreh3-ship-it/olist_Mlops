def sort_by_purchase_timestamp(ml_table):
    return ml_table.sort_values("order_purchase_timestamp")


def split_data(
    ml_table,
    train_ratio=0.70,
    validation_ratio=0.85,
):
    train_data = int(len(ml_table) * train_ratio)
    val_data = int(len(ml_table) * validation_ratio)

    train = ml_table.iloc[:train_data].copy()
    validation = ml_table.iloc[train_data:val_data].copy()
    test = ml_table.iloc[val_data:].copy()

    return train, validation, test