import great_expectations as gx


NUMERIC_FEATURES = [
    "total_items",
    "total_price",
    "total_freight_value",
    "total_payment",
    "max_installments",
    "payment_count",
    "distance_km",
]

CATEGORICAL_FEATURES = {
    "seller_state": [
        "AC",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MG",
        "MS",
        "MT",
        "PA",
        "PB",
        "PE",
        "PI",
        "PR",
        "RJ",
        "RN",
        "RO",
        "RS",
        "SC",
        "SE",
        "SP",
        "Unknown",
    ],
    "customer_state": [
        "AC",
        "AL",
        "AM",
        "AP",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MG",
        "MS",
        "MT",
        "PA",
        "PB",
        "PE",
        "PI",
        "PR",
        "RJ",
        "RN",
        "RO",
        "RR",
        "RS",
        "SC",
        "SE",
        "SP",
        "TO",
    ],
    "month_name": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "day_name": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
}


def validate_data(df):
    context = gx.get_context()

    data_source = context.data_sources.add_pandas(name="validation_source")

    data_asset = data_source.add_dataframe_asset(name="features")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("features_batch")

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="olist_features_suite")

    for column in NUMERIC_FEATURES:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeOfType(
                column=column,
                type_="float64",
            )
        )

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

    for column, allowed_values in CATEGORICAL_FEATURES.items():
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=allowed_values,
            )
        )

    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite,
    )

    results = validator.validate()

    return results.success
