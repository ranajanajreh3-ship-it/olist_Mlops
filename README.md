#Olist MLOps Project

## Overview

This project builds an end-to-end machine learning and MLOps pipeline for predicting whether an Olist e-commerce order will be delivered late or on time.

The project was developed as part of the Qafzah MLOps Training Program.

The project covers the complete workflow from data preparation and model development to deployment, experiment tracking, testing, CI/CD, logging, and monitoring.

## Project Structure

```text
olist_Mlops/

├── app/                         # FastAPI application
├── config/                      # Project configuration
├── data/                        # Datasets and DVC metadata
├── models/                      # Model and preprocessing artifacts
├── notebooks/                   # Jupyter notebooks
├── requirements/                # Project dependencies
│   ├── requirements.txt
│   ├── requirements-runtime.txt
│   └── requirements-dev.txt
├── src/                         # Reusable Python modules
├── tests/                       # Automated tests
├── logs/                        # Prediction logs
├── Dockerfile                   # FastAPI Docker image
├── Dockerfile.mlflow            # MLflow Docker image
├── docker-compose.yml           # Multi-service Docker setup
├── .env.example                 # Environment variable template
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── CI_CD.yml            # GitHub Actions workflow
```

## Dataset

The project uses the Brazilian E-Commerce Public Dataset by Olist.

The dataset contains information about orders, customers, sellers, products, payments, and deliveries.

## Machine Learning Task

The target variable is:

```text
delivery_status
```

The model predicts whether an order is:

- `late`
- `on time`

A time-based train/validation/test split is used to reduce data leakage and better represent real-world prediction.

## Model

The current selected model is a tuned Random Forest classifier.

The prediction threshold is:

```text
0.4
```

The trained model and preprocessing objects are stored in the `models/` directory.

Main model artifacts:

```text
models/final_random_forest.pkl
models/transformer.pkl
models/final_threshold.pkl
```

## Configuration

Project paths and parameters are stored in:

```text
config/config.yaml
```

This avoids hardcoding paths and important parameters inside the Python code.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```powershell
pip install -r requirements/requirements.txt
```

For development and testing:

```powershell
pip install -r requirements/requirements-dev.txt
```

## Running the Project

The machine learning workflow is developed and validated through the notebooks in:

```text
notebooks/
```

The notebooks cover:

1. Data preparation
2. Label creation
3. Train/validation/test splitting
4. Exploratory data analysis
5. Feature engineering
6. Model training, tuning, and evaluation

The reusable Python modules are located in:

```text
src/
```

## Features

### Numerical Features

```text
total_items
total_price
total_freight_value
total_payment
max_installments
payment_count
distance_km
```

### Categorical Features

```text
month_name
day_name
customer_state
seller_state
```

## MLOps Components

The project includes:

- Configuration management
- Data and artifact versioning with DVC
- Data validation with Great Expectations
- Experiment tracking and model registry with MLflow
- Automated testing with pytest
- FastAPI prediction service
- Docker and Docker Compose
- CI/CD with GitHub Actions
- Prediction logging
- Monitoring

## Data Versioning & Validation

### Data and Artifact Versioning

DVC (Data Version Control) is used to track the datasets and trained model artifacts used by the project.

The following files are tracked using DVC:

- `data/train.csv`
- `data/validation.csv`
- `data/test.csv`
- `models/final_random_forest.pkl`
- `models/transformer.pkl`
- `models/final_threshold.pkl`

The DVC remote is hosted on DagsHub.

### Data Validation

Great Expectations is used to validate incoming model features before they reach the preprocessing and prediction stages.

The validation checks include:

- Required columns exist.
- Numeric columns have the expected data type.
- Numeric columns do not contain missing values.
- Numeric values satisfy the defined minimum ranges.
- Categorical columns do not contain missing values.
- Categorical values belong to the allowed set of values.

### Validation Failure Policy

The service uses a **Reject** policy when validation fails.

If the input data does not satisfy the defined expectations, the request is rejected and the model is not executed.

The prediction flow is:

`Input → Great Expectations Validation → Reject if Invalid → Preprocessing → Model → Prediction`

## MLflow

MLflow is used for:

- Experiment tracking
- Model management
- Model registry
- Model versioning
- Model aliases

The registered model is:

```text
OlistDeliveryModel
```

The production alias is:

```text
champion
```

MLflow uses PostgreSQL as the backend store and MinIO for artifact storage.

## Docker & Docker Compose

The project is containerized using Docker and Docker Compose.

The Docker setup includes the following services:

- **API** — FastAPI prediction service.
- **PostgreSQL** — MLflow backend store.
- **MLflow** — Experiment tracking and model registry.
- **MinIO** — S3-compatible artifact storage.

### Dockerfile

The API uses a lightweight `python:3.13-slim` base image.

The API image contains only the files required at runtime:

- `app/`
- `src/`
- `config/`
- `models/`
- `requirements/requirements-runtime.txt`

Jupyter notebooks and project datasets are not copied into the API image.

### Environment Variables

Sensitive credentials and connection settings are stored in a local `.env` file and are not committed to Git.

An `.env.example` file is provided as a template:

```text
POSTGRES_PASSWORD=your_postgres_password
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
```

The `.env` file is included in `.gitignore`.

### Running with Docker Compose

The complete system can be started with:

```powershell
docker compose up -d --build
```

Docker Compose starts PostgreSQL, MinIO, MLflow, the MLflow initialization service, and the FastAPI service.

MLflow includes a health check, and the API waits for MLflow to become healthy before starting.

### API

Once the containers are running, the FastAPI Swagger documentation is available at:

```text
http://localhost:8000/docs
```

The API health endpoint can be checked at:

```text
http://localhost:8000/health
```

The model information endpoint is available at:

```text
http://localhost:8000/model-info
```

The monitoring metrics endpoint is available at:

```text
http://localhost:8000/metrics
```

### MLflow

MLflow is available at:

```text
http://localhost:5000
```

The MLflow backend uses PostgreSQL, while model artifacts are stored in MinIO.

The MLflow initialization service automatically checks whether the registered model `OlistDeliveryModel` has the `champion` alias. If the model is already registered, no duplicate model version is created.

### Service Ports

| Service       | Port |
|---------------|-----:|
| FastAPI       | 8000 |
| MLflow        | 5000 |
| MinIO API     | 9000 |
| MinIO Console | 9001 |
| PostgreSQL    | 5432 |

## Testing

Automated tests are implemented using pytest.

Run the test suite with:

```powershell
PYTHONPATH=. pytest
```

The project test suite covers the main application and prediction functionality.

## Code Quality

The project uses Ruff for code quality and formatting checks.

Run Ruff with:

```powershell
ruff check .
```

Run Ruff formatting check with:

```powershell
ruff format --check .
```

Pre-commit hooks are also configured for automated code-quality checks.

Run all pre-commit hooks with:

```powershell
pre-commit run --all-files
```

## CI/CD

The project uses **GitHub Actions** for continuous integration and deployment.

The workflow is located at:

```text
.github/workflows/CI_CD.yml
```

The CI/CD pipeline performs the following tasks:

1. Checkout the repository
2. Set up Python
3. Install project dependencies
4. Configure DVC
5. Pull versioned data and model artifacts
6. Run Ruff
7. Check code formatting
8. Run automated tests
9. Build the Docker image
10. Push the Docker image to GitHub Container Registry

The Docker image is published to:

```text
ghcr.io/ranajanajreh3-ship-it/olist-mlops:latest
```

## Logging

The API records prediction information in:

```text
logs/predictions.csv
```

The prediction log contains:

```text
timestamp
request_id
prediction
probability_late
model_version
latency_ms
```

These logs allow predictions to be reviewed later and provide a basis for evaluating predictions against the actual delivery outcome once the real delivery date becomes available.

## Monitoring

The API exposes monitoring metrics through:

```text
http://localhost:8000/metrics
```

Prometheus-compatible metrics include:

### Request Metrics

```text
api_requests_total
```

Tracks the total number of API requests.

### Request Latency

```text
api_request_latency_seconds
```

Tracks API request latency.

### API Errors

```text
api_errors_total
```

Tracks API errors.

### Prediction Distribution

```text
api_predictions_total
```

Tracks the number of predictions for each prediction class.

### Late Prediction Rate

```text
api_late_prediction_rate
```

Tracks the current proportion of predictions classified as `late`.

### Prediction Drift

```text
api_prediction_drift
```

Tracks the absolute difference between the current late-prediction rate and the baseline late-prediction rate.

The baseline late-prediction rate is approximately:

```text
8.7%
```

This provides a simple monitoring mechanism for changes in prediction distribution over time.

## Monitoring and Alerts

The following alert conditions are defined for the prediction service.

### Error Rate Alert

Trigger an alert when the API error rate exceeds:

```text
5%
```

### Latency Alert

Trigger an alert when API request latency exceeds:

```text
1 second
```

### Prediction Drift Alert

Trigger an alert when the absolute difference between the current late-prediction rate and the baseline rate exceeds:

```text
10 percentage points
```

## Security and Secrets

Sensitive credentials are stored using environment variables.

The local `.env` file is excluded from Git using `.gitignore`.

GitHub Actions uses repository secrets for DVC and DagsHub authentication.

## Development Status

The project includes the main components required for an end-to-end MLOps workflow:

- Machine learning model development
- Feature engineering and preprocessing
- Data validation with Great Expectations
- Data and artifact versioning with DVC
- Experiment tracking and model registry with MLflow
- FastAPI prediction service
- Docker and Docker Compose
- Automated testing with pytest
- Code quality checks with Ruff
- Pre-commit hooks
- CI/CD with GitHub Actions
- Docker image publishing to GitHub Container Registry
- Prediction logging
- Service monitoring
- Prediction distribution monitoring
- Prediction drift monitoring
- Defined monitoring alert thresholds

The project is ready for final submission.

## Repository

GitHub repository:

https://github.com/ranajanajreh3-ship-it/olist_Mlops.git
