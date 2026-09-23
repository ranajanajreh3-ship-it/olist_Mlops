
# Olist MLOps Project

## Overview

This project builds an end-to-end machine learning and MLOps pipeline for predicting whether an Olist e-commerce order will be delivered **late** or **on time**.

The project was developed as part of the **Qafzah MLOps Training Program**.

The project covers the complete workflow from data preparation and model development to deployment, experiment tracking, testing, CI/CD, logging, and monitoring.

---

## Machine Learning Task

The target variable is:

```text
delivery_statusThe model predicts one of two classes:

late
on time

The project uses a time-based train/validation/test split to reduce data leakage and better represent a real-world prediction scenario.

Prediction Threshold

The final prediction threshold is:

0.4

A prediction is classified as late when the predicted probability of being late is greater than or equal to the threshold.

Dataset

The project uses the Brazilian E-Commerce Public Dataset by Olist.

The dataset contains information related to:

Orders
Customers
Sellers
Products
Payments
Delivery dates

The data was processed and transformed into features suitable for machine learning.

Project Structure
olist_Mlops/

│
├── app/                         # FastAPI application
│
├── config/                      # Project configuration
│
├── data/                        # Datasets and DVC metadata
│
├── models/                      # Model and preprocessing artifacts
│
├── notebooks/                   # Data analysis and ML development
│
├── requirements/                # Project dependencies
│   ├── requirements.txt
│   ├── requirements-runtime.txt
│   └── requirements-dev.txt
│
├── src/                         # Reusable Python modules
│
├── tests/                       # Automated tests
│
├── logs/                        # Prediction logs
│
├── Dockerfile                   # FastAPI Docker image
├── Dockerfile.mlflow            # MLflow Docker image
├── docker-compose.yml           # Multi-service Docker setup
├── .env.example                 # Environment variable template
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── CI_CD.yml            # GitHub Actions workflow
Machine Learning Pipeline

The machine learning workflow was developed through several notebooks covering the main stages of the project:

Data preparation
Label creation
Train/validation/test splitting
Exploratory data analysis
Feature engineering
Model training, tuning, and evaluation

The reusable logic was then refactored into Python modules under:

src/
Features

The final prediction pipeline uses numerical and categorical features related to the order and delivery process.

Numerical Features
total_items
total_price
total_freight_value
total_payment
max_installments
payment_count
distance_km
Categorical Features
month_name
day_name
customer_state
seller_state

The preprocessing pipeline is saved and reused during prediction to ensure that production data is transformed consistently with the training data.

Model

The selected machine learning model is a tuned Random Forest Classifier.

The trained model and preprocessing artifacts are stored under:

models/

Important model artifacts include:

models/final_random_forest.pkl
models/transformer.pkl
models/final_threshold.pkl
Configuration

Project paths and important parameters are stored in:

config/config.yaml

This helps avoid hardcoding important paths and parameters directly inside the Python code.

Data Versioning with DVC

DVC (Data Version Control) is used to track datasets and trained model artifacts.

The following files are tracked using DVC:

data/train.csv
data/validation.csv
data/test.csv

models/final_random_forest.pkl
models/transformer.pkl
models/final_threshold.pkl

DVC allows the project to keep track of the versions of data and model artifacts used throughout development.

The project uses a DVC remote hosted through DagsHub.

Data Validation

Great Expectations is used to validate input features before they reach the preprocessing and prediction stages.

Validation checks include:

Required columns exist
Numeric columns have the expected data type
Numeric columns do not contain missing values
Numeric values satisfy defined minimum ranges
Categorical columns do not contain missing values
Categorical values belong to the allowed set of values
Validation Failure Policy

The service uses a Reject policy when validation fails.

The prediction flow is:

Input
  ↓
Great Expectations Validation
  ↓
Reject if Invalid
  ↓
Preprocessing
  ↓
Model
  ↓
Prediction

Invalid requests are rejected before the model is executed.

MLflow

MLflow is used for:

Experiment tracking
Model management
Model registry
Model versioning
Model aliases

The registered model is:

OlistDeliveryModel

The production model uses the:

champion

alias.

The MLflow backend store uses PostgreSQL, while model artifacts are stored using MinIO as S3-compatible object storage.

FastAPI

The machine learning model is served through a FastAPI application.

The API provides endpoints for:

Health checking
Model information
Single predictions
Batch predictions
Prometheus metrics
API Documentation

Once the service is running:

http://localhost:8000/docs
Health Check
http://localhost:8000/health
Model Information
http://localhost:8000/model-info
Metrics
http://localhost:8000/metrics
Docker and Docker Compose

The project is containerized using Docker and Docker Compose.

The Docker environment includes:

FastAPI
PostgreSQL
MLflow
MinIO

The API image uses:

python:3.13-slim

The API image contains only the files required at runtime:

app/
src/
config/
models/
requirements/requirements-runtime.txt

Jupyter notebooks and development datasets are not copied into the API image.

Environment Variables

Sensitive credentials and connection settings are stored locally in a .env file.

The .env file is not committed to Git.

An example template is provided:

.env.example

Example variables:

POSTGRES_PASSWORD=your_postgres_password
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
Running the Project
1. Clone the Repository
git clone https://github.com/ranajanajreh3-ship-it/olist_Mlops.git
cd olist_Mlops
2. Create the Environment File

Create a .env file based on:

.env.example

Add the required local configuration values.

3. Start the System

The complete Docker environment can be started with:

docker compose up -d --build

Docker Compose starts the required services and their dependencies.

Services

The project uses the following services:

Service	Port
FastAPI	8000
MLflow	5000
MinIO API	9000
MinIO Console	9001
PostgreSQL	5432
MLflow

MLflow is available at:

http://localhost:5000

The MLflow environment uses:

PostgreSQL → Backend Store
MinIO      → Artifact Storage

The MLflow initialization process checks whether the registered model already has the champion alias before creating a new model version.

Testing

The project uses pytest for automated testing.

Run the tests with:

PYTHONPATH=. pytest

The project test suite includes tests for the main application and prediction pipeline.

Code Quality

The project uses:

Ruff
Ruff Format
Pre-commit

Run Ruff:

ruff check .

Run formatting checks:

ruff format --check .

Run all pre-commit hooks:

pre-commit run --all-files
CI/CD

The project uses GitHub Actions for continuous integration and deployment.

The workflow is located at:

.github/workflows/CI_CD.yml

The CI/CD pipeline performs tasks including:

Checkout the repository
Set up Python
Install dependencies
Configure DVC
Pull versioned data and model artifacts
Run Ruff
Check code formatting
Run automated tests
Build the Docker image
Push the Docker image to GitHub Container Registry

The Docker image is published to:

ghcr.io/ranajanajreh3-ship-it/olist_mlops
Logging

The API records prediction information in:

logs/predictions.csv

The prediction log contains information such as:

timestamp
request_id
prediction
probability_late
model_version
latency_ms

These logs allow predictions to be reviewed later and provide a basis for evaluating predictions against the actual delivery outcome once the real delivery date becomes available.

Monitoring

The API exposes monitoring metrics through:

http://localhost:8000/metrics

Prometheus-compatible metrics include:

Request Metrics
api_requests_total

Tracks the total number of API requests.

Request Latency
api_request_latency_seconds

Tracks API request latency.

API Errors
api_errors_total

Tracks API errors.

Prediction Distribution
api_predictions_total

Tracks the number of predictions for each prediction class.

Late Prediction Rate
api_late_prediction_rate

Tracks the current proportion of predictions classified as late.

Prediction Drift
api_prediction_drift

Tracks the absolute difference between the current late-prediction rate and the baseline late-prediction rate.

The baseline late-prediction rate is approximately:

8.7%

This provides a simple monitoring mechanism for changes in prediction distribution over time.

Monitoring and Alerts

The following alert conditions are defined for the prediction service:

Error Rate Alert

Trigger an alert when the API error rate exceeds:

5%
Latency Alert

Trigger an alert when API request latency exceeds:

1 second
Prediction Drift Alert

Trigger an alert when the absolute difference between the current late-prediction rate and the baseline rate exceeds:

10 percentage points

These thresholds provide initial operational monitoring rules for the deployed prediction service.

Security and Secrets

Sensitive credentials are not stored directly in the source code.

The project uses environment variables for sensitive configuration, and the local .env file is excluded from Git through .gitignore.

GitHub Actions uses repository secrets for CI/CD credentials such as DVC/DagsHub authentication.

Repository

The complete project is available on GitHub:

https://github.com/ranajanajreh3-ship-it/olist_Mlops
Project Status

The project currently includes the main components of an end-to-end MLOps workflow:

Machine learning model development
Data preprocessing and feature engineering
Data validation with Great Expectations
Data and artifact versioning with DVC
Experiment tracking with MLflow
Model registry and model aliasing
FastAPI prediction service
Docker and Docker Compose
Automated testing with pytest
Code quality checks with Ruff
Pre-commit hooks
CI/CD with GitHub Actions
Docker image publishing
Prediction logging
Service monitoring
Prediction distribution monitoring
Prediction drift monitoring
Defined monitoring alert thresholds

The project is ready for final submission.
