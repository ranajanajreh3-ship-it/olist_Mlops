\# Olist MLOps Project



\## Overview



This project builds a machine learning pipeline for predicting whether an Olist e-commerce order will be delivered late or on time.



The project is being developed as part of the Qafza MLOps training program.



\## Project Structure



```text

olist\_Mlops/

│

├── app/                    # Application and API

├── config/                 # Project configuration

├── data/                   # Datasets

├── models/                 # Saved models and preprocessing objects

├── notebooks/              # Jupyter notebooks

├── requirements/            # Project dependencies
│   ├── requirements.txt
│   ├── requirements-runtime.txt
│   └── requirements-dev.txt
├── src/                    # Reusable Python modules
├── tests/                  # Tests
├── Dockerfile              # API Docker image
├── Dockerfile.mlflow       # MLflow Docker image
├── docker-compose.yml      # Multi-service Docker stack
├── .env.example            # Environment variable template
├── README.md
└── .gitignore

```



\## Dataset



The project uses the Brazilian E-Commerce Public Dataset by Olist.



The dataset contains information about orders, customers, sellers, products, payments, and deliveries.



\## Machine Learning Task



The target variable is:



```text

delivery\_status

```



The model predicts whether an order is:



\* `late`

\* `on time`



A time-based train/validation/test split is used to reduce data leakage and better represent real-world prediction.



\## Model



The current selected model is a tuned Random Forest classifier.



The prediction threshold is currently:



```text

0.4

```



The trained model and preprocessing objects are stored in the `models/` directory.



\## Configuration



Project paths and parameters are stored in:



```text

config/config.yaml

```



This avoids hardcoding paths and important parameters inside the Python code.



\## Installation



Create and activate a Python virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\activate

```



Install the project dependencies:



```powershell

pip install -r requirements/requirements.txt

```



For development and testing:



```powershell

pip install -r requirements/requirements-dev.txt

```



\## Running the Project



The machine learning workflow is currently developed and validated through the notebooks in:



```text

notebooks/

```



The notebooks cover:



1\. Data preparation

2\. Label creation

3\. Train/validation/test splitting

4\. Exploratory data analysis

5\. Feature engineering

6\. Model training, tuning, and evaluation



The reusable Python modules will be placed in:



```text

src/

```



as the project is refactored into an MLOps pipeline.



\## MLOps Components



The project includes:


\* Configuration management

\* Data versioning with DVC

\* Data validation with Great Expectations

\* Experiment tracking and model registry with MLflow

\* Automated testing with pytest

\* FastAPI prediction service

\* Docker and Docker Compose

\* CI/CD

* Logging
* Monitoring (planned)


## Development Status

The core machine learning pipeline and the main MLOps components have been implemented, including:

* Configuration management
* Data and artifact versioning with DVC
* Data validation with Great Expectations
* Experiment tracking and model registry with MLflow
* Automated testing with pytest
* FastAPI prediction service
* Logging
* Docker and Docker Compose

Further improvements, monitoring, and CI/CD can be added as the project continues to evolve.
## Data Versioning & Validation

### Data and Artifact Versioning

DVC (Data Version Control) is used to track the datasets and trained model artifacts used by the project. This allows each experiment and prediction result to be traced back to the corresponding version of the data and model files.

The following files are tracked using DVC:

* `data/train.csv`
* `data/validation.csv`
* `data/test.csv`
* `models/final_random_forest.pkl`
* `models/transformer.pkl`
* `models/final_threshold.pkl`

### Data Validation

Great Expectations is used to validate incoming model features before they reach the preprocessing and prediction stages.

The validation checks include:

* Required columns exist.
* Numeric columns have the expected data type.
* Numeric columns do not contain missing values.
* Numeric values satisfy the defined minimum ranges.
* Categorical columns do not contain missing values.
* Categorical values belong to the allowed set of values.

### Validation Failure Policy

The service uses a **Reject** policy when validation fails.

If the input data does not satisfy the defined expectations, the request is rejected and the model is not executed.

The prediction flow is:

`Input → Great Expectations Validation → Reject if Invalid → Preprocessing → Model → Prediction`

## Docker & Docker Compose

The project is containerized using Docker and Docker Compose.

The Docker setup includes the following services:

* **API** — FastAPI prediction service.
* **PostgreSQL** — MLflow backend store.
* **MLflow** — Experiment tracking and model registry.
* **MinIO** — S3-compatible artifact storage.

### Dockerfile

The API uses a lightweight `python:3.13-slim` base image.

The API image contains only the files required at runtime:

* `app/`
* `src/`
* `config/`
* `models/`
* `requirements/requirements-runtime.txt`

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

The complete system can be started with a single command:

```powershell
docker compose up -d
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

### MLflow

MLflow is available at:

```text
http://localhost:5000
```

The MLflow backend uses PostgreSQL, while model artifacts are stored in MinIO.

The MLflow initialization service automatically checks whether the registered model `OlistDeliveryModel` has the `champion` alias. If the model is already registered, no duplicate model version is created.

### Service Ports

| Service       | Port |
| ------------- | ---: |
| FastAPI       | 8000 |
| MLflow        | 5000 |
| MinIO API     | 9000 |
| MinIO Console | 9001 |
| PostgreSQL    | 5432 |




