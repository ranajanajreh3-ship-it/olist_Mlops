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

├── requirements/           # Project dependencies

│   ├── requirements.txt

│   └── requirements-dev.txt

├── src/                    # Reusable Python modules

├── tests/                  # Tests

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



The project will progressively include:



\* Configuration management

\* Data versioning with DVC

\* Data validation with Great Expectations

\* Experiment tracking and model registry with MLflow

\* Automated testing with pytest

\* FastAPI prediction service

\* Docker and Docker Compose

\* CI/CD

\* Logging and monitoring



\## Development Status



The machine learning notebooks and initial model pipeline have been completed.



The remaining MLOps components are being implemented progressively.

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



