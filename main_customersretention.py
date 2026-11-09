"""Customer Churn API Docker.
Serves the churn model (XGBoost pipeline + decision threshold) over HTTP.

Version history
---------------
1.0.0  2026-11-04  First API: /health, /model-info, /predict (threshold 0.42).
                   Input validation with Literal / Field, unknown fields rejected.
1.1.0  2026-11-06  Optional customerID, returned in the response (not used by the model).
                   MonthlyCharges = 0 accepted (promotional months).
1.2.0  2026-11-07  SQL Server integration: GET /customers/{customer_id}/prediction
                   reads the customer through dbo.usp_GetCustomerFeatures.
                   Connection settings read from .env (no credentials in code).
                   Service account api_user_ChurnDB with EXECUTE-only permissions.
1.3.0  2026-11-07  Batch scoring: POST /batch/score scores all active customers
                   and stores the results in ml.ChurnPredictions (with batch ID,
                   model name and training date for traceability).
                   
"""

import json
import os
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Literal
from datetime import date

import joblib
import pandas as pd
import pyodbc
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# --- Load model artifacts once, at startup ---
PIPELINE_PATH = BASE_DIR / "customersretention_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "customersretention_threshold.json"

pipeline = joblib.load(PIPELINE_PATH)
with open(CONFIG_PATH) as f:
    config = json.load(f)

# --- Database connection: values come from environment variables (.env) ---
DB_CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={os.getenv('DB_SERVER', 'localhost,1433')};"
    f"DATABASE={os.getenv('DB_NAME', 'ChurnDB')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')};"
    "TrustServerCertificate=yes;"
)


def run_procedure(sql, *params):
    """Execute a stored procedure and return its rows as a list of dicts."""
    with closing(pyodbc.connect(DB_CONN_STR, timeout=5)) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


app = FastAPI(
    title="Customer Churn API Docker",
    description="Predicts the probability that a customer will churn.",
    version="1.3.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return config


EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}

# --- Allowed values (see "API Input Contract" in the README) ---
YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customerID: str | None = Field(
        default=None,
        description="Optional. Not used by the model; returned in the response for traceability.",
    )
    gender: Literal["Female", "Male"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(
        ge=0,
        description="Current monthly charge. 0 is accepted (e.g. promotional month), "
                    "but the training data has no customers below ~18, so predictions there are less reliable.",
    )
    TotalCharges: float = Field(ge=0, description="Use 0 for new customers (tenure = 0)")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"examples": [EXAMPLE_CUSTOMER]},
    }


def score(customer: Customer) -> dict:
    """Run the pipeline on one validated customer and apply the decision threshold."""
    features = customer.model_dump(exclude={"customerID"})
    df = pd.DataFrame([features])[config["input_features"]]
    proba = float(pipeline.predict_proba(df)[0, 1])
    return {
        "customerID": customer.customerID,
        "churn_probability": round(proba, 3),
        "contact": proba >= config["threshold"],
        "threshold": config["threshold"],
    }


@app.post("/predict")
def predict(customer: Customer):
    return score(customer)


@app.get("/customers/{customer_id}/prediction")
def predict_customer(customer_id: str):
    try:
        rows = run_procedure("EXEC dbo.usp_GetCustomerFeatures @CustomerID = ?", customer_id)
    except pyodbc.Error:
        raise HTTPException(status_code=503, detail="Database unavailable")

    if not rows:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    try:
        customer = Customer(**rows[0])
    except ValidationError as e:
        raise HTTPException(
            status_code=500,
            detail={"message": "Customer data in the database failed validation",
                    "errors": e.errors(include_url=False, include_context=False)},
        )
    return score(customer)


def save_predictions(records):
    """Insert scored customers into ml.ChurnPredictions in a single round trip."""
    sql = ("INSERT INTO ml.ChurnPredictions "
           "(BatchID, customerID, ChurnProbability, Contact, Threshold, ModelName, ModelCreated) "
           "VALUES (?, ?, ?, ?, ?, ?, ?)")
    with closing(pyodbc.connect(DB_CONN_STR, timeout=5)) as conn:
        cursor = conn.cursor()
        cursor.fast_executemany = True
        cursor.executemany(sql, records)
        conn.commit()


@app.post("/batch/score")
def batch_score():
    """Score all active customers and store the results in ml.ChurnPredictions."""
    start = time.perf_counter()

    try:
        rows = run_procedure("EXEC dbo.usp_GetCustomersForScoring")
    except pyodbc.Error:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Validate every customer; invalid rows are reported, not scored
    valid, invalid_ids = [], []
    for row in rows:
        try:
            valid.append(Customer(**row))
        except ValidationError:
            invalid_ids.append(row.get("customerID"))

    if not valid:
        raise HTTPException(status_code=500, detail="No valid customers to score")

    # Score everyone at once (much faster than one by one)
    df = pd.DataFrame([c.model_dump(exclude={"customerID"}) for c in valid])[config["input_features"]]
    probas = pipeline.predict_proba(df)[:, 1]

    threshold = config["threshold"]
    batch_id = str(uuid.uuid4())
    model_created = date.fromisoformat(config["created"])
    records = [
        (batch_id, c.customerID, float(p), bool(p >= threshold), threshold,
         config["model"], model_created)
        for c, p in zip(valid, probas)
    ]

    try:
        save_predictions(records)
    except pyodbc.Error:
        raise HTTPException(status_code=503, detail="Could not save predictions")

    return {
        "batch_id": batch_id,
        "scored": len(records),
        "to_contact": sum(r[3] for r in records),
        "invalid": len(invalid_ids),
        "invalid_ids": invalid_ids[:20],
        "threshold": threshold,
        "seconds": round(time.perf_counter() - start, 2),
    }    