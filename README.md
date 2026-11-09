# Customer Churn Prediction with Interpretability

Predicting which telecom customers are likely to leave, explaining **why**, choosing
whom to contact based on **business cost** rather than accuracy, and **serving the
model in production**: a FastAPI service connected to SQL Server, packaged with Docker.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-green)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![SQL Server](https://img.shields.io/badge/SQL%20Server-2022-CC2927)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

**Contents:** [Problem](#problem) · [Data](#data) · [EDA](#key-findings-from-the-eda) ·
[Approach](#approach) · [Results](#results) · [Interpretability](#interpretability-readable-rules) ·
[Business threshold](#business-cost-threshold) · [Production deployment](#production-deployment) ·
[Repository](#repository-structure) · [Changelog](#changelog)

## Problem

A telecom company loses about **1 in 4 customers** (26.5% churn). The retention
team can't contact everyone, and a list of "at-risk" customers is not enough:
to act, they need to know **who** is likely to leave, **why**, and **whether
contacting them pays off**.

## Data

[Telco Customer Churn (IBM sample, via Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
7,043 customers, 19 features (demographics, account information and subscribed
services) and a binary target `Churn`.

The dataset is not included in this repository. Download it from Kaggle and place
it in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Key findings from the EDA

Churn is **concentrated, not spread out**:

| Who churns | Share of all churners | Churn rate |
|---|---|---|
| Month-to-month contract | 88.6% | 42.7% |
| Month-to-month, first year | **54.8%** (1,024 customers) | 51.4% |
| Electronic check payment | 57.3% | 45.3% |
| Two-year contract | 2.5% | 2.8% |

**Protection services retain customers.** Among internet customers, churn drops
from 56.7% with no protection/support service (online security, backup, device
protection, tech support) to 5.3% with all four, for both DSL and fiber.

![Churn by number of services](images/02_services_and_protection.png)

**The fiber optic paradox.** Fiber is the most expensive internet service (median
$91.68 vs $56.15 for DSL) and concentrates 69.4% of churners. But within fiber,
customers who pay *more* churn *less*, because higher bills come from protection
services. The real risk is **fiber without add-ons**: 63.6% churn vs 8.4% with all
four protection services.

## Approach

1. **Cleaning:** `TotalCharges` stored as text; 11 blanks belong to new customers
   (`tenure = 0`) and are filled with 0. `SeniorCitizen` (0/1) is mapped to No/Yes.
2. **Validation:** stratified 80/20 train/test split; 5-fold stratified
   cross-validation on the training set; the test set is used only once.
3. **Preprocessing in scikit-learn pipelines** (no leakage, saved with the model):
   duplicated "No internet/phone service" levels removed, an engineered feature
   `n_protection` (both in `features.py`), and two preprocessors:
   - **Linear:** redundant features removed after a VIF check (`MonthlyCharges`
     was ~97% explained by the services), scaling and one-hot with a reference category.
   - **Tree:** all features, no scaling.
4. **Class imbalance:** `class_weight="balanced"` / `scale_pos_weight`, metrics
   suited to imbalance (PR-AUC as the main metric), and a cost-based threshold.
5. **Models:** Logistic Regression (interpretable baseline), Decision Tree
   (readable rules) and XGBoost, tuned with `RandomizedSearchCV` (50 combinations).

## Results

**Cross-validation (training set)**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **XGBoost (tuned)** | **0.845** | **0.662** | **0.805** | 0.517 | **0.629** |
| Logistic Regression | 0.839 | 0.645 | 0.788 | 0.513 | 0.621 |
| XGBoost (default) | 0.813 | 0.608 | 0.654 | 0.545 | 0.594 |
| Decision Tree (depth 4) | 0.822 | 0.588 | 0.763 | 0.504 | 0.606 |
| Dummy (always "no churn") | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 |

- Default XGBoost **overfit heavily** (train PR-AUC 0.979 vs 0.608 on validation).
  Tuning fixed it (0.691 vs 0.662) with very shallow trees (`max_depth=2`,
  476 trees, learning rate 0.03).
- Tuned XGBoost beats Logistic Regression in **all 5 folds** on PR-AUC. The gain is
  small (+0.017), which shows churn is driven mostly by simple, additive effects.

**Test set (1,409 customers, 374 churners, evaluated once)**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **XGBoost (tuned)** | **0.862** | **0.680** | 0.824 | 0.538 | **0.651** |
| Logistic Regression | 0.853 | 0.656 | 0.794 | 0.524 | 0.631 |
| Decision Tree | 0.853 | 0.632 | 0.848 | 0.506 | 0.633 |
| XGBoost (default) | 0.834 | 0.630 | 0.684 | 0.559 | 0.615 |

![ROC and Precision-Recall curves](images/04_roc_pr_curves.png)

## Interpretability: readable rules

A depth-4 Decision Tree finds the same drivers as the EDA without being told:
contract type first, then fiber optic, then tenure. **Four rules capture 72.9% of
all churners** in the training set, all of them month-to-month customers:

| Rule (all month-to-month) | Customers | Churners | Churn rate |
|---|---|---|---|
| Fiber + first ~1.5 years | 693 | 436 | 62.9% |
| Fiber + 1.5 to 4.5 years | 676 | 297 | 43.9% |
| No fiber + first ~6 months, above the basic plan | 406 | 197 | 48.5% |
| Fiber + brand new (total charges ≤ $120) | 187 | 160 | **85.6%** |

![Decision Tree](images/03_decision_tree.png)

## Business-cost threshold

Instead of the default 0.5, the threshold is chosen by **money saved**, using each
customer's monthly charges (illustrative assumptions: $50 per retention offer,
12 months of revenue lost per churner, 30% of contacted churners retained).
It is selected on out-of-fold predictions of the training set and applied once
to the test set.

![Savings by threshold](images/06_savings_by_threshold.png)

| Strategy (test set) | Savings vs doing nothing |
|---|---|
| Do nothing | $0 (lose $336,272) |
| Contact everyone | $30,432 |
| **Tuned XGBoost, threshold 0.42** | **$54,327** (16.2%) |

- **The model nearly doubles the value of a retention campaign** compared with
  contacting everyone.
- At threshold 0.42 it detects **320 of 374 churners (85.6%)**, contacting 657
  customers (precision 48.7%).
- The savings curve is **flat between ~0.35 and 0.50**: the exact threshold
  matters little.
- **The real lever is the offer's success rate:** savings range from 1.2% (10%
  success) to 34.6% (50% success).

<img src="images/07_final_confusion_matrix.png" width="350" alt="Final confusion matrix">

## Error analysis

- **Missed churners look like loyal customers:** median tenure of 39 months
  (vs 7 for detected churners), 55.6% on one- or two-year contracts. Their reasons
  for leaving are not in the dataset.
- **False alarms look like churners** (88.7% month-to-month, median tenure of 16
  months): at-risk customers who stayed, at least for now.
- Improving the model needs **new information** (complaints, service calls, usage),
  not more tuning.

## Recommended actions

1. **Lock in new customers early:** incentives to move month-to-month customers to
   a one-year contract in their first months.
2. **Bundle protection services with fiber:** a free trial of tech support or
   online security for fiber customers without add-ons.
3. **Move electronic check payers to automatic payment.**
4. **Score customers monthly and measure the offer** with a control group, then
   recalibrate the threshold with the real success rate.

## Limitations

- **Association is not causation:** only an A/B test can show whether offering
  protection services or longer contracts *causes* lower churn.
- The cost figures are **illustrative assumptions**, not company data.
- About half of the contacted customers would not churn (precision 48.7%).
- The data is a **static snapshot** with no dates, usage or customer service history.

---

# Production Deployment

The trained pipeline is served by a **FastAPI** service that reads customers from
**SQL Server** through stored procedures, scores them and stores the predictions.
Both services run in **Docker** and start together with **Docker Compose**.

## Architecture

```
                         docker compose up
┌──────────────────────────────────────────────────────────────────────────┐
│  Private network                                                         │
│                                                                          │
│  ┌─────────────────────────────────┐        ┌──────────────────────────┐ │
│  │ api  (FastAPI + uvicorn)        │        │ sqlserver (SQL Server)   │ │
│  │  1. validate (Pydantic)         │  EXEC  │  dbo.Customers (raw)     │ │
│  │  2. predict  (pipeline.joblib)  │ ─────► │  dbo.vw_CustomerFeatures │ │
│  │  3. decide   (threshold 0.42)   │        │  dbo.usp_Get... (2 SPs)  │ │
│  │                                 │ INSERT │  ml.ChurnPredictions     │ │
│  │                                 │ ─────► │  ml.usp_GetRetentionList │ │
│  └───────────────┬─────────────────┘        └────────────┬─────────────┘ │
└──────────────────┼───────────────────────────────────────┼───────────────┘
                   │ :8000                                 │ :1433
       CRM / notebook / browser (/docs)          SSMS (database administration)
```

The calling system only sees the API: it sends a customer ID (or the customer's
data) and receives a JSON response. The model can be retrained or replaced without
changing the callers, as long as the input contract is kept.

## Quick start

**Requirements:** Docker 29+ (Docker Desktop on Windows/Mac), the Telco CSV in `data/`.

```bash
# 1. Configuration (never committed)
cp .env.example .env                  # fill in the passwords

# 2. Start SQL Server and the API
docker volume create sqlserver-data
docker compose up -d
docker compose ps                     # sqlserver: healthy, churn-api: running

# 3. First time only: create the database
#    Run sql/01 to sql/05 in order (SSMS or sqlcmd), then load the data
#    with database_setup.ipynb

# 4. Open the API documentation
#    http://127.0.0.1:8000/docs
```

| Command | Purpose |
|---|---|
| `docker compose up -d` | Start everything (SQL Server first, the API once the database is healthy) |
| `docker compose ps` | Status of both services |
| `docker compose logs api` | API logs |
| `docker compose down` | Stop and remove the containers (data stays in the volume) |
| `docker compose up -d --build` | Rebuild the API image after changing the code or the model |

For development with automatic reload, stop the API container
(`docker compose stop api`) and run `fastapi dev main_customersretention.py`;
it reads the connection settings from `.env` (`DB_SERVER=localhost,1433`).

## API

| Endpoint | Method | Purpose | Responses |
|---|---|---|---|
| `/health` | GET | Liveness check | `200 {"status": "ok"}` |
| `/model-info` | GET | Model name, threshold, input features, test metrics, library versions, training date | `200` |
| `/predict` | POST | Score a customer sent in the request body | `200` · `422` invalid input |
| `/customers/{customer_id}/prediction` | GET | Read a customer from SQL Server and score it | `200` · `404` unknown customer · `500` database data failed validation · `503` database unavailable |
| `/batch/score` | POST | Score all active customers and store the results | `200` summary · `503` database unavailable |

Example response:

```json
{"customerID": "7590-VHVEG", "churn_probability": 0.865, "contact": true, "threshold": 0.42}
```

The API applies the business threshold with `predict_proba()`; it never uses
`predict()`, which would cut at 0.5.

### Input contract

The model expects the data **after** cleaning. The SQL view produces this format
from the raw table; callers of `/predict` must send it directly.

| Field | Type | Allowed values |
|---|---|---|
| `customerID` | str, optional | Not used by the model; returned in the response |
| `tenure` | int | ≥ 0 |
| `MonthlyCharges` | float | ≥ 0 (0 accepted for promotional months; the training data has no customers below ~18, so predictions there are less reliable) |
| `TotalCharges` | float | ≥ 0 (0 for new customers) |
| `SeniorCitizen` | str | `Yes`, `No` |
| `gender` | str | `Female`, `Male` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | str | `Yes`, `No` |
| `MultipleLines` | str | `Yes`, `No`, `No phone service` |
| `InternetService` | str | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | str | `Yes`, `No`, `No internet service` |
| `Contract` | str | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | str | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |

`"No internet service"` and `"No phone service"` are accepted because
`engineer_features` maps them to `"No"` inside the pipeline.

### Why strict validation is mandatory

The pipeline's `OneHotEncoder` uses `handle_unknown="ignore"`: an unknown category
does not raise an error, it **silently changes the prediction**. In testing,
`Contract = "monthly"` instead of `"Month-to-month"` moved the probability from
0.928 to 0.787 with no warning. The API therefore validates every input with
Pydantic **before** it reaches the model, and returns `422` with a message that
names the field and the expected values:

| Rule | Example rejected with 422 |
|---|---|
| `Literal[...]`: only categories seen in training (case-sensitive) | `Contract = "monthly"`, `gender = "female"` |
| Types | `tenure = "two"` |
| `Field(ge=0)` | `tenure = -5` |
| All 19 features required | Missing `Contract` |
| `extra="forbid"` | Unknown field `discount` |

Data read from the database goes through the same validation. In a batch, invalid
rows are reported and skipped instead of stopping the run.

## Database

### Cleaning moved from pandas to SQL

The database stores raw, CRM-style data (`SeniorCitizen` as 0/1, `TotalCharges`
as `NULL` for new customers). A view applies the same cleaning used before
training, so the model receives exactly the format it learned from:

| Cleaning rule | Training (pandas) | Serving (SQL view) |
|---|---|---|
| `SeniorCitizen` 0/1 → No/Yes | `.map({1: "Yes", 0: "No"})` | `CASE SeniorCitizen WHEN 1 THEN 'Yes' ELSE 'No' END` |
| `TotalCharges` missing → 0 | `pd.to_numeric(...).fillna(0)` | `ISNULL(TotalCharges, 0)` |

If the two diverged, the model would receive data different from what it learned
(**training-serving skew**). Keeping the rules in a single view means both
procedures stay consistent.

### Scripts (`sql/`, run in order)

| Script | Creates |
|---|---|
| `01_create_database.sql` | `ChurnDB` |
| `02_create_tables.sql` | `dbo.Customers` (raw CRM-style data) |
| `03_views_and_procedures.sql` | `dbo.vw_CustomerFeatures` (cleaning) and the read procedures `usp_GetCustomerFeatures` (one customer) and `usp_GetCustomersForScoring` (active customers) |
| `04_api_user_ChurnDB.sql` | Service account `api_user_ChurnDB` (set its password to match `DB_PASSWORD` in `.env`) |
| `05_predictions.sql` | Schema `ml`, table `ml.ChurnPredictions`, procedure `ml.usp_GetRetentionList` |
| `06_analysis_queries.sql` | Analysis queries (not part of setup) |

Scripts 01, 03, 04 and 05 are re-runnable (`IF ... IS NULL`, `CREATE OR ALTER`).
Data is stored in the Docker volume `sqlserver-data`, so it survives container removal.

### Security: least privilege

| Account | Used by | Permissions |
|---|---|---|
| `sa` | SQL Server container (administration, health check) | Server administrator |
| Personal admin login | Developer, in SSMS | Creates tables and procedures |
| `api_user_ChurnDB` | The API | `EXECUTE` on the two read procedures and `INSERT` on `ml.ChurnPredictions` only |

The API account cannot read tables directly (`SELECT` on `dbo.Customers` is
denied; it reads only through the procedures, via ownership chaining) and has no
access to other databases. Queries use parameters (`?`), never string
concatenation, to prevent SQL injection.

No credentials are stored in the code, `docker-compose.yml` or the image: they are
read from `.env` at startup (see `.env.example`). Compose passes the `sa` password
only to SQL Server and the `DB_*` settings only to the API.

### Batch scoring and the retention list

`POST /batch/score` reads the active customers (`Churn = 'No'`), validates them,
scores them in a single vectorised call and inserts the results into
`ml.ChurnPredictions` with a **batch ID**, the **model name** and its **training
date** (traceability: every prediction can be tied to the model that produced it).
Timestamps are stored in UTC.

**Result:** 5,174 active customers scored in about 0.3 s; 1,678 flagged for
contact. Repeated runs give identical results.

`ml.usp_GetRetentionList` returns every customer of the latest run, ranked by risk,
with the action (`Contact` / `No action`) and the **expected revenue at risk over
12 months** (probability × monthly charge × 12). Ranking by revenue at risk instead
of probability prioritises different customers, which matters when the retention
budget only covers part of the list.

> These customers belong to the dataset used for training, so these scores
> demonstrate the system end to end; they are not an evaluation of the model.

## Testing

`despliegue_produccion.ipynb` tests the pipeline and the API from a clean kernel.
The same cells run unchanged against the API in development (`fastapi dev`) and in
Docker Compose, and return identical results.

| Section | What it tests | Expected |
|---|---|---|
| 1-3 | Library versions, pipeline loading, allowed categories | Pipeline loads outside the training notebook |
| 4 | Pipeline directly on 3 demo customers (reference values) | 0.928 · 0.026 · 0.197 |
| 5 | The same customers through `POST /predict` | `200`, same probabilities |
| 6 | Invalid inputs (one field wrong per case) | `422` in every case (`assert`) |
| 7 | Optional `customerID`, `MonthlyCharges = 0` | `200`, `customerID` returned |
| 8 | Database endpoints: existing / unknown customer, batch | `200` · `404` · 5,174 scored, 0 invalid |

## Production environment

The model was trained and is served with these exact versions. The `.joblib` file
is a serialized Python object: different versions may fail to load it or silently
change the predictions.

| Component | Version | Purpose |
|---|---|---|
| Python | 3.14.6 | Runtime (image `python:3.14-slim-bookworm`) |
| scikit-learn | 1.9.0 | Pipeline and preprocessing |
| xgboost | 3.4.1 | Model |
| pandas | 3.0.3 | Data handling |
| numpy | 2.4.6 | Arrays |
| joblib | 1.5.3 | Model serialization |
| fastapi | 0.162.0 | API framework |
| pydantic | 2.13.4 | Input validation |
| uvicorn | 0.40.0 | API server |
| pyodbc | 5.3.0 | SQL Server connection |
| python-dotenv | 1.2.1 | Reads settings from `.env` |
| ODBC Driver 18 for SQL Server | 18 | Installed in the API image |
| SQL Server | 2022 (Developer) | Database |
| Docker | 29.9.2 | Containers |

| Model artifact | Value |
|---|---|
| Algorithm | XGBoost (tuned) |
| Pipeline | `customersretention_pipeline.joblib` (feature engineering + preprocessing + model) |
| Configuration | `customersretention_threshold.json` (threshold, input features, metrics, versions) |
| Training date | 2026-11-07 |
| Threshold | 0.42 |
| Test metrics at the threshold | ROC-AUC 0.862 · PR-AUC 0.680 · Recall 0.856 · Precision 0.487 |

`features.py` holds the custom transformation used inside the pipeline. joblib
stores only a reference to it, so it must be available wherever the pipeline is
loaded.

## From local to production

The local SQL Server container stands in for a company database. In a real
environment the API code and the image stay the same; what changes:

| Topic | This project | Production |
|---|---|---|
| Database | Created locally in Docker | Existing server (cloud or fixed IP); procedures maintained by the DBA / data team |
| What is deployed | `docker compose up` | Only the API image, pushed to a registry and pulled by the server or a container service |
| Connection | `.env` file | Same variables (`DB_SERVER=<host>,1433`, ...) from the platform or a secrets manager |
| Certificate | `TrustServerCertificate=yes` (self-signed) | Validate the server certificate |
| API account | `api_user_ChurnDB` | Service account with the same least-privilege grants |
| Predictions | `ml` schema | Dedicated schema agreed with the DBA |
| Batch trigger | `POST /batch/score` called manually | Scheduled job calling the same endpoint |
| Data | Public Telco dataset | Development or test copy with personal data masked; never tested against production |

Two common failure points: existing procedures returning different column names or
codes than the model expects (Pydantic validation catches it), and cleaning rules
diverging between training and serving (training data should be extracted with the
same procedure the API uses).

---

## Repository structure

```
├── data/                                       # dataset (not included, see Data)
├── images/                                     # charts used in this README
├── sql/                                        # database setup scripts (01-05) and analysis queries (06)
├── customersretention.ipynb                    # training: EDA, models, threshold, error analysis
├── despliegue_produccion.ipynb                 # deployment tests (pipeline, API, database)
├── database_setup.ipynb                        # loads the CSV into SQL Server
├── features.py                                 # feature engineering used inside the pipeline
├── customersretention_pipeline.joblib          # trained pipeline (used by the API)
├── customersretention_threshold.json           # threshold, input features, metrics, versions
├── main_customersretention.py                  # FastAPI service
├── Dockerfile_customersretention               # API image (Python, libraries, ODBC driver)
├── Dockerfile_customersretention.dockerignore
├── docker-compose.yml                          # API + SQL Server
├── requirements.txt                            # notebooks (training and analysis)
├── requirements_customersretention.txt         # API (pinned versions)
├── .env.example                                # configuration template (copy to .env)
└── README.md
```

## How to reproduce the training

```bash
git clone https://github.com/silvigattino/customers-retention.git
cd customers-retention
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook customersretention.ipynb
```

Running the notebook end to end trains the models and saves the pipeline and its
configuration. After retraining, rebuild the API image with
`docker compose up -d --build`.

## Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-11-04 | API with `/health`, `/model-info`, `/predict` (threshold 0.42); strict input validation |
| 1.1.0 | 2026-11-06 | Optional `customerID` returned in the response; `MonthlyCharges = 0` accepted; Docker image |
| 1.2.0 | 2026-11-07 | SQL Server integration: `GET /customers/{customer_id}/prediction`; settings from `.env`; least-privilege service account |
| 1.3.0 | 2026-11-07 | Batch scoring (`POST /batch/score`) with traceability in `ml.ChurnPredictions`; retention list; Docker Compose for the API and SQL Server |

## Next steps

- Explain individual predictions with **SHAP** and compare the drivers with business knowledge.
- Add customer service and usage data to catch "loyal-looking" churners.
- Make `TrustServerCertificate` configurable to validate certificates in production.
- Publish the API image to a container registry.
- Monitor the score distribution between batch runs to detect data drift.
