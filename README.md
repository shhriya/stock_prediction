# Stock Forecasting Data Pipeline

This project implements a full-fledged **stock price forecasting pipeline** using daily market data, cloud-based storage and processing, machine learning models, and a web-based frontend for visualizations. It automates data collection, transformation, model training, prediction, and visualization using **Google Cloud**, **Airflow**, **Streamlit**, and **CI/CD with GitHub Actions + Render**.


# Features

-  Daily stock data ingestion using `yfinance`
-  Storage in Google Bigtable and transfer to BigQuery
-  Predictive modeling using SARIMA (time series forecasting)
-  Automated Airflow DAG for orchestration
-  Interactive Streamlit dashboard
-  Full test coverage with `pytest` and GitHub Actions
-  Continuous Deployment on Render

---

# Architecture Overview

graph TD
    A[yfinance (Daily Stock Data)] --> B[Bigtable (Raw Storage)]
    B --> C[BigQuery (Processed Storage)]
    C --> D[SARIMA Model Training & Prediction]
    D --> E[BigQuery (Prediction Results)]
    E --> F[Streamlit Frontend (Visualization)]
    subgraph "Airflow DAG"
        B --> C
        C --> D
    end


stock_project/
├── scripts/
│   ├── app.py                      # Streamlit app
│   ├── consumer.py                # Pulls daily data from yfinance and writes to Bigtable
│   ├── producer.py
│   ├── main.py                    # Bigtable ↔ BigQuery interaction + model training
│   ├── bigtable_to_bigquery.py   # Data preprocessing for modeling
│   ├── test_app.py
│   ├── test_main.py
│   ├── test_consumer.py
│   ├── test_producer.py
│   ├── test_transform.py
├── producer/
│   └── kafka_producer.py
├── consumer/
│   └── bigtable_consumer.py      # [Fixed typo from "bigtanle_consumer.py"]
├── model/
│   └── train_model.py
├── bigtable/
│   ├── bigtable_setup.py
│   └── bigtable_to_bigquery.py
├── bigquery/
│   ├── aggregated_data.sql
│   └── run_stored_procedure.py
├── kafka/                         # Reserved for Kafka config/scripts (if any)
├── historical_data/
│   └── backfill_to_bigquery.py
├── dags/
│   └── stock_pipeline_dag.py     # Airflow DAG to orchestrate the pipeline
├── credentials/
│   └── credentials.json          # (GCP service account - should not be pushed to repo)
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions workflow for CI/CD
├── requirements.txt
├── entrypoint.sh                 # For Airflow setup (automatically runs necessary commands)
├── Dockerfile
├── docker-compose.yaml
├── README.md
├── .gitignore
└── .env



# Airflow DAG Details

- **DAG Name**: `stock_prediction_pipeline`
- **Schedule**: Once daily after market close
- **Tasks**:
  - `run_consumer`: Ingest data from `yfinance` and write to Bigtable
  - `bigtable_to_bigquery`: Move data from Bigtable to BigQuery
  - `train_model`: Run SARIMA model and store predictions in BigQuery

# Testing & CI

- Components tested with `pytest`
- Google Cloud services mocked
- GitHub Actions runs:
  - Install dependencies
  - Run unit tests
  - Report failures and block bad merges

# Deployment (CI/CD)

# Streamlit & Airflow via Render
- Deployed on [Render](https://render.com/)
- Auto-redeploy on `main` branch push via GitHub integration
- Secrets and credentials are stored in Render’s environment variables



[![CI](https://github.com/shhriya/stock_project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shhriya/stock_project/actions)


# Local Setup

1. Clone the repository  
   `git clone https://github.com/shhriya/stock_project.git`

2. Create and activate virtual environment  
   `python -m venv .venv && source .venv/bin/activate`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Add your GCP credentials to `credentials/credentials.json`

5. Run Streamlit app  
   `streamlit run scripts/app.py`



# Author
Shriya K

- GitHub: (https://github.com/shhriya)
- Email: shriyashree0411@gmail.com
