 # dags/stock_pipeline_dag.py
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2023, 1, 1),
    'catchup': False
}

credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

with DAG(
    dag_id='stock_prediction_pipeline',
    max_active_runs=1,   # Only 1 run at a time
    concurrency=1,
    schedule_interval='30 15 * * 1-5',  # 3:30 PM IST on weekdays
    default_args=default_args,
    description='Stock pipeline: yfinance → Bigtable → BigQuery → model training',
    tags=['stock', 'bigtable', 'bigquery', 'ml'],
) as dag:

    # Step 1: Pull today's stock data from yfinance and write to Bigtable
    run_consumer = BashOperator(
        task_id='run_consumer',
        bash_command='python3 /opt/airflow/scripts/consumer.py'
    )

    # Step 2: Transfer latest Bigtable records to BigQuery
    bigtable_to_bq = BashOperator(
        task_id='bigtable_to_bigquery',
        bash_command=f'export GOOGLE_APPLICATION_CREDENTIALS="{credentials_path}" && python3 /opt/airflow/scripts/bigtable_to_bigquery.py'
    )

    # Step 3: Train forecasting model using BigQuery data
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /opt/airflow/scripts/main.py',
        env={
            'GOOGLE_APPLICATION_CREDENTIALS': credentials_path
        },
        dag=dag,
    )

    # DAG flow: run all steps in sequence
    run_consumer >> bigtable_to_bq >> train_model
