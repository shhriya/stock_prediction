from google.cloud import bigtable
from google.cloud import bigquery
import pandas as pd
import os

# Set credentials path (hardcoded for Docker)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/airflow/google-credentials.json"

# Configuration
project_id = "stock-pricing-458605"
instance_id = "stock-instance"
table_id = "stock-table"
dataset_id = 'stock_dataset'
bq_table = 'stock_data'

print(f"Project ID: {project_id}")
print(f"Instance ID: {instance_id}")
print(f"Table ID: {table_id}")

# Setup Bigtable client
bt_client = bigtable.Client(project=project_id, admin=False)
instance = bt_client.instance(instance_id)
table = instance.table(table_id)

# Setup BigQuery client
bq_client = bigquery.Client(project=project_id)

# Create dataset if not exists
dataset_ref = bq_client.dataset(dataset_id)
try:
    bq_client.get_dataset(dataset_ref)
except Exception:
    dataset = bigquery.Dataset(dataset_ref)
    bq_client.create_dataset(dataset)
    print(f"Dataset {dataset_id} created.")

# Read from Bigtable
rows = table.read_rows()
data = []

for row in rows:
    row_data = {}
    for cf, cols in row.cells.items():
        for col, cell in cols.items():
            row_data[col.decode()] = cell[0].value.decode()
    data.append(row_data)

# Load to BigQuery
if data:
    df = pd.DataFrame(data)
    table_ref = bq_client.dataset(dataset_id).table(bq_table)
    job = bq_client.load_table_from_dataframe(df, table_ref)
    job.result()
    print("Exported data to BigQuery.")
else:
    print("No data to export.")
