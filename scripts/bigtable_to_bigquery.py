from google.cloud import bigtable
from google.cloud.bigquery import Client, LoadJobConfig, SchemaField
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
# Helper function to transform Bigtable row into dictionary
def bigtable_to_bigquery(row):
    try:
        row_data = {}
        for cf, cols in row.cells.items():
            for col, cell in cols.items():
                key = col.decode()
                value = cell[0].value.decode()
                if key in ['open', 'high', 'low', 'close']:
                    value = float(value)
                row_data[key] = value
        return row_data
    except Exception as e:
        print(f"Error transforming row: {e}")
        return None

def main():
    # Connect to Bigtable
    project_id = os.getenv("PROJECT_ID")
    instance_id = os.getenv("INSTANCE_ID")
    table_id = os.getenv("TABLE_ID")
    dataset_id = os.getenv("DATASET_ID")
    bq_table = os.getenv("BQ_TABLE")

    if not all([project_id, instance_id, table_id, dataset_id, bq_table]):
        raise ValueError("One or more required environment variables are missing.")

    # Connect to Bigtable
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)
    table = instance.table(table_id)
    # Read rows from Bigtable
    rows = table.read_rows()
    data = []

    for row in rows:
        transformed = bigtable_to_bigquery(row)
        if transformed:
            data.append(transformed)

    if not data:
        print("No valid data to insert into BigQuery.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Connect to BigQuery
    bq_client = Client(project=project_id)
    full_table_id = f"{project_id}.{dataset_id}.{bq_table}"

    job_config = LoadJobConfig(
        schema=[
            SchemaField("symbol", "STRING"),
            SchemaField("open", "FLOAT"),
            SchemaField("high", "FLOAT"),
            SchemaField("low", "FLOAT"),
            SchemaField("close", "FLOAT"),
            SchemaField("inserted_at", "TIMESTAMP"),
        ],
        write_disposition="WRITE_APPEND",
    )

    job = bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
    job.result()

    print(f"Loaded {len(df)} rows into {full_table_id}.")

if __name__ == "__main__":
    main()




# from google.cloud import bigtable
# from google.cloud import bigquery
# import pandas as pd
# import os

# # Set credentials path (hardcoded for Docker)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/airflow/credentials/credentials.json"

# # Configuration
# project_id = os.getenv("PROJECT_ID")
# instance_id = os.getenv("INSTANCE_ID")
# table_id = os.getenv("TABLE_ID")
# dataset_id = os.getenv("DATASET_ID")
# bq_table = os.getenv("BQ_TABLE")

# print(f"Project ID: {project_id}")
# print(f"Instance ID: {instance_id}")
# print(f"Table ID: {table_id}")

# # Setup Bigtable client
# bt_client = bigtable.Client(project=project_id, admin=False)
# instance = bt_client.instance(instance_id)
# table = instance.table(table_id)

# # Setup BigQuery client
# bq_client = bigquery.Client(project=project_id)

# # Create dataset if not exists
# dataset_ref = bq_client.dataset(dataset_id)
# try:
#     bq_client.get_dataset(dataset_ref)
# except Exception:
#     dataset = bigquery.Dataset(dataset_ref)
#     bq_client.create_dataset(dataset)
#     print(f"Dataset {dataset_id} created.")

# # Read from Bigtable
# rows = table.read_rows()
# data = []

# for row in rows:
#     row_data = {}
#     for cf, cols in row.cells.items():
#         for col, cell in cols.items():
#             row_data[col.decode()] = cell[0].value.decode()
#     data.append(row_data)

# # Load to BigQuery
# if data:
#     df = pd.DataFrame(data)
#     table_ref = bq_client.dataset(dataset_id).table(bq_table)
#     job = bq_client.load_table_from_dataframe(df, table_ref)
#     job.result()
#     print("Exported data to BigQuery.")
# else:
#     print("No data to export.")
