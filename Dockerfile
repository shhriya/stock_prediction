# Use a base image with Python 3.10
FROM apache/airflow:2.8.1-python3.10

USER root

# Install system-level dependencies if needed
RUN apt-get update && apt-get install -y gcc

USER airflow

COPY requirements.txt /

# Install all Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs and scripts
COPY dags/ /opt/airflow/dags/
COPY scripts/ /opt/airflow/scripts/
# COPY credentials/ /opt/airflow/credentials/
