FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/

COPY dags/ /opt/airflow/dags/
COPY scripts/ /opt/airflow/scripts/
CMD ["python3", "scripts/producer.py"]
