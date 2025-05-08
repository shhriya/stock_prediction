#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install --no-cache-dir -r /requirements.txt


while ! nc -z postgres 5432; do
  echo "Waiting for postgres..."
  sleep 2
done

echo "Initializing Airflow DB..."
airflow db upgrade

echo "Creating default user (if not exists)..."
airflow users create \
  --username admin \
  --firstname Airflow \
  --lastname Admin \
  --role Admin \
  --email admin@example.com \
  --password admin || true

echo "Starting Airflow scheduler in background..."
airflow scheduler &

echo "Starting Airflow webserver..."
exec airflow webserver
