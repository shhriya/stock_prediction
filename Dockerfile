# # # Use an official Python image
# FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y gcc

# # Copy dependency file
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app code
# COPY . .

# # Expose the default Streamlit port (but Render auto-detects PORT)
# EXPOSE 8501

# # Start Streamlit, binding to host 0.0.0.0 and port from Render
# CMD ["streamlit", "run", "scripts/app.py", "--server.port=10000", "--server.address=0.0.0.0"]


# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create directory for credentials
RUN mkdir -p /app/credentials

# Expose the default Streamlit port (Render auto-detects it anyway)
EXPOSE 8501

# Write the credentials JSON from environment variable and start Streamlit
CMD bash -c "\
  mkdir -p /app/credentials && \
  echo \"$GCP_CREDENTIAL_JSON\" > /app/credentials/credentials.json && \
  export GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json && \
  streamlit run scripts/app.py --server.port=10000 --server.address=0.0.0.0"
