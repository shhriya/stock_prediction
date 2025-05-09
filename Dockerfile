# # Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# Expose the default Streamlit port (but Render auto-detects PORT)
EXPOSE 8501

# Start Streamlit, binding to host 0.0.0.0 and port from Render
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
