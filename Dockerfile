FROM python:3.11-slim 

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy application files
COPY . .

# Expose the API server and Streamlit UI ports
EXPOSE 8000
EXPOSE 8501

# Default entrypoint runs the FastAPI API server
CMD ["python", "main.py", "--mode", "api"]
