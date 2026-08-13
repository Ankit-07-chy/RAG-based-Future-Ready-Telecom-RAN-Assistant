FROM python:3.11-slim 

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN curl --retry 5 --retry-delay 5 -L -o /tmp/torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl https://download.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
    curl --retry 5 --retry-delay 5 -L -o /tmp/torchvision-0.28.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl https://download.pytorch.org/whl/cpu/torchvision-0.28.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
    pip install --no-cache-dir --default-timeout=1000 /tmp/torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl /tmp/torchvision-0.28.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
    rm /tmp/torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl /tmp/torchvision-0.28.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl
RUN pip install --no-cache-dir --default-timeout=1000 --index-url https://pypi.org/simple -r requirements.txt

# Copy application files
COPY . .

# Expose the API server and Streamlit UI ports
EXPOSE 8000
EXPOSE 8501

# Default entrypoint runs the FastAPI API server
CMD ["python", "main.py", "--mode", "api"]
