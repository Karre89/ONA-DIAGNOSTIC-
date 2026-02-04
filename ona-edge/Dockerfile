# ONA Edge Docker Image
# Supports both NVIDIA Jetson and Intel/AMD CPUs

# For NVIDIA Jetson, use: FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
# For CPU/Intel NUC, use the default Python image
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/scans /data/models /data/logs

# Set environment variables
ENV ONA_DATA_DIR=/data
ENV ONA_DEVICE=cpu
ENV PYTHONUNBUFFERED=1

# Expose ports
# 11112 = DICOM
# 8080 = Web UI
EXPOSE 11112 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Run application
CMD ["python", "main.py"]
