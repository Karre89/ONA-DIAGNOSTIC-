FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create data directories
RUN mkdir -p /var/imaging-edge/incoming \
    /var/imaging-edge/processed \
    /var/imaging-edge/results \
    /var/imaging-edge/heatmaps \
    /var/imaging-edge/models \
    /var/imaging-edge/samples

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY app/ app/

# Expose ports (8080 for web UI, 104 for DICOM SCP)
EXPOSE 8080 104

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
