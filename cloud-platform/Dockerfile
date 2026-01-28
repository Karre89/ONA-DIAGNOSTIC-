FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY app/ app/

# Expose port (Railway uses PORT env var)
EXPOSE ${PORT:-8000}

# Run the application - use shell form to expand PORT variable
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
