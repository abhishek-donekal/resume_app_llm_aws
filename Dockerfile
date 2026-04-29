# Multi-stage Dockerfile for LLaMA 2 Resume Customizer API

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

LABEL maintainer="your-email@example.com"
LABEL description="LLaMA 2 Resume Customizer API"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data/raw /app/data/processed /app/model_output /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_HOME=/app/model_cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "-m", "uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
