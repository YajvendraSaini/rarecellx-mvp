# Use python 3.10 slim image for a much smaller footprint
FROM python:3.10-slim

# Prevent python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run expects the server to listen on port 8080 by default
ENV PORT=8080

# Set our application working directory
WORKDIR /app

# Install system dependencies required by scanpy, anndata, and h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements to leverage Docker caching
COPY requirements.txt .

# Optimize PyTorch installation to use the CPU-only version.
# Standard PyTorch brings in 2.5GB+ of CUDA/GPU drivers which are 
# not needed for standard Cloud Run cpu nodes.
RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application files (model, data, and app modules).
# .dockerignore will prevent venv and __pycache__ from being copied.
COPY . .

# Expose the correct cloud run port
EXPOSE 8080

# Command to run uvicorn, reading the PORT env variable correctly.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
