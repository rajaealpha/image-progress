FROM python:3.11-slim

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn fastapi


# Copy application code
COPY construction_progress/ ./construction_progress/

# Expose port
EXPOSE 5001

# Run on port 5001
CMD ["python", "-W", "ignore", "-m", "uvicorn", "construction_progress.api:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "warning"]
