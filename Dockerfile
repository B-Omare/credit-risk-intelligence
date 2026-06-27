FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition first (for layer caching)
COPY pyproject.toml .
COPY creditpulse/ ./creditpulse/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    xgboost==2.1.4 \
    scikit-learn \
    joblib \
    pyarrow \
    pydantic \
    httpx

# Copy the rest of the project
COPY . .

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "creditpulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]