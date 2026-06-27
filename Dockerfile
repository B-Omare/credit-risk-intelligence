FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package definition first (layer caching)
COPY pyproject.toml .
COPY creditpulse/__init__.py creditpulse/

# Install base + API dependencies
RUN pip install --no-cache-dir -e ".[api]"

# Copy the rest of the source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "creditpulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
