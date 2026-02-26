FROM python:3.11-slim

# Playwright build arg (optional — adds ~400 MB; required only for NYSCEF)
ARG INCLUDE_PLAYWRIGHT=false

# Install system dependencies for image processing and PDF handling
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright + Chromium only when requested (NYSCEF support)
RUN if [ "$INCLUDE_PLAYWRIGHT" = "true" ]; then \
        playwright install chromium --with-deps; \
    fi

# Copy application code
COPY analyzer/ ./analyzer/
COPY profiles/ ./profiles/
COPY manage_users.py ./manage_users.py

# Create data directory for state persistence
RUN mkdir -p /app/data

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the analyzer
CMD ["python", "-m", "analyzer.main"]
