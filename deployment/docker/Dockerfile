# Production Dockerfile for Spikeformer Neuromorphic Kit
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NEUROMORPHIC_ENV=production

# Create app user
RUN groupadd -r spikeformer && useradd -r -g spikeformer spikeformer

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R spikeformer:spikeformer /app
USER spikeformer

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import spikeformer; print('Health check passed')" || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python3", "-m", "spikeformer.cli.main", "--serve", "--port", "8080"]
