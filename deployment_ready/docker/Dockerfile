# Spikeformer Neuromorphic Kit - Production Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SPIKEFORMER_ENV=production

# Create non-root user for security
RUN groupadd -r spikeformer && useradd -r -g spikeformer spikeformer

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R spikeformer:spikeformer /app

# Switch to non-root user
USER spikeformer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from spikeformer.health import health_check; health_check()"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "spikeformer.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
