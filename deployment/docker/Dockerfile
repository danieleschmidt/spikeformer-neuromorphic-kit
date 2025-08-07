# Multi-stage Dockerfile for Neuromorphic Computing Production Deployment

# Stage 1: Build stage with development dependencies
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Build the package
RUN pip install -e .

# Run tests during build (optional, can be commented out for faster builds)
# RUN python -m pytest tests/ -xvs

# Stage 2: Production runtime stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEUROMORPHIC_ENVIRONMENT=production \
    NEUROMORPHIC_PORT=8080 \
    NEUROMORPHIC_HOST=0.0.0.0

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    libhdf5-103 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r neuromorphic && useradd -r -g neuromorphic neuromorphic

# Create app directory and set ownership
WORKDIR /app
RUN chown neuromorphic:neuromorphic /app

# Copy built application from builder stage
COPY --from=builder --chown=neuromorphic:neuromorphic /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder --chown=neuromorphic:neuromorphic /usr/local/bin/ /usr/local/bin/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/ ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/deployment/ ./deployment/

# Copy configuration files
COPY --chown=neuromorphic:neuromorphic deployment/config/ ./config/
COPY --chown=neuromorphic:neuromorphic deployment/scripts/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/models /app/cache && \
    chown -R neuromorphic:neuromorphic /app/logs /app/data /app/models /app/cache

# Switch to non-root user
USER neuromorphic

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["serve"]

# Stage 3: Development stage (for local development)
FROM builder as development

# Install additional development tools
RUN pip install jupyter ipython ipdb

# Set development environment
ENV NEUROMORPHIC_ENVIRONMENT=development \
    NEUROMORPHIC_LOG_LEVEL=DEBUG

# Expose additional ports for development
EXPOSE 8080 8888 6006

# Use root for development convenience
USER root

# Override entrypoint for development
ENTRYPOINT ["python"]
CMD ["-m", "spikeformer.cli"]

# Stage 4: Edge deployment stage (minimal footprint)
FROM python:3.11-alpine as edge

# Set environment variables for edge
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEUROMORPHIC_ENVIRONMENT=edge \
    NEUROMORPHIC_PORT=8080 \
    NEUROMORPHIC_MAX_MEMORY=1GB

# Install minimal runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    openblas \
    libgomp

# Create app user
RUN addgroup -g 1001 neuromorphic && \
    adduser -D -u 1001 -G neuromorphic neuromorphic

# Create minimal app structure
WORKDIR /app
RUN chown neuromorphic:neuromorphic /app

# Copy only essential files for edge deployment
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/models.py ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/neurons.py ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/encoding.py ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/hardware.py ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/spikeformer/__init__.py ./spikeformer/
COPY --from=builder --chown=neuromorphic:neuromorphic /app/deployment/edge_server.py ./server.py

# Install minimal Python dependencies
RUN pip install --no-cache-dir torch numpy

USER neuromorphic

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["python", "server.py"]

# Stage 5: Research stage (with all development tools)
FROM builder as research

# Install additional research dependencies
RUN pip install \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb \
    mlflow \
    optuna \
    ray \
    dask \
    plotly \
    seaborn \
    scikit-learn

# Set research environment
ENV NEUROMORPHIC_ENVIRONMENT=research \
    NEUROMORPHIC_LOG_LEVEL=DEBUG \
    JUPYTER_ENABLE_LAB=yes

# Create research directories
RUN mkdir -p /app/experiments /app/datasets /app/notebooks /app/results && \
    chown -R neuromorphic:neuromorphic /app/experiments /app/datasets /app/notebooks /app/results

# Copy research utilities
COPY --chown=neuromorphic:neuromorphic deployment/research/ ./research/
COPY --chown=neuromorphic:neuromorphic notebooks/ ./notebooks/

USER neuromorphic

# Expose research ports
EXPOSE 8080 8888 6006 4040 8265

# Research-specific health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health && curl -f http://localhost:8888/ || exit 1

# Default to Jupyter Lab for research
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]