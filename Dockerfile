# Multi-stage build for SpikeFormer Neuromorphic Kit
# Stage 1: Base Python environment
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r spikeformer && useradd -r -g spikeformer spikeformer

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./
COPY spikeformer/ ./spikeformer/

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook

# Copy development files
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY .pre-commit-config.yaml ./

# Setup pre-commit
RUN pre-commit install

# Change ownership
RUN chown -R spikeformer:spikeformer /app

USER spikeformer

# Expose ports for Jupyter and development servers
EXPOSE 8888 8000 3000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 3: Production base
FROM base as production-base

# Install only production dependencies
RUN pip install -e ".[all]" --no-dev

# Copy only necessary files
COPY spikeformer/ ./spikeformer/
COPY README.md LICENSE ./

# Change ownership
RUN chown -R spikeformer:spikeformer /app

USER spikeformer

# Stage 4: CPU-only production
FROM production-base as production-cpu

# Remove GPU-specific packages to reduce size
RUN pip uninstall -y torch torchvision && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["spikeformer"]

# Stage 5: GPU production
FROM production-base as production-gpu

# Keep GPU-enabled PyTorch
# Note: Base image already has CUDA-enabled PyTorch

ENTRYPOINT ["spikeformer"]

# Stage 6: Loihi2 specialized image
FROM production-base as loihi2

# Install Intel NxSDK (mock installation - real would require Intel SDK)
RUN mkdir -p /opt/nxsdk && \
    echo "# Intel NxSDK placeholder" > /opt/nxsdk/README.md

ENV NXSDK_ROOT=/opt/nxsdk
ENV PATH=$PATH:/opt/nxsdk/bin

# Install Loihi2 specific dependencies
RUN pip install ".[loihi2]"

ENTRYPOINT ["spikeformer", "--backend", "loihi2"]

# Stage 7: SpiNNaker specialized image
FROM production-base as spinnaker

# Install SpiNNaker dependencies
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SpiNNaker specific dependencies
RUN pip install ".[spinnaker]"

ENV SPINN_DIRS=/opt/spinnaker

ENTRYPOINT ["spikeformer", "--backend", "spinnaker"]

# Stage 8: Edge deployment image
FROM python:3.13-alpine as edge

# Install minimal dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers

WORKDIR /app

# Copy only essential files
COPY --from=production-base /app/spikeformer ./spikeformer
COPY requirements.txt ./

# Install minimal dependencies for edge deployment
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[edge]"

# Create non-root user
RUN addgroup -S spikeformer && adduser -S -G spikeformer spikeformer
RUN chown -R spikeformer:spikeformer /app

USER spikeformer

ENTRYPOINT ["spikeformer", "--optimize-for", "edge"]