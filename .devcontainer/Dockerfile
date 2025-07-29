# Development container for Spikeformer Neuromorphic Kit
ARG PYTHON_VERSION=3.11
ARG NODE_VERSION=18

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # Build tools
        build-essential \
        cmake \
        pkg-config \
        # Graphics and display
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Hardware interface tools
        usbutils \
        pciutils \
        lsb-release \
        # Networking and utilities
        curl \
        wget \
        git \
        htop \
        vim \
        nano \
        tree \
        jq \
        unzip \
        # Database clients
        postgresql-client \
        redis-tools \
        # Monitoring tools
        procps \
        lsof \
        strace \
        # Development tools
        gdb \
        valgrind \
        # LaTeX for documentation
        texlive-latex-base \
        texlive-fonts-recommended \
        pandoc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Node.js (for tooling)
ARG NODE_VERSION
RUN su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"

# Install Python dependencies for development
COPY requirements.txt /tmp/requirements.txt
COPY requirements.md /tmp/requirements.md
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    # Code quality
    black \
    isort \
    ruff \
    mypy \
    bandit \
    safety \
    # Testing
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    pytest-mock \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    sphinx-autobuild \
    myst-parser \
    # Jupyter
    jupyterlab \
    jupyter \
    ipywidgets \
    # ML/Data Science
    matplotlib \
    seaborn \
    plotly \
    tensorboard \
    # Development utilities
    pre-commit \
    commitizen \
    # Container security scanning
    trivy \
    # Performance profiling
    py-spy \
    memory-profiler \
    line-profiler

# Install specialized neuromorphic tools (if available)
RUN pip install --no-cache-dir \
    # Mock packages for development (replace with actual when available)
    # nxsdk \
    # spynnaker \
    # These would be hardware-specific and may not be publicly available
    || echo "Neuromorphic hardware SDKs not available in this environment"

# Install container tools
RUN curl -fsSL https://get.docker.com | sh \
    && usermod -aG docker vscode

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

# Install Helm
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install security scanning tools
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Install hadolint for Dockerfile linting
RUN wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64 \
    && chmod +x /usr/local/bin/hadolint

# Setup development environment
WORKDIR /workspace

# Create cache directories
RUN mkdir -p /home/vscode/.cache/pip \
    && mkdir -p /home/vscode/.cache/mypy \
    && mkdir -p /home/vscode/.cache/pytest \
    && chown -R vscode:vscode /home/vscode/.cache

# Configure git for development
RUN git config --system --add safe.directory /workspace

# Set up Python path and environment
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install oh-my-zsh for better terminal experience
RUN su vscode -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Configure oh-my-zsh
RUN su vscode -c "echo 'export PYTHONPATH=/workspace' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'export PATH=/workspace/scripts:\$PATH' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'alias ll=\"ls -alF\"' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'alias la=\"ls -A\"' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'alias l=\"ls -CF\"' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'alias pytest=\"python -m pytest\"' >> /home/vscode/.zshrc" \
    && su vscode -c "echo 'alias pip-upgrade=\"pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U\"' >> /home/vscode/.zshrc"

# Create development shortcuts
RUN echo '#!/bin/bash\ncd /workspace && python -m pytest tests/ -v' > /usr/local/bin/test-all \
    && chmod +x /usr/local/bin/test-all

RUN echo '#!/bin/bash\ncd /workspace && python -m pytest tests/unit/ -v' > /usr/local/bin/test-unit \
    && chmod +x /usr/local/bin/test-unit

RUN echo '#!/bin/bash\ncd /workspace && python -m pytest tests/integration/ -v' > /usr/local/bin/test-integration \
    && chmod +x /usr/local/bin/test-integration

RUN echo '#!/bin/bash\ncd /workspace && black . && isort . && ruff check . --fix' > /usr/local/bin/format-code \
    && chmod +x /usr/local/bin/format-code

RUN echo '#!/bin/bash\ncd /workspace && mypy spikeformer/' > /usr/local/bin/type-check \
    && chmod +x /usr/local/bin/type-check

# Final setup
USER vscode
WORKDIR /workspace

# Set default shell to zsh
ENV SHELL=/bin/zsh