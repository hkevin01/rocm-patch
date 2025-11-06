# ROCm Patch Repository - Docker Environment
# Base image with ROCm support
FROM rocm/dev-ubuntu-22.04:latest

LABEL maintainer="ROCm Patch Community"
LABEL description="ROCm Patch Repository Development Environment"
LABEL version="0.1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    vim \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install testing and development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    black \
    flake8 \
    pylint \
    isort \
    mypy \
    bandit \
    safety

# Install ROCm Python packages
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/rocm5.7 || true

# Create workspace directory
WORKDIR /workspace/rocm-patch

# Copy project files
COPY . /workspace/rocm-patch/

# Install project in development mode
RUN pip install -e . || true

# Create log directory
RUN mkdir -p /var/log/rocm-patch && chmod 777 /var/log/rocm-patch

# Create backup directory
RUN mkdir -p /var/backups/rocm-patches && chmod 777 /var/backups/rocm-patches

# Set up user (non-root)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} rocmuser || true && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash rocmuser || true && \
    usermod -aG video,render rocmuser || true

# Give permissions
RUN chown -R rocmuser:rocmuser /workspace /opt/venv

# Switch to non-root user
USER rocmuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD rocminfo || exit 1

# Default command
CMD ["/bin/bash"]
