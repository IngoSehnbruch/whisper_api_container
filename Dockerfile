FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install dependencies in stages for better visibility
COPY requirements*.txt ./

# 1. Install PyTorch (CPU version)
RUN echo "Installing PyTorch..." && \
    pip install --no-cache-dir -r requirements.torch.txt

# 2. Install Whisper and its dependencies
RUN echo "Installing Whisper..." && \
    pip install --no-cache-dir -r requirements.whisper.txt

# 3. Install other requirements
RUN echo "Installing other dependencies..." && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 whisper

# Create cache directory with proper permissions
RUN mkdir -p /app/cache && chown -R whisper:whisper /app

# Switch to non-root user
USER whisper

# Copy application code
COPY --chown=whisper:whisper ./app /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_MEMORY=8192
ENV MAX_CONCURRENT=5
ENV DEFAULT_MODEL=tiny
ENV USE_CUDA=false
ENV TRUSTED_PROXIES=""
ENV WYOMING_HOST=0.0.0.0
ENV WYOMING_PORT=10300

# Expose ports
EXPOSE 8000
EXPOSE 10300

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
