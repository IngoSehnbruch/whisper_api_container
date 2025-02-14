FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 whisper
USER whisper

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=whisper:whisper requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=whisper:whisper ./app /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_MEMORY=8192
ENV MAX_CONCURRENT=5
ENV DEFAULT_MODEL=tiny
ENV USE_CUDA=false
ENV TRUSTED_PROXIES=""

# Create cache directory
RUN mkdir -p /app/cache && chown whisper:whisper /app/cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
