# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package files
COPY pyproject.toml .
COPY acp_mcp_server/ ./acp_mcp_server/

# Install the package in development mode
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default environment variables
ENV ACP_BASE_URL=http://localhost:8000

# Health check endpoint (works for HTTP transports)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Default entrypoint using the installed package
ENTRYPOINT ["python", "-m", "acp_mcp_server"]

# Default command (can be overridden)
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
