# Production Dockerfile for TRIP-WIRE (Hugging Face Spaces / Render / Railway)
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and FFmpeg video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/.cache && \
    chown -R appuser:appuser /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Copy codebase
COPY --chown=appuser:appuser . /app

# Pre-download YOLOv8n weights so the container starts instantly offline
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8501 8000

# Default command: launch Streamlit Application
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
