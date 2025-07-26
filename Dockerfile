# Dockerfile
FROM python:3.10-slim

# Set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ben \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy everything
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
