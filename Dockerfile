# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -c "from transformers import pipeline; \
    pipeline('text-generation', model='Finnish-NLP/Ahma-3B'); \
    pipeline('text-generation', model='gpt2')"

# Ensure that the .env file can be used
ENV PYTHONUNBUFFERED=1

# Run the imapguard.py when the container launches
CMD ["python", "imapguard.py"]
