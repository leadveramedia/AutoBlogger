FROM python:3.11-slim

# Install ffmpeg and rclone (same as GitHub Actions workflow)
RUN apt-get update && apt-get install -y ffmpeg curl unzip && rm -rf /var/lib/apt/lists/*
RUN curl https://rclone.org/install.sh | bash

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD gunicorn web_app:app --workers 1 --threads 8 --timeout 600
