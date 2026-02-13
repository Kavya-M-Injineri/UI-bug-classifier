FROM python:3.10-slim

WORKDIR /app

# Install system deps for TensorFlow and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY train_model.py .
COPY generate_samples.py .

# Copy static frontend
COPY static/ static/

# Copy trained model if it exists
COPY model/ model/

# Create uploads directory
RUN mkdir -p uploads

# Hugging Face Spaces runs on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
