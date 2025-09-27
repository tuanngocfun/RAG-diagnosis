# Medical RAG Chatbot - Complete Development Environment
# GPU base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# System deps + Node 18 (no inline comments on continued lines)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    wget \
    sudo \
    poppler-utils \
    tesseract-ocr \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
 && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Faster, safer Python bootstrap
RUN pip install --upgrade pip setuptools wheel

# (optional) constraints to pin httpx for google-genai
# If you don't actually have constraints.txt, delete these two lines
COPY constraints.txt /app/constraints.txt

# Install main requirements
COPY requirements_recovered.txt /app/requirements_recovered.txt
RUN pip install --no-cache-dir -r /app/requirements_recovered.txt -c /app/constraints.txt

# Install app-specific requirements
# NOTE: if your repo path is app/server/requirements.txt, this COPY is correct.
# If your server lives at server/requirements.txt, change the path accordingly.
COPY app/server/requirements.txt /app/app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/app/server/requirements.txt -c /app/constraints.txt

# Copy project
COPY . /app/

# Frontend deps
WORKDIR /app/app/web
RUN npm install

# Back to root; correct, concrete PYTHONPATH
WORKDIR /app
ENV PYTHONPATH="/app"

EXPOSE 8000 3000

# IMPORTANT:
# If your FastAPI app file is /app/app/server/main.py with `app = FastAPI(...)`,
# use "app.server.main:app". If it is /app/server/main.py, use "server.main:app".
CMD ["uvicorn", "app.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
