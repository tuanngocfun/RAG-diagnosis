# CUDA + cuDNN + Python with PyTorch 2.6.0 (CUDA 12.4) – GPU ready
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ---- OS deps for OCR / CV / PDF ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng curl \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      poppler-utils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (cache-friendly, but easy to override) ----
# You can point REQUIREMENTS_FILE to another file via --build-arg
ARG REQUIREMENTS_FILE=requirements_recovered.txt
COPY ${REQUIREMENTS_FILE} /tmp/requirements.txt

# Optional: use this to invalidate the install layer without editing files
ARG CACHE_BUST=0

# Optional: override a few packages without touching the base file
# ex: --build-arg 'EXTRA_OVERRIDES=transformers==4.54.0 tokenizers==0.21.0'
ARG EXTRA_OVERRIDES=""

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir google-genai==0.6.0 \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && if [ -n "$EXTRA_OVERRIDES" ]; then \
        python -m pip install --no-cache-dir $EXTRA_OVERRIDES ; \
    fi

# ---- Offline-friendly HF defaults (safe even if online) ----
ENV TRANSFORMERS_CACHE=/data4t/hf/transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---- Copy your code last to avoid invalidating pip layer ----
COPY . /app

# (Optional) show Tesseract at container start:
# CMD ["bash", "-lc", "tesseract --version && bash"]

# Single CMD only (removes the warning)
CMD ["bash"]
