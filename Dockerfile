FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
#LABEL org.opencontainers.image.authors="Mike Wyant Jr. <kainewynd2@gmail.com>"
#LABEL org.opencontainers.image.ref.name="transcription_assistant"
# TO DO: Figure out how to pass tags
#LABEL org.opencontainers.image.version="0.132"
ARG DEBIAN_FRONTEND=noninteractive
ENV PORT=8000

WORKDIR /app

# NVIDIA Cache workaround (TEMPORARY)
RUN set -eux; \
    grep -R "developer.download.nvidia.com" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
    grep -Rl "developer.download.nvidia.com" /etc/apt 2>/dev/null | while IFS= read -r f; do \
      echo "commenting $f"; \
      sed -i.bak -E 's/^(.*developer\.download\.nvidia\.com.*)$/#\1/' "$f"; \
    done; \
    rm -rf /var/lib/apt/lists/*; 

# System deps
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y --no-install-recommends \
	bash \
        ca-certificates \
        curl \
        git \
        vim \
        build-essential \
        libsndfile1-dev \
        ffmpeg \
        python3-full \
        python3-dev \
        python3-pip \
	python3-venv

# Install Python deps. Use PyTorch cu124 wheel index so the cu124 binary resolves.
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
#   /opt/venv/bin/pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 fastapi>=0.111.0 -r requirements.txt
    /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

# Copy app
COPY app ./app
COPY templates ./templates

# Reset NVIDIA stuff
RUN set -eux; \
    # restore any files moved to tmp backup
    if ls /tmp/apt-sources-backup/* >/dev/null 2>&1; then \
      mkdir -p /etc/apt/sources.list.d; \
      for f in /tmp/apt-sources-backup/*; do \
        echo "restoring $f -> /etc/apt/sources.list.d/"; \
        mv "$f" /etc/apt/sources.list.d/ || true; \
      done; \
    fi; \
    # restore any renamed .disabled files
    for f in /etc/apt/sources.list.d/*.disabled; do \
      [ -f "$f" ] && mv "$f" "${f%.disabled}"; \
    done; \
    # restore any sed-created .bak backups (moves file.bak -> file)
    for f in /etc/apt/sources.list.d/*.bak; do \
      [ -f "$f" ] && mv "$f" "${f%.bak}"; \
    done; \
    # restore top-level sources.list backup if present
    [ -f /etc/apt/sources.list.bak ] && mv /etc/apt/sources.list.bak /etc/apt/sources.list || true; \
    # refresh apt lists
    rm -rf /var/lib/apt/lists/*;
#    apt-get update

# Delete cache
RUN rm -rf /var/lib/apt/lists/*

#Open Ports
EXPOSE 8000

# Use sh -c so $PORT expands
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
