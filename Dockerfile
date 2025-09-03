FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PORT=8000
WORKDIR /app

#TEMP: NVIDIA repo workaround (kept in case needed)
#RUN set -eux;
#grep -R "developer.download.nvidia.com" /etc/apt/sources.list /etc/apt/sources.list.d/.list 2>/dev/null || true;
#grep -Rl "developer.download.nvidia.com" /etc/apt 2>/dev/null | while IFS= read -r f; do
#sed -i.bak -E 's/^(.developer.download.nvidia.com.)$/#\1/' "$f" || true;
#done;
#rm -rf /var/lib/apt/lists/;

#System deps (include audio libs)
RUN apt-get update && apt-get upgrade -y
  RUN apt-get install -y --no-install-recommends \
        bash \
	ca-certificates \
	curl \
	git \
	vim \
	build-essential \
	libsndfile1 \
	ffmpeg \
	sox \
	python3 \
	python3-dev \
	python3-venv \
	wget \
	ca-certificates

#Set cache locations inside the image so downloads land in writable paths
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
RUN mkdir -p $HF_HOME $TORCH_HOME && chmod -R 777 /app/.cache || true

#Copy requirements first so we can leverage layer caching when editing code
COPY requirements.txt /app/requirements.txt

#Create venv, install torch (GPU wheel) explicitly, then install the rest.
#Adjust torch versions to match your CUDA runtime. We use cu128 for CUDA 12.8.
RUN python3 -m venv /opt/venv && \
/opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
/opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128 torchvision==0.22.1 torchaudio==2.7.1 && \
/opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt && \
/opt/venv/bin/python -c "import fastapi, uvicorn, torch; print('py ok', torch.version, 'cuda_available=', torch.cuda.is_available())"

#Ensure venv is used in all subsequent steps
ENV PATH="/opt/venv/bin:${PATH}"

#Copy the app (after deps) so code changes don't bust installed layers
COPY app ./app
COPY templates ./templates
COPY gpu_check.py /app/gpu_check.py

#Restore any NVIDIA sources if your workaround created backups (kept intact)
#RUN set -eux;
#if ls /tmp/apt-sources-backup/* >/dev/null 2>&1; then
#mkdir -p /etc/apt/sources.list.d;
#for f in /tmp/apt-sources-backup/; do mv "$f" /etc/apt/sources.list.d/ || true; done;
#fi;
#for f in /etc/apt/sources.list.d/.disabled; do [ -f "f" ] && mv "$f" "{f%.disabled}" || true; done;
#for f in /etc/apt/sources.list.d/.bak; do [ -f "f" ] && mv "$f" "{f%.bak}" || true; done;
#[ -f /etc/apt/sources.list.bak ] && mv /etc/apt/sources.list.bak /etc/apt/sources.list || true;
RUN rm -rf /var/lib/apt/lists/

EXPOSE 8000
CMD ["sh", "-c", "/opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info"]
