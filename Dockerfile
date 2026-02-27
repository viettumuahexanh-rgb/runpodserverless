FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY runpod_serverless/requirements.txt /tmp/requirements.txt

# Pin CUDA wheels known to work with MimicMotion on RunPod.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

ENV PYTHONPATH="/workspace:/workspace/MimicMotion"
ENV HF_HOME="/runpod-volume/hf"
ENV TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
ENV BASE_MODEL_PATH="stabilityai/stable-video-diffusion-img2vid-xt"
ENV MODELS_ROOT="/runpod-volume/models"
ENV MIMICMOTION_DIR="/workspace/MimicMotion"

CMD ["python3", "-u", "runpod_serverless/handler.py"]
