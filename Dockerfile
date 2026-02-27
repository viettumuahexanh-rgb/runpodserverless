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

COPY . /workspace

# Pin CUDA wheels known to work with MimicMotion on RunPod.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 && \
    if [ -f /workspace/requirements.txt ]; then \
      python3 -m pip install --no-cache-dir -r /workspace/requirements.txt; \
    elif [ -f /workspace/runpod_serverless/requirements.txt ]; then \
      python3 -m pip install --no-cache-dir -r /workspace/runpod_serverless/requirements.txt; \
    else \
      echo "requirements.txt not found in build context"; \
      ls -la /workspace; \
      exit 1; \
    fi

RUN set -eux; \
    if [ ! -d /workspace/MimicMotion ]; then \
      git clone --depth 1 https://github.com/Tencent/MimicMotion.git /workspace/MimicMotion; \
    fi; \
    if [ ! -f /workspace/mimicmotion_infer_6gb.py ]; then \
      if [ -f /workspace/runpod_serverless/mimicmotion_infer_6gb.py ]; then \
        cp /workspace/runpod_serverless/mimicmotion_infer_6gb.py /workspace/mimicmotion_infer_6gb.py; \
      elif [ -f /workspace/mimicmotion_infer_6gb.py ]; then \
        true; \
      else \
        echo "mimicmotion_infer_6gb.py missing in build context"; \
        exit 1; \
      fi; \
    fi; \
    if [ -f /workspace/runpod_serverless/loader.py ]; then \
      cp /workspace/runpod_serverless/loader.py /workspace/MimicMotion/mimicmotion/utils/loader.py; \
    elif [ -f /workspace/loader.py ]; then \
      cp /workspace/loader.py /workspace/MimicMotion/mimicmotion/utils/loader.py; \
    fi; \
    if [ -f /workspace/runpod_serverless/preprocess.py ]; then \
      cp /workspace/runpod_serverless/preprocess.py /workspace/MimicMotion/mimicmotion/dwpose/preprocess.py; \
    elif [ -f /workspace/preprocess.py ]; then \
      cp /workspace/preprocess.py /workspace/MimicMotion/mimicmotion/dwpose/preprocess.py; \
    fi; \
    if [ -f /workspace/runpod_serverless/wholebody.py ]; then \
      cp /workspace/runpod_serverless/wholebody.py /workspace/MimicMotion/mimicmotion/dwpose/wholebody.py; \
    elif [ -f /workspace/wholebody.py ]; then \
      cp /workspace/wholebody.py /workspace/MimicMotion/mimicmotion/dwpose/wholebody.py; \
    fi

ENV PYTHONPATH="/workspace:/workspace/MimicMotion"
ENV HF_HOME="/runpod-volume/hf"
ENV TRANSFORMERS_CACHE="/runpod-volume/hf/transformers"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
ENV BASE_MODEL_PATH="stabilityai/stable-video-diffusion-img2vid-xt"
ENV MODELS_ROOT="/runpod-volume/models"
ENV MIMICMOTION_DIR="/workspace/MimicMotion"

CMD ["bash", "-lc", "if [ -f /workspace/runpod_serverless/handler.py ]; then python3 -u /workspace/runpod_serverless/handler.py; else python3 -u /workspace/handler.py; fi"]
