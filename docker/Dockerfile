FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# work dir
WORKDIR /home
USER root

#*# 1. Install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y \
    build-essential \
    ca-certificates \
    sudo \
    cmake \
    gcc \
    g++ \
    curl \
    git \
    wget \
    python3-pip \
    python3-dev \
    vim \
    python-is-python3 \
    python3.10 \
    python3-pip \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && apt-get clean

#*# 2. git clone source codes
RUN cd /home \
    && pip install wandb>=0.15.0 \
        torch>=1.9.0 \
        torchvision>=0.10.0 \
        numpy>=1.21.0 \
        matplotlib>=3.3.0 \
        tqdm>=4.62.0 \
        einops>=0.3.0 \
        h5py>=3.1.0 \
        opencv-python>=4.5.0 \
        omegaconf \
        huggingface_hub>=0.23.0

#*# default cmd
CMD ["bash"]
## end of the Dockerfile