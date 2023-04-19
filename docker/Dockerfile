FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt -y upgrade \
    && apt install -f -y ffmpeg libsm6 libxext6 git-lfs \
    && pip install ultralytics opencv-python transformers sentence-transformers\
    && unset DEBIAN_FRONTEND