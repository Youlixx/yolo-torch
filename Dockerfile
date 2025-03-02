FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update \
 && apt-get install -y git \
 && python -m pip install pip --upgrade \
 && pip install jupyter pycocotools opencv-python-headless ruff