FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WESPEAKER_MODEL_PATH=/models/wespeaker.onnx

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt

ARG WESPEAKER_MODEL_ID="resnet34-lm-voxceleb"
ARG WESPEAKER_MODEL_URL="https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx?download=true"
ARG WESPEAKER_MODEL_SHA256="7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
RUN mkdir -p /models \
    && curl --fail --location --retry 3 "$WESPEAKER_MODEL_URL" --output /models/wespeaker.onnx \
    && echo "$WESPEAKER_MODEL_SHA256  /models/wespeaker.onnx" | sha256sum --check

ENV WESPEAKER_MODEL_ID=$WESPEAKER_MODEL_ID

COPY app.py /app/app.py

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
