FROM python:3.11-alpine

RUN apk add --no-cache git \
    && pip install --no-cache-dir pytest

WORKDIR /workspace
COPY app.py /workspace/app.py

RUN git init \
    && git config user.email smoke@example.com \
    && git config user.name "Smoke Test" \
    && git add app.py \
    && git commit -m "base"
