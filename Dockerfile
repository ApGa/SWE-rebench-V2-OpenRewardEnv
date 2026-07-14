FROM docker:27-cli AS docker-cli

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/orwd_data \
    OPENREWARD_PORT=8080 \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    python3 \
    python3-pip \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python 3.11 /opt/venv

COPY requirements.txt /app/
RUN uv pip install --python /opt/venv/bin/python -r /app/requirements.txt

COPY . /app/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('OPENREWARD_PORT', '8080'), timeout=3)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/venv/bin/python", "/app/server.py"]
