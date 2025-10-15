# Python version can be changed, e.g.
#FROM python:3.12-slim-bookworm
#FROM python:3.12
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
# Ask mentor: Should I use this below, or just use puthon:3.12?
FROM docker.io/python:3.12.1-slim-bookworm 

LABEL org.opencontainers.image.authors="BU <jedelist@bu.edu>" \
      org.opencontainers.image.title="MNIST Classifier ChRIS Plugin Federated Learning" \
      org.opencontainers.image.description="A ChRIS plugin for MNIST Classification with Federated Learning. Used to validate Federated Learning Pipeline."

# Make Python behave well in containers (might delete, check how it works first)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG SRCDIR=/usr/local/src/app
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}

WORKDIR /
CMD ["chrNIST"]
