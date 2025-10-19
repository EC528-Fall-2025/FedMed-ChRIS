FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="pl-matt" \
      org.opencontainers.image.description="A ChRIS plugin for federated learning with OpenFL on MNIST"

ARG SRCDIR=/usr/local/src/pl-matt
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}

WORKDIR /

CMD ["pl-matt"]