# syntax=docker/dockerfile:1.7
# Multi-stage, slim, CPU-only image for z-jev.

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.5.1

# ---------------------------------------------------------------------------
# Stage 1: build dependencies in a throw-away image.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build deps for any wheels that need compiling (none today, but future-proof).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only torch from the PyTorch CPU index. We pin a recent stable version
# known to ship wheels for python 3.11 on linux/amd64.
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==${TORCH_VERSION}"

COPY pyproject.toml README.md ./
COPY z_jev ./z_jev
COPY examples ./examples
COPY scripts ./scripts

RUN pip install --no-deps "."

# ---------------------------------------------------------------------------
# Stage 2: runtime image. Smaller, no compilers, non-root user.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ZJEV_HOST=0.0.0.0 \
    ZJEV_PORT=8000

# Minimal runtime libs (ca-certificates for outbound HTTPS, tini for
# signal handling, dumb-init-style PID 1 courtesy).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd --system --gid 1000 zjev \
    && useradd --system --uid 1000 --gid zjev --home /app --shell /usr/sbin/nologin zjev

WORKDIR /app

# Copy installed Python packages + project from the builder stage.
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/z_jev ./z_jev
COPY --from=builder /build/examples ./examples
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/pyproject.toml ./pyproject.toml
COPY --from=builder /build/README.md ./README.md

# Mount points for checkpoints and JSONL data.
RUN mkdir -p /app/checkpoints /app/data \
    && chown -R zjev:zjev /app

USER zjev

EXPOSE 8000

# OCI labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md).
LABEL org.opencontainers.image.title="z-jev" \
      org.opencontainers.image.description="Non-autoregressive decision heads (Jev primitives) on top of GLM-5 / a tiny CPU replica." \
      org.opencontainers.image.source="https://github.com/hamr-hub/z-jev" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="z-jev contributors"

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec python -m z_jev.serve --host ${ZJEV_HOST} --port ${ZJEV_PORT} ${ZJEV_CHECKPOINT:+--checkpoint ${ZJEV_CHECKPOINT}}"]
