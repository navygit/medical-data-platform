# Multi-stage build. The runtime image carries no build toolchain and runs as a
# non-root user -- a container that touches patient data should not be able to
# write to its own code.
#
# Core deps only by default. The [ml] extra adds ~2 GB of CUDA wheels, which does
# not belong in an image whose job is ingestion, QC and governance. Build with
# `--build-arg EXTRAS=ml` when you need training.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile any sdist-only wheels; it stays in the
# builder stage and never reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY common ./common
COPY pipelines ./pipelines
COPY scripts ./scripts

ARG EXTRAS=""
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ -n "$EXTRAS" ]; then \
         /opt/venv/bin/pip install ".[$EXTRAS]"; \
       else \
         /opt/venv/bin/pip install .; \
       fi


FROM python:3.11-slim AS runtime

# libgomp1 is required by scipy/scikit-learn at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash mdp

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY --chown=mdp:mdp . .

# Data and artifacts are volume mount points, not image content.
RUN mkdir -p /app/data /app/outputs /app/releases && chown -R mdp:mdp /app
USER mdp

# Fails the build if the package cannot import -- catches a broken install here
# rather than in production.
RUN python -c "import common, pipelines; print('medical-data-platform ready')"

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import common" || exit 1

CMD ["python", "-m", "pipelines.brats.run", "--config", "configs/brats.yaml"]
