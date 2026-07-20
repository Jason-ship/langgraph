# ────────────────────────────────────────────────────────────────────────────
# NovelFactory LangGraph — Production Dockerfile (Multi-stage)
#
# Security-hardened build: Python 3.12-slim base, non-root user.
# Multi-stage: builder compiles deps, runtime is minimal.
#
# Build:  docker build -t novelfactory-langgraph:latest .
# Run:    docker compose -p langgraph up -d
# ────────────────────────────────────────────────────────────────────────────

# ============================================================================
# Builder Stage — compile Python dependencies
# ============================================================================
FROM python:3.12-slim AS builder

ARG APP_VERSION=8.0.0

LABEL org.opencontainers.image.title="NovelFactory LangGraph"
LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.description="AI Novel Creation System - Multi-Agent Architecture"

WORKDIR /app

# Use China mirror for apt (faster than deb.debian.org)
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list 2>/dev/null \
    || true

# Install build dependencies (gcc/libpq-dev for C extensions like psycopg, asyncpg)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python/uv mirror config (must be before any pip/uv install)
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Install uv for fast, reliable dependency resolution
RUN pip install --no-cache-dir uv

# Install project dependencies
COPY pyproject.toml ./
COPY src/ ./src
RUN uv pip install --system --no-cache hatchling \
    && uv pip install --system --no-cache .

# ============================================================================
# Runtime Stage — minimal image for production
# ============================================================================
FROM python:3.12-slim

ARG APP_VERSION=8.0.0
ENV NOVELFACTORY_VERSION=${APP_VERSION}

LABEL org.opencontainers.image.title="NovelFactory LangGraph"
LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.description="AI Novel Creation System - Multi-Agent Architecture"

WORKDIR /app

# Use China mirror for apt (faster than deb.debian.org)
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list 2>/dev/null \
    || true

# Install runtime dependencies only (no gcc/libpq-dev — strictly runtime)
# - libpq5: runtime shared lib for postgres clients
# - tini: proper init system (zombie reaping, signal forwarding)
# - curl: for healthcheck
# - npm: needed to install lark-cli (Feishu CLI) and for healthcheck script
# - redis-tools: redis-cli for the healthcheck script
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        tini \
        curl \
        npm \
        redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install lark-cli globally for Feishu document sync
# Step 1: Install npm JS wrapper (skip postinstall binary download — unreliable in Docker)
RUN npm install -g --ignore-scripts @larksuite/cli@1.0.57

# Step 2: Manually download + extract lark-cli binary
RUN LARK_CLI_DIR="/usr/local/lib/node_modules/@larksuite/cli" \
    && mkdir -p "$LARK_CLI_DIR/bin" \
    && for url in \
        "https://registry.npmmirror.com/-/binary/lark-cli/v1.0.57/lark-cli-1.0.57-linux-amd64.tar.gz" \
        "https://github.com/larksuite/cli/releases/download/v1.0.57/lark-cli-1.0.57-linux-amd64.tar.gz" \
    ; do \
        echo "Trying: $url" \
        && if curl -fsSL --connect-timeout 10 --max-time 120 --retry 2 -o /tmp/lark-cli.tar.gz "$url" ; then \
            echo "Download OK from $url" && break ; \
        else \
            echo "Download FAILED from $url" ; \
        fi \
    ; done \
    && tar -xzf /tmp/lark-cli.tar.gz -C "$LARK_CLI_DIR/bin" \
    && chmod +x "$LARK_CLI_DIR/bin/lark-cli" \
    && rm /tmp/lark-cli.tar.gz \
    && ln -sf "$LARK_CLI_DIR/bin/lark-cli" /usr/local/bin/lark-cli \
    && lark-cli --version

# Install PM2 for process management
RUN npm install -g pm2@latest

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy Python CLI entry points from builder stage (uvicorn is needed for CMD)
# Note: uvicorn entry point sits in /usr/local/bin and is not part of site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application source from builder stage
COPY --from=builder /app/src ./src

# Copy remaining deploy artifacts
COPY deploy/scripts ./deploy/scripts

# Copy healthcheck script
COPY deploy/scripts/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN tr -d '\r' < /usr/local/bin/healthcheck.sh > /tmp/healthcheck_fixed.sh \
    && mv /tmp/healthcheck_fixed.sh /usr/local/bin/healthcheck.sh \
    && chmod +x /usr/local/bin/healthcheck.sh

# Copy PM2 ecosystem config
COPY ecosystem.config.js ./ecosystem.config.js

# ── v5.1.1: 安全加固 — 创建非 root 用户运行服务 ────────────────────────────
# 创建 novelfactory 用户组和用户, 设置 /app 和 /data 的所有权
RUN groupadd -r novelfactory -g 1000 \
    && useradd -r -g novelfactory -u 1000 -m -s /bin/bash novelfactory \
    && chown -R novelfactory:novelfactory /app \
    && mkdir -p /data/logs \
    && chown -R novelfactory:novelfactory /data

# Environment
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STORAGE_PATH=/data \
    HOME=/home/novelfactory \
    PYTHONWARNINGS=ignore::SyntaxWarning

# 切换到非 root 用户运行
USER novelfactory

# Expose API port
EXPOSE 8000

# Health check — verifies API + DB/Redis/Neo4j/Milvus connectivity
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh || exit 1

# Use tini as init system (zombie reaping, signal forwarding)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start NovelFactory FastAPI server
CMD ["uvicorn", "novelfactory.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
