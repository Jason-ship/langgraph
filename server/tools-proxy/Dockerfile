FROM python:3.12-slim

WORKDIR /app

# Install curl (small, for downloading lark-cli binary)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install lark-cli binary directly (no npm needed)
RUN LARK_CLI_VERSION="1.0.57" \
    && LARK_CLI_DIR="/usr/local/lib/lark-cli" \
    && mkdir -p "$LARK_CLI_DIR/bin" \
    && for url in \
        "https://github.com/larksuite/cli/releases/download/v${LARK_CLI_VERSION}/lark-cli-${LARK_CLI_VERSION}-linux-amd64.tar.gz" \
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

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py init_lark.py ./

EXPOSE 5004

CMD ["sh", "-c", "python init_lark.py 2>/dev/null; exec uvicorn app:app --host 0.0.0.0 --port 5004 --log-level info"]
