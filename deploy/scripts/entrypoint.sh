#!/bin/sh
# entrypoint.sh — NovelFactory container startup (PM2 edition)
# Called by tini (ENTRYPOINT in Dockerfile), so NO tini wrapper needed here.
set -e

# Ensure log directory exists
mkdir -p /data/logs

# Initialize lark-cli configuration from environment variables
python3 /app/deploy/scripts/init_lark.py 2>/dev/null || true

# Start Novelfactory API via PM2
# pm2-runtime runs in the foreground (keeps container alive)
exec pm2-runtime start /app/ecosystem.config.js --only novelfactory-api
