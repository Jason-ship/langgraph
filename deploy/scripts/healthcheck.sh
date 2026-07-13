#!/usr/bin/env bash
# Lightweight healthcheck — checks API endpoint first (fastest).
# Uses curl --max-time for all operations. DB/Redis checks are secondary.
set -uo pipefail

# 1. Check API health endpoint (most critical — 3s timeout is plenty)
curl -sf --max-time 3 http://localhost:8000/health > /dev/null 2>&1 || { echo "FAIL: API not responding"; exit 1; }

# 2. Check PostgreSQL connectivity (5s connect timeout)
timeout 8 python3 -c "
import os, psycopg
c=psycopg.connect(host=os.environ.get('DB_HOST','postgres'),port=int(os.environ.get('DB_PORT','5432')),user=os.environ.get('DB_USER','noveluser'),password=os.environ.get('DB_PASSWORD',''),dbname=os.environ.get('DB_NAME','novelfactory'),connect_timeout=5)
c.cursor().execute('SELECT 1').fetchone()
c.close()
" 2>/dev/null || { echo "FAIL: DB"; exit 1; }

# 3. Check Redis connectivity (3s timeout)
REDIS_PASS="${REDIS_PASSWORD:-}"
if [ -n "$REDIS_PASS" ]; then
  timeout 3 redis-cli -h "${REDIS_HOST:-redis}" -p "6379" -a "$REDIS_PASS" --no-auth-warning ping 2>/dev/null | grep -q PONG || { echo "FAIL: Redis"; exit 1; }
else
  timeout 3 redis-cli -h "${REDIS_HOST:-redis}" -p "6379" ping 2>/dev/null | grep -q PONG || { echo "FAIL: Redis"; exit 1; }
fi

echo "OK"
exit 0
