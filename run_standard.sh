#!/bin/sh
curl -s -X POST "http://localhost:8123/threads/265a96e0-3874-44f3-8200-8e26f6113a2e/runs/wait" \
  -H "Content-Type: application/json" \
  -d @/tmp/standard_run.json
