#!/bin/sh
curl -s -X POST "http://localhost:8123/threads/957d2409-9edc-4c04-b845-9a0a8819fdcf/runs/wait" \
  -H "Content-Type: application/json" \
  -d @/tmp/attract_run.json
