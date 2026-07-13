#!/bin/bash
# Start dream_novel_v5 writing thread
PAYLOAD_FILE="/tmp/novelfactory/start_payload.json"
OUTPUT_FILE="/data/outputs/梦魇档案_v5.jsonl"
nohup curl -s -X POST http://localhost:8123/threads/dream_novel_v5/runs -H "Content-Type: application/json" -d @"$PAYLOAD_FILE" >> "$OUTPUT_FILE" 2>&1 &
echo "PID=$!"
