#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl /opt/anaconda3/envs/llm/bin/python -u "$ROOT/goldx_alert.py" >> "$LOG_DIR/goldx.log" 2>&1 &
echo "started pid=$!"
