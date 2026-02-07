#!/usr/bin/env bash
#注意把line7的your/python/path修改成你的python路径
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl /your/python/path -u "$ROOT/goldx_alert.py" >> "$LOG_DIR/goldx.log" 2>&1 &
echo "started pid=$!"
