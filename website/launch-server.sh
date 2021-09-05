#!/bin/bash

set -e

export COMMAND="python3 $1 > log.txt 2>&1"

rm -r __pycache__/ 2> /dev/null || true

(tmux kill-session -t flask-server 2> /dev/null || exit 0) && sleep 3 && tmux new -s flask-server -d "$COMMAND" 2> /dev/null

echo "Success!"