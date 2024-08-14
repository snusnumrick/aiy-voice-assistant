#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
exec tmux new-session -s aiy "${SCRIPT_DIR}/run.sh"
