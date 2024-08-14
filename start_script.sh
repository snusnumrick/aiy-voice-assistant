#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
tmux new-session -d -s aiy "${SCRIPT_DIR}/run.sh"