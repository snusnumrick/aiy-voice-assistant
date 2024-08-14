#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
tmux new-session -d -s my_script_session "${SCRIPT_DIR}/run.sh"