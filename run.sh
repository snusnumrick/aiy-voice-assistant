#!/bin/bash

echo "--- Audio Controls ---"
amixer scontrols
echo "--- End Audio Controls ---"

# get path to script (project root)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# add local bin directories to PATH
HOME_DIR="$(echo "$SCRIPT_DIR" | cut -d'/' -f1-3)"
export PATH=$HOME_DIR/.pyenv/shims:$HOME_DIR/.pyenv/bin:$HOME_DIR/.local/bin:$PATH

# Navigate to the project directory
cd "${SCRIPT_DIR}"

# Pull the latest changes from the repository
git pull

# set audio volume
amixer sset 'Master' 40% || amixer sset 'Speaker' 55% || echo "Failed to set volume"

# Run the Python script
poetry run python main.py
