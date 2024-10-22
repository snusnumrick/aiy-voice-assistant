#!/bin/bash

echo "--- Audio Controls ---"
amixer scontrols
echo "--- End Audio Controls ---"

# get path to script (project root)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to the project directory
cd "${SCRIPT_DIR}"

# wait for the network
max_attempts=12
attempt=1
while ! ping -c 1 github.com >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo "Network not available after $max_attempts attempts, exiting"
        exit 1
    fi
    echo "Waiting for network... attempt $attempt"
    sleep 5
    attempt=$((attempt + 1))
done

# Pull the latest changes from the repository
git pull

# set audio volume
amixer sset 'Master' 40% || amixer sset 'Speaker' 55% || echo "Failed to set volume"

# add local bin directories to PATH
HOME_DIR="$(echo "$SCRIPT_DIR" | cut -d'/' -f1-3)"
export PATH=$HOME_DIR/.pyenv/shims:$HOME_DIR/.pyenv/bin:$HOME_DIR/.local/bin:$PATH
echo $PATH

# Run the Python script
poetry run python main.py
