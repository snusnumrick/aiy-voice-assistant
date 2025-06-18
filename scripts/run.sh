#!/bin/bash

echo "--- Audio Controls ---"
amixer scontrols
echo "--- End Audio Controls ---"

# get path to project root (one level up from scripts)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Navigate to the project directory
cd "${PROJECT_ROOT}" || exit

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

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
amixer sset 'Master' 90% || amixer sset 'Speaker' 55% || echo "Failed to set volume"

# Run the Python script with new logging flags
sudo poetry run python main.py --log-dir "${PROJECT_ROOT}/logs" --log-level INFO
