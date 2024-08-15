#!/bin/bash

# get path to script (project root)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Navigate to the project directory
cd "${SCRIPT_DIR}"

# Activate the virtual environment
source ./venv/bin/activate

# Pull the latest changes from the repository
git pull

# Run the Python script
python main.py

# Keep the tmux session alive
tail -f /dev/null