#!/bin/bash

# get path to script (project root)
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Log the start of the script
echo "Starting run.sh" >> /home/anton/aiy-voice-assistant/logfile.log

# Navigate to the project directory
cd "${SCRIPT_DIR}"
echo "Changed directory to ${SCRIPT_DIR}" >> /home/anton/aiy-voice-assistant/logfile.log

# Activate the virtual environment
source ./venv/bin/activate
echo "Virtual environment activated" >> /home/anton/aiy-voice-assistant/logfile.log

# Pull the latest changes from the repository
git pull >> /home/anton/aiy-voice-assistant/logfile.log 2>&1
echo "Git pull completed" >> /home/anton/aiy-voice-assistant/logfile.log

# Run the Python script
python main.py >> /home/anton/aiy-voice-assistant/logfile.log 2>&1
echo "Started main.py" >> /home/anton/aiy-voice-assistant/logfile.log

# Keep the tmux session alive
tail -f /dev/null