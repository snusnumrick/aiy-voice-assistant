#!/bin/bash

# Get script directory and project root
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Change to project root directory
cd "${PROJECT_ROOT}" || exit

# Source the .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Get the config file path from environment or use default
CONFIG_FILE=${CONFIG_FILE:-"config.json"}

# Extract admin email from config.json, defaulting to treskunov@gmail.com
ADMIN_EMAIL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('admin_email', 'treskunov@gmail.com'))" 2>/dev/null || echo "treskunov@gmail.com")

# Check current log file for errors
LOG_FILE=${LOG_FILE:-"logs/assistant.log"}

if [ -f "$LOG_FILE" ]; then
    # Search for error messages
    ERROR_COUNT=$(grep -i "error\|exception\|failed\|traceback" "$LOG_FILE" | wc -l)

    if [ $ERROR_COUNT -gt 0 ]; then
        echo "DEBUG: Errors detected in the log file, preparing content for email."
        # Extract error messages
        ERROR_CONTENT=$(grep -i "error\|exception\|failed\|traceback" -A 3 "$LOG_FILE")
        # Export variables for Python
        export ERROR_COUNT
        export ERROR_CONTENT

        # Use the email_tools.py script to send the email
        poetry run python -c "
import dotenv
dotenv.load_dotenv()
import os
from src.email_tools import send_email
from src.config import Config
config = Config()
send_email(
    'AIY Assistant Log Errors Found',
    f'Found {os.environ.get(\"ERROR_COUNT\")} errors in the log file:\\n\\n{os.environ.get(\"ERROR_CONTENT\")}',
    config,
    '${ADMIN_EMAIL}'
)
"
        echo "DEBUG: email report sent with ${ERROR_CONTENT}."
    else
        echo "DEBUG: No errors found in the log file."
    fi
fi
exit 0