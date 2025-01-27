#!/bin/bash

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
        # Extract error messages
        ERROR_CONTENT=$(grep -i "error\|exception\|failed\|traceback" -A 3 "$LOG_FILE")
        
        # Use the email_tools.py script to send the email
        ERROR_COUNT_VAR=$ERROR_COUNT
        ERROR_CONTENT_VAR=$ERROR_CONTENT
        poetry run python -c "
from src.email_tools import send_email
from src.config import Config
config = Config()
send_email(
    'AIY Assistant Log Errors Found',
    f'Found {$ERROR_COUNT_VAR} errors in the log file:\\n\\n{$ERROR_CONTENT_VAR}',
    config,
    '${ADMIN_EMAIL}'
)
"
    fi
fi
