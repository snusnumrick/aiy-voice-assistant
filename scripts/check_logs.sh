#!/bin/bash

# Source the .env file if it exists
if [ -f .env ]; then
#    echo "DEBUG: .env file found, sourcing it."

    source .env
else
    echo "DEBUG: .env file not found, skipping sourcing."
fi

# Get the config file path from environment or use default
CONFIG_FILE=${CONFIG_FILE:-"config.json"}
#echo "DEBUG: Using config file path: $CONFIG_FILE"

# Extract admin email from config.json, defaulting to treskunov@gmail.com
ADMIN_EMAIL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('admin_email', 'treskunov@gmail.com'))" 2>/dev/null || echo "treskunov@gmail.com")
#echo "DEBUG: Admin email resolved to: $ADMIN_EMAIL"

# Check current log file for errors
LOG_FILE=${LOG_FILE:-"logs/assistant.log"}
#echo "DEBUG: Using log file path: $LOG_FILE"

if [ -f "$LOG_FILE" ]; then
#    echo "DEBUG: Log file found: $LOG_FILE"
    # Search for error messages
    ERROR_COUNT=$(grep -i "error\|exception\|failed\|traceback" "$LOG_FILE" | wc -l)
#    echo "DEBUG: Number of errors found: $ERROR_COUNT"

    if [ $ERROR_COUNT -gt 0 ]; then
        echo "DEBUG: Errors detected in the log file, preparing content for email."
        # Extract error messages
        ERROR_CONTENT=$(grep -i "error\|exception\|failed\|traceback" -A 3 "$LOG_FILE")
#        echo "DEBUG: Extracted error content: $ERROR_CONTENT"
        # Export variables for Python
        export ERROR_COUNT
        export ERROR_CONTENT
#        echo "DEBUG: Environment variables set for Python usage. ERROR_COUNT=$ERROR_COUNT, ERROR_CONTENT set."

        # Use the email_tools.py script to send the email
#        echo "DEBUG: Calling the Python script for sending the email."
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
