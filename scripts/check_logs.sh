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
    # Search for error messages (excluding false positives)
    ERROR_COUNT=$(grep -i "error\|exception\|failed\|traceback" "$LOG_FILE" | grep -v "notification enabled" | wc -l)

    if [ $ERROR_COUNT -gt 0 ]; then
        echo "DEBUG: Errors detected in the log file, preparing content for email."
        # Extract error messages (excluding false positives)
        ERROR_CONTENT=$(grep -i "error\|exception\|failed\|traceback" "$LOG_FILE" | grep -v "notification enabled" -A 3)


        # Prepare email subject and body
        EMAIL_SUBJECT="AIY Assistant Log Errors Found"
        EMAIL_BODY="Found ${ERROR_COUNT} errors in the log file:

${ERROR_CONTENT}"

        # Call send_email.sh script
        "${SCRIPT_DIR}/send_email.sh" "${EMAIL_SUBJECT}" "${EMAIL_BODY}" "${ADMIN_EMAIL}"

        echo "DEBUG: email report sent with ${ERROR_CONTENT}."
    else
        echo "DEBUG: No errors found in the log file."
    fi
fi
exit 0