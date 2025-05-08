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

# Function to read config value with default
get_config_value() {
    local key=$1
    local default=$2
    value=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('$key', '$default'))" 2>/dev/null || echo "$default")
    echo "$value"
}

# Email configuration from config.json with defaults
ASSISTANT_EMAIL=$(get_config_value "assistant_email_address" "cubick@treskunov.net")
SMTP_SERVER=$(get_config_value "smtp_server" "mail.treskunov.net")
SMTP_PORT=$(get_config_value "smtp_port" "26")
USERNAME=$(get_config_value "assistant_email_username" "cubick@treskunov.net")

# Get password from environment variable
PASSWORD="${EMAIL_PASSWORD}"

# Check if required environment variable is set
if [ -z "$PASSWORD" ]; then
    echo "Error: EMAIL_PASSWORD environment variable is not set"
    exit 1
fi

# Function to send email
send_email() {
    local subject="$1"
    local body="$2"
    local to_address="$3"

    if [ -z "$subject" ] || [ -z "$body" ] || [ -z "$to_address" ]; then
        echo "Error: Missing required parameters"
        echo "Usage: $0 \"subject\" \"body\" \"recipient@email.com\""
        exit 1
    fi

    # Create temporary file for email content
    EMAIL_CONTENT=$(mktemp)

    # Create email headers and body
    cat > "$EMAIL_CONTENT" << EOF
From: ${ASSISTANT_EMAIL}
To: ${to_address}
Subject: ${subject}

${body}
EOF

    # Send email using curl and SMTP
    curl --url "smtp://${SMTP_SERVER}:${SMTP_PORT}" \
         --ssl-reqd \
         --mail-from "${ASSISTANT_EMAIL}" \
         --mail-rcpt "${to_address}" \
         --upload-file "$EMAIL_CONTENT" \
         --user "${USERNAME}:${PASSWORD}" \
         --silent

    if [ $? -eq 0 ]; then
        echo "Email sent successfully to ${to_address}"
    else
        echo "Error sending email to ${to_address}"
        exit 1
    fi

    # Clean up temporary file
    rm -f "$EMAIL_CONTENT"
}

# If script is called directly (not sourced), send email with provided parameters
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    send_email "$1" "$2" "$3"
fi