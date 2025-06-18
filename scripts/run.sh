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

# ==== Fix Tailscale-related issues
echo "Starting Tailscale setup and verification..."
TAILSCALE_LOG="/tmp/tailscale_setup.log"
echo "$(date): Starting Tailscale setup check" > $TAILSCALE_LOG

# Source the .env file if it exists (for email configuration)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
fi

# Get the config file path from environment or use default
CONFIG_FILE=${CONFIG_FILE:-"${PROJECT_ROOT}/config.json"}

# Extract admin email from config.json
ADMIN_EMAIL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('admin_email', 'treskunov@gmail.com'))" 2>/dev/null || echo "treskunov@gmail.com")

# Always reinstall Tailscale to ensure it's not corrupted
echo "$(date): Installing/Reinstalling Tailscale..." >> $TAILSCALE_LOG
curl -fsSL https://tailscale.com/install.sh | sh >> $TAILSCALE_LOG 2>&1
INSTALL_RESULT=$?

if [ $INSTALL_RESULT -eq 0 ]; then
    echo "$(date): Tailscale installed successfully" >> $TAILSCALE_LOG
else
    echo "$(date): ERROR: Failed to install Tailscale" >> $TAILSCALE_LOG
fi

# Make scripts executable
chmod +x "${SCRIPT_DIR}/tailscale-up.sh" 2>> $TAILSCALE_LOG
chmod +x "${SCRIPT_DIR}/tailscale-down.sh" 2>> $TAILSCALE_LOG
echo "$(date): Set executable permissions on Tailscale scripts" >> $TAILSCALE_LOG

# Check if Tailscale cron jobs exist, add them if they don't
if ! crontab -l 2>/dev/null | grep -q "tailscale-up.sh"; then
    echo "$(date): Setting up Tailscale cron jobs..." >> $TAILSCALE_LOG
    (crontab -l 2>/dev/null | grep -v tailscale-) > /tmp/temp_cron
    echo "0 22 * * * ${SCRIPT_DIR}/tailscale-up.sh" >> /tmp/temp_cron
    echo "0 7 * * * ${SCRIPT_DIR}/tailscale-down.sh" >> /tmp/temp_cron
    crontab /tmp/temp_cron 2>> $TAILSCALE_LOG
    CRON_RESULT=$?
    rm /tmp/temp_cron

    if [ $CRON_RESULT -eq 0 ]; then
        echo "$(date): Tailscale cron jobs set up successfully" >> $TAILSCALE_LOG
    else
        echo "$(date): ERROR: Failed to set up Tailscale cron jobs" >> $TAILSCALE_LOG
    fi
else
    echo "$(date): Tailscale cron jobs already exist" >> $TAILSCALE_LOG
fi

# Try to start Tailscale if it's not running
echo "$(date): Checking Tailscale status..." >> $TAILSCALE_LOG
if ! tailscale status &>/dev/null; then
    echo "$(date): Attempting to start Tailscale..." >> $TAILSCALE_LOG
    "${SCRIPT_DIR}/tailscale-up.sh" >> $TAILSCALE_LOG 2>&1
    TAILSCALE_RESULT=$?

    if [ $TAILSCALE_RESULT -eq 0 ]; then
        echo "$(date): Tailscale started successfully" >> $TAILSCALE_LOG
    else
        echo "$(date): ERROR: Failed to start Tailscale" >> $TAILSCALE_LOG
    fi
else
    echo "$(date): Tailscale is already running" >> $TAILSCALE_LOG
fi

# Check for errors in the log
if grep -q "ERROR" $TAILSCALE_LOG; then
    echo "$(date): Errors detected in Tailscale setup, sending email report" >> $TAILSCALE_LOG

    # Extract error messages
    ERROR_CONTENT=$(grep -A 2 "ERROR" $TAILSCALE_LOG)

    # Prepare email subject and body
    EMAIL_SUBJECT="AIY Assistant Tailscale Setup Errors"
    EMAIL_BODY="Errors encountered during Tailscale setup on system restart:

${ERROR_CONTENT}

Full log:
$(cat $TAILSCALE_LOG)"

    # Call send_email.sh script
    "${SCRIPT_DIR}/send_email.sh" "${EMAIL_SUBJECT}" "${EMAIL_BODY}" "${ADMIN_EMAIL}"
    echo "$(date): Email report sent" >> $TAILSCALE_LOG
else
    echo "$(date): Tailscale setup completed without errors" >> $TAILSCALE_LOG
fi

echo "Tailscale setup check completed. See $TAILSCALE_LOG for details."

# ==== end of Fix Tailscale-related issues

# set audio volume
amixer sset 'Master' 90% || amixer sset 'Speaker' 55% || echo "Failed to set volume"

# Run the Python script with new logging flags
poetry run sudo python main.py --log-dir "${PROJECT_ROOT}/logs" --log-level INFO
