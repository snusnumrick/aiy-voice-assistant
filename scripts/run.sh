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

# ==== Fix APT repository issues
echo "Fixing APT repository issues..."
APT_FIX_LOG="/tmp/apt_fix.log"
echo "$(date): Starting APT repository fix" > $APT_FIX_LOG

# Remove the problematic aiyprojects repository if it exists
if [ -f "/etc/apt/sources.list.d/aiyprojects.list" ]; then
    echo "$(date): Removing aiyprojects repository file" >> $APT_FIX_LOG
    sudo rm -f /etc/apt/sources.list.d/aiyprojects.list >> $APT_FIX_LOG 2>&1
fi

# Also check for any references in main sources.list and other files
sudo find /etc/apt -name "*.list*" -exec grep -l "aiyprojects" {} \; 2>/dev/null | while read -r file; do
    if [ -f "$file" ]; then
        echo "$(date): Commenting out aiyprojects entries in $file" >> $APT_FIX_LOG
        sudo sed -i 's/^deb.*aiyprojects.*/#&/' "$file" >> $APT_FIX_LOG 2>&1
        sudo sed -i 's/^deb-src.*aiyprojects.*/#&/' "$file" >> $APT_FIX_LOG 2>&1
    fi
done

# Update package lists after fixing repositories
echo "$(date): Updating package lists..." >> $APT_FIX_LOG
sudo apt update >> $APT_FIX_LOG 2>&1
APT_UPDATE_RESULT=$?

if [ $APT_UPDATE_RESULT -eq 0 ]; then
    echo "$(date): APT update completed successfully" >> $APT_FIX_LOG
else
    echo "$(date): WARNING: APT update completed with warnings/errors" >> $APT_FIX_LOG
fi

echo "APT repository fix completed. See $APT_FIX_LOG for details."

# ==== end of Fix APT repository issues

# ==== Fix Tailscale-related issues
echo "Starting Tailscale setup and verification..."
TAILSCALE_LOG="/tmp/tailscale_setup.log"
echo "$(date): Starting Tailscale setup check" > $TAILSCALE_LOG

# Source the .env file if it exists (for email configuration)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "$(date): Sourcing .env file" >> $TAILSCALE_LOG
    source "${PROJECT_ROOT}/.env"
    # Log email configuration variables (without showing actual passwords)
    echo "$(date): Email config - SMTP_SERVER: ${SMTP_SERVER:-not set}" >> $TAILSCALE_LOG
    echo "$(date): Email config - SMTP_PORT: ${SMTP_PORT:-not set}" >> $TAILSCALE_LOG
    echo "$(date): Email config - SMTP_USER: ${SMTP_USER:-not set}" >> $TAILSCALE_LOG
    echo "$(date): Email config - SMTP_PASSWORD: [${SMTP_PASSWORD:+is set}${SMTP_PASSWORD:-not set}]" >> $TAILSCALE_LOG
else
    echo "$(date): WARNING: .env file not found, email configuration may be missing" >> $TAILSCALE_LOG
fi

# Get the config file path from environment or use default
CONFIG_FILE=${CONFIG_FILE:-"${PROJECT_ROOT}/config.json"}
echo "$(date): Using config file: $CONFIG_FILE" >> $TAILSCALE_LOG

# Extract admin email from config.json with more detailed logging
echo "$(date): Attempting to extract admin_email from config.json" >> $TAILSCALE_LOG
if [ -f "$CONFIG_FILE" ]; then
    echo "$(date): Config file exists" >> $TAILSCALE_LOG
    ADMIN_EMAIL=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('admin_email', 'treskunov@gmail.com'))" 2>> $TAILSCALE_LOG || echo "treskunov@gmail.com")
    echo "$(date): Admin email set to: $ADMIN_EMAIL" >> $TAILSCALE_LOG
else
    echo "$(date): WARNING: Config file not found, using default admin email" >> $TAILSCALE_LOG
    ADMIN_EMAIL="treskunov@gmail.com"
fi

# Check Tailscale status before attempting installation
echo "$(date): Checking current Tailscale status" >> $TAILSCALE_LOG
if command -v tailscale >/dev/null 2>&1; then
    tailscale status >> $TAILSCALE_LOG 2>&1
    echo "$(date): Tailscale version: $(tailscale version 2>/dev/null || echo 'unknown')" >> $TAILSCALE_LOG
else
    echo "$(date): Tailscale not currently installed" >> $TAILSCALE_LOG
fi

# Always reinstall Tailscale to ensure it's not corrupted
echo "$(date): Installing/Reinstalling Tailscale..." >> $TAILSCALE_LOG
curl -fsSL https://tailscale.com/install.sh | sh >> $TAILSCALE_LOG 2>&1
INSTALL_RESULT=$?

if [ $INSTALL_RESULT -eq 0 ]; then
    echo "$(date): Tailscale installed successfully" >> $TAILSCALE_LOG
else
    echo "$(date): ERROR: Failed to install Tailscale with exit code $INSTALL_RESULT" >> $TAILSCALE_LOG
fi

# Add more detailed email sending process
if [ -n "${ERRORS_DETECTED}" ]; then
    echo "$(date): Errors detected in Tailscale setup, sending email report" >> $TAILSCALE_LOG

    # Log the email sending command and parameters (without showing password)
    echo "$(date): Attempting to send email to $ADMIN_EMAIL via ${SMTP_SERVER:-default mail server}" >> $TAILSCALE_LOG

    # Create a temporary email content file
    EMAIL_CONTENT="/tmp/tailscale_email_content.txt"
    {
        echo "Subject: Tailscale Setup Error Report"
        echo "From: Tailscale Setup <${SMTP_USER:-noreply@example.com}>"
        echo "To: $ADMIN_EMAIL"
        echo "Content-Type: text/plain"
        echo ""
        echo "Tailscale setup encountered errors on $(hostname) at $(date)"
        echo ""
        echo "--- Log File Contents ---"
        cat $TAILSCALE_LOG
        echo "--- End Log File Contents ---"
    } > $EMAIL_CONTENT

    # Try to send email and log the result
    if command -v mail >/dev/null 2>&1; then
        echo "$(date): Sending email using 'mail' command" >> $TAILSCALE_LOG
        cat $EMAIL_CONTENT | mail -s "Tailscale Setup Error Report" $ADMIN_EMAIL >> $TAILSCALE_LOG 2>&1
        MAIL_RESULT=$?
        echo "$(date): Mail command exit code: $MAIL_RESULT" >> $TAILSCALE_LOG
    elif command -v sendmail >/dev/null 2>&1; then
        echo "$(date): Sending email using 'sendmail' command" >> $TAILSCALE_LOG
        cat $EMAIL_CONTENT | sendmail -t >> $TAILSCALE_LOG 2>&1
        MAIL_RESULT=$?
        echo "$(date): Sendmail command exit code: $MAIL_RESULT" >> $TAILSCALE_LOG
    else
        echo "$(date): ERROR: No mail sending utility found (mail or sendmail)" >> $TAILSCALE_LOG
        MAIL_RESULT=1
    fi

    if [ $MAIL_RESULT -eq 0 ]; then
        echo "$(date): Email report sent successfully" >> $TAILSCALE_LOG
    else
        echo "$(date): ERROR: Failed to send email report" >> $TAILSCALE_LOG
    fi

    # Remove temporary email content file
    rm -f $EMAIL_CONTENT

    echo "$(date): Email report sent in $TAILSCALE_LOG" >> $TAILSCALE_LOG
fi

# ==== end of Fix Tailscale-related issues

# ==== Set System Timezone ====
echo "Setting system timezone..."
TIMEZONE_LOG="/tmp/timezone_setup.log"
echo "$(date): Starting timezone setup" > $TIMEZONE_LOG

# Get timezone using the Python utility function
# Ensure .env is sourced so GOOGLE_API_KEY is available for get_timezone
if [ -f "${PROJECT_ROOT}/.env" ]; then
    source "${PROJECT_ROOT}/.env"
fi
TIMEZONE=$(poetry run python -c "from src.tools import get_timezone; print(get_timezone())" 2>> $TIMEZONE_LOG)

if [ -z "$TIMEZONE" ]; then
    echo "$(date): ERROR: Failed to get timezone." >> $TIMEZONE_LOG
    # Optionally, send an email notification here if critical
else
    echo "$(date): Determined timezone: $TIMEZONE" >> $TIMEZONE_LOG
    export TZ="$TIMEZONE"
    echo "$(date): TZ environment variable set to $TIMEZONE" >> $TIMEZONE_LOG

    echo "$(date): Updating system timezone files..." >> $TIMEZONE_LOG
    if [ -f "/usr/share/zoneinfo/$TIMEZONE" ]; then
        sudo cp "/usr/share/zoneinfo/$TIMEZONE" /etc/localtime >> $TIMEZONE_LOG 2>&1
        echo "$TIMEZONE" | sudo tee /etc/timezone > /dev/null 2>> $TIMEZONE_LOG
        echo "$(date): System timezone files updated." >> $TIMEZONE_LOG

        echo "$(date): Restarting cron service..." >> $TIMEZONE_LOG
        sudo service cron restart >> $TIMEZONE_LOG 2>&1
        CRON_RESTART_RESULT=$?
        if [ $CRON_RESTART_RESULT -eq 0 ]; then
            echo "$(date): Cron service restarted successfully." >> $TIMEZONE_LOG
        else
            echo "$(date): ERROR: Failed to restart cron service. Exit code: $CRON_RESTART_RESULT" >> $TIMEZONE_LOG
        fi
    else
        echo "$(date): ERROR: Timezone info file not found: /usr/share/zoneinfo/$TIMEZONE" >> $TIMEZONE_LOG
    fi
fi
echo "Timezone setup completed. See $TIMEZONE_LOG for details."
# ==== end of Set System Timezone ====

# set audio volume
amixer sset 'Master' 90% || amixer sset 'Speaker' 55% || echo "Failed to set volume"

# Run the Python script with new logging flags
poetry run python main.py --log-dir "${PROJECT_ROOT}/logs" --log-level INFO
