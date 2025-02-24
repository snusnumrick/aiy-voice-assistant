#!/bin/bash

# Email configuration
ASSISTANT_EMAIL="cubick@treskunov.net"
USER_EMAIL="treskunov@gmail.com"
SMTP_SERVER="mail.treskunov.net"
SMTP_PORT="26"
USERNAME="cubick@treskunov.net"

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
    local to_address="${3:-$USER_EMAIL}"  # Use provided address or default

    # Create temporary file for email content
    EMAIL_CONTENT=$(mktemp)

    # Create email headers and body
    cat > "$EMAIL_CONTENT" << EOF
From: ${ASSISTANT_EMAIL}
To: ${to_address}
Subject: ${subject}

${body}
EOF

    # Send email using sendmail or curl
    if command -v sendmail &> /dev/null; then
        sendmail -t < "$EMAIL_CONTENT"
        echo "Email sent using sendmail"
    else
        # Alternative using curl and SMTP
        curl --url "smtp://${SMTP_SERVER}:${SMTP_PORT}" \
             --ssl-reqd \
             --mail-from "${ASSISTANT_EMAIL}" \
             --mail-rcpt "${to_address}" \
             --upload-file "$EMAIL_CONTENT" \
             --user "${USERNAME}:${PASSWORD}" \
             --silent

        if [ $? -eq 0 ]; then
            echo "Email sent successfully"
        else
            echo "Error sending email"
        fi
    fi

    # Clean up temporary file
    rm -f "$EMAIL_CONTENT"
}

# Example usage:
# send_email "Test Subject" "This is the email body" "optional_recipient@example.com"