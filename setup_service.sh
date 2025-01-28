#!/bin/bash

# Define variables
SERVICE_NAME="aiy"
SERVICE_DESCRIPTION="AIY Voice Assistant Service"
USER=$(whoami)
GROUP=$(id -gn)
WORKING_DIR="$(dirname "$(readlink -f "$0")")"
SCRIPTS_DIR="${WORKING_DIR}/scripts"
LOGS_DIR="${WORKING_DIR}/logs"

# Check for sudo
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

# Make scripts executable
chmod +x "${SCRIPTS_DIR}/run.sh"
chmod +x "${SCRIPTS_DIR}/check_logs.sh"
chmod +x "${SCRIPTS_DIR}/tailscale-up.sh"
chmod +x "${SCRIPTS_DIR}/tailscale-down.sh"

# Set up Tailscale cron jobs
(crontab -l 2>/dev/null | grep -v tailscale-) > temp_cron
echo "0 22 * * * ${SCRIPTS_DIR}/tailscale-up.sh" >> temp_cron
echo "0 7 * * * ${SCRIPTS_DIR}/tailscale-down.sh" >> temp_cron
crontab temp_cron
rm temp_cron

# Ensure logs directory exists with correct permissions
mkdir -p "${LOGS_DIR}"
chmod 755 "${LOGS_DIR}"
chown "${SUDO_USER}":"$(id -gn ${SUDO_USER})" "${LOGS_DIR}"

# Create the service file content
cat << EOF > "${SERVICE_NAME}.service"
[Unit]
Description=${SERVICE_DESCRIPTION}
After=network-online.target sound.target
Wants=network-online.target sound.target
Required=network-online.target sound.target

[Service]
Type=simple
Environment="PATH=/home/${SUDO_USER}/.pyenv/shims:/home/${SUDO_USER}/.pyenv/bin:/home/${SUDO_USER}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HOME=/home/${SUDO_USER}"
ExecStartPre=/bin/sleep 10
ExecStart=/bin/bash -c "${SCRIPTS_DIR}/run.sh"
WorkingDirectory=${WORKING_DIR}
User=${SUDO_USER}
Group=$(id -gn ${SUDO_USER})
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Move the service file to the correct location
mv "${SERVICE_NAME}.service" /etc/systemd/system/

# Create logrotate configuration
cat << EOF > "${SERVICE_NAME}-logrotate"
${LOGS_DIR}/*.log {
    daily
    rotate 5
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ${SUDO_USER} $(id -gn ${SUDO_USER})
    su ${SUDO_USER} $(id -gn ${SUDO_USER})
    prerotate
        cd ${WORKING_DIR} && ${SCRIPTS_DIR}/check_logs.sh
    endscript
}
EOF

# Move the logrotate configuration to the correct location
mv "${SERVICE_NAME}-logrotate" /etc/logrotate.d/${SERVICE_NAME}

# Set correct permissions for the logrotate configuration
chmod 644 /etc/logrotate.d/${SERVICE_NAME}

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable "${SERVICE_NAME}.service"

# Start the service
systemctl start "${SERVICE_NAME}.service"

echo "Service ${SERVICE_NAME} has been created, enabled, and started."
echo "Logrotate configuration for ${SERVICE_NAME} has been set up."
echo "Tailscale management scripts have been installed and scheduled."
echo "You can check service status with: sudo systemctl status ${SERVICE_NAME}.service"
echo "You can check Tailscale schedules with: sudo crontab -l"
echo "You can monitor Tailscale operations in syslog with: grep tailscale-scheduler /var/log/syslog"