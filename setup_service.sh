#!/bin/bash

# make poetry available to Root
sudo cp ~/.local/bin/poetry /usr/bin

# Define variables
SERVICE_NAME="aiy"
SERVICE_DESCRIPTION="AIY Voice Assistant Service"
USER=$(whoami)
GROUP=$(id -gn)
WORKING_DIR="$(dirname "$(readlink -f "$0")")"
RUN_SCRIPT="${WORKING_DIR}/run.sh"
LOG_FILE="${WORKING_DIR}/logfile.log"
ERROR_LOG="${WORKING_DIR}/errorfile.log"

# Make run.sh executable
chmod +x "${RUN_SCRIPT}"

# Create the service file content
cat << EOF > "${SERVICE_NAME}.service"
[Unit]
Description=${SERVICE_DESCRIPTION}
After=network-online.target sound.target
Wants=network-online.target sound.target
Required=network-online.target sound.target

[Service]
Type=simple
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HOME=/home/${USER}"
ExecStartPre=/bin/sleep 10
ExecStart=/bin/bash -c "${RUN_SCRIPT}"
WorkingDirectory=${WORKING_DIR}
User=${USER}
Group=${GROUP}
StandardOutput=append:${LOG_FILE}
StandardError=append:${ERROR_LOG}

[Install]
WantedBy=multi-user.target
EOF

# Move the service file to the correct location
sudo mv "${SERVICE_NAME}.service" /etc/systemd/system/

# Create logrotate configuration
cat << EOF > "${SERVICE_NAME}-logrotate"
${LOG_FILE}
${ERROR_LOG} {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ${USER} ${GROUP}
}
EOF

# Move the logrotate configuration to the correct location
sudo mv "${SERVICE_NAME}-logrotate" /etc/logrotate.d/${SERVICE_NAME}

# Set correct permissions for the logrotate configuration
sudo chmod 644 /etc/logrotate.d/${SERVICE_NAME}

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable "${SERVICE_NAME}.service"

# Start the service
sudo systemctl start "${SERVICE_NAME}.service"

echo "Service ${SERVICE_NAME} has been created, enabled, and started."
echo "Logrotate configuration for ${SERVICE_NAME} has been set up."
echo "You can check its status with: sudo systemctl status ${SERVICE_NAME}.service"