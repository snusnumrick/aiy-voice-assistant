# scripts/tailscale-down.sh
#!/bin/bash
set -e

logger -t tailscale-scheduler "Starting Tailscale disable process"

if ! tailscale status >/dev/null 2>&1; then
    logger -t tailscale-scheduler "Tailscale is already down"
    exit 0
fi

if tailscale down; then
    logger -t tailscale-scheduler "Tailscale disabled successfully"
else
    logger -t tailscale-scheduler "Failed to disable Tailscale"
    exit 1
fi