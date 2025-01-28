# scripts/tailscale-up.sh
#!/bin/bash
set -e

logger -t tailscale-scheduler "Starting Tailscale enable process"

if tailscale status >/dev/null 2>&1; then
    logger -t tailscale-scheduler "Tailscale is already running"
    exit 0
fi

if tailscale up; then
    logger -t tailscale-scheduler "Tailscale enabled successfully"
else
    logger -t tailscale-scheduler "Failed to enable Tailscale"
    exit 1
fi