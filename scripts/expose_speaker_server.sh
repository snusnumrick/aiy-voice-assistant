#!/usr/bin/env bash
# Run on the embedding server after starting WeSpeaker with a bearer token.
set -euo pipefail

port="${1:-8081}"
case "$port" in
    ''|*[!0-9]*) echo 'Usage: bash scripts/expose_speaker_server.sh [local-port]' >&2; exit 1 ;;
esac
if (( port < 1 || port > 65535 )); then
    echo 'Local port must be between 1 and 65535.' >&2
    exit 1
fi
for tool in curl tailscale python3; do
    command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 1; }
done

origin="http://127.0.0.1:$port"
curl --fail --silent --show-error --max-time 10 "$origin/healthz" >/dev/null
status="$(curl --silent --show-error --max-time 10 --output /dev/null \
    --write-out '%{http_code}' -H 'Content-Type: audio/wav' \
    --data-binary '' "$origin/v1/embeddings")"
if [[ "$status" != 401 ]]; then
    echo "Refusing public access: unauthenticated request returned $status, expected 401." >&2
    echo 'Start the service with a nonempty WESPEAKER_API_KEY first.' >&2
    exit 1
fi

tailscale status --json | python3 -c '
import json, sys
state = json.load(sys.stdin)
if state.get("BackendState") != "Running":
    sys.exit("Tailscale must be running on this server. Run tailscale up first.")
'
# Avoid publishing private Serve handlers or replacing an existing service.
tailscale serve status --json | python3 -c '
import json, sys
if json.load(sys.stdin):
    sys.exit("Existing Serve/Funnel configuration found; inspect tailscale serve status first.")
'

tailscale funnel --bg "$origin"
tailscale funnel status
echo 'Use the HTTPS address shown above plus /v1/embeddings on each Cubie.'
