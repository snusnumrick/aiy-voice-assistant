# WeSpeaker embedding service

This service runs a speaker-specific WeSpeaker ResNet34-LM ONNX model away from the
Raspberry Pi. It accepts the assistant's 16 kHz mono PCM WAV and returns a normalized,
model-namespaced speaker embedding.

Build and run locally:

```bash
docker build -t cubie-wespeaker speaker_server
docker run --rm -p 8080:8080 \
  -e WESPEAKER_API_KEY=replace-with-a-random-secret \
  cubie-wespeaker
```

Test it:

```bash
curl --fail \
  -H 'Authorization: Bearer replace-with-a-random-secret' \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  http://127.0.0.1:8080/v1/embeddings
```

Configure each Cubie with the public HTTPS endpoint and the same bearer token:

```json
{
  "speaker_embedding_provider": "wespeaker",
  "wespeaker_embedding_url": "https://speaker.example/v1/embeddings"
}
```

Store `WESPEAKER_API_KEY` in `.env`, not tracked configuration. The token protects the
HTTP endpoint; it is unrelated to using voice as authentication.

## Public HTTPS access from the current server

The current deployment uses `https://ubuntu-desktop.tail14ac87.ts.net/v1/embeddings`.
On `192.168.50.254`, `cubie-wespeaker-public` publishes only `127.0.0.1:8082` to
Funnel and uses the same image and bearer token as `cubie-wespeaker-lan` on LAN port
8081. Both containers restart automatically. To recreate this Funnel after disabling
it, run `sudo bash scripts/expose_speaker_server.sh 8082` on that server.

Run Tailscale Funnel **on the machine hosting the container**. Remote Cubies do not need
Tailscale. Funnel provides the public HTTPS address and certificate without router port
forwarding. See the [Funnel CLI documentation](https://tailscale.com/docs/reference/tailscale-cli/funnel).

Keep the existing service and its `WESPEAKER_API_KEY`. For a new installation, put a random
token in an untracked `.env` file, then start the container with automatic restart:

```bash
docker run -d --name cubie-wespeaker --restart unless-stopped \
  -p 127.0.0.1:8081:8080 --env-file .env cubie-wespeaker
```

The `.env` file must contain a nonempty `WESPEAKER_API_KEY`. If a container already exists,
inspect its deployment before replacing it; preserve its model and token.

Install and sign in to Tailscale on the server, then run from this repository:

```bash
tailscale up
bash scripts/expose_speaker_server.sh 8081
```

On Linux, these Tailscale operations may need `sudo` or an authorized Tailscale operator.
Approve HTTPS/Funnel enablement in the browser if Tailscale prompts for it. The script
checks service health and rejects an endpoint without bearer protection. It also stops
if Serve/Funnel already has configuration, so existing services can be reviewed first.
`--bg` keeps the Funnel configuration active after the command exits and across restarts.
Keep the server awake and Tailscale running; do not use the Cubie Tailscale-down schedule
on the server.

Set `wespeaker_embedding_url` in each Cubie's `user.json` to the printed HTTPS address
plus `/v1/embeddings`, and put the server's same `WESPEAKER_API_KEY` in each Cubie's `.env`.
Restart the assistant after changing configuration. Keep the same model to preserve
compatibility with existing speaker profiles.

Before switching clients, verify `/healthz` over the public address, confirm an
unauthenticated POST to `/v1/embeddings` returns `401`, and send a real 16 kHz PCM WAV
with the bearer token to confirm a successful embedding. Repeat from a connection
outside the LAN, such as cellular data. Do not paste tokens into logs or tracked files.

Inspect or disable this Funnel with:

```bash
tailscale funnel status
tailscale funnel --https=443 off
```

The container keeps no speaker profiles or other mutable state, so the same image can be
deployed in more than one region and can safely scale to zero. Each Cubie still performs the
profile comparison locally. If profiles should be shared, synchronize `speaker_profiles.json`
separately; do not route it through this embedding endpoint.

The image downloads the official WeSpeaker `voxceleb_resnet34_LM.onnx` artifact from a pinned
Hugging Face revision and verifies its SHA-256. The upstream multilingual SimAM-ResNet34 is a
good later comparison, but its official download host currently fails TLS verification, so it is
not the default build artifact. A different compatible WeSpeaker ONNX model can be selected with
the `WESPEAKER_MODEL_URL`, `WESPEAKER_MODEL_SHA256`, and `WESPEAKER_MODEL_ID` build arguments.
When changing models, also set `wespeaker_embedding_dimension` on each Cubie to the new model's
output size.

The WeSpeaker toolkit is Apache-2.0. Pretrained-model licensing follows the corresponding
training dataset; review WeSpeaker's model documentation before redistribution.
