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
