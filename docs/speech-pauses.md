# Speech pauses

The LLM emits **application cues** `<[tiny]>`, `<[small]>`, `<[medium]>`,
`<[large]>`, and `<[huge]>` in every spoken language. These remain canonical
through streaming, language routing and sentence buffering. They are removed
from stored conversation history and recognition context.

Each synthesis attempt renders its own provider format. The fallback receives
the original canonical text, never the first provider's rendered request.

| Provider/model | Request text |
| --- | --- |
| Yandex | Original context-dependent `<[size]>` cues |
| ElevenLabs Multilingual v2, Turbo v2/v2.5, Flash v2/v2.5 | SSML `<break time="…s" />` |
| ElevenLabs v3/v4 | `[short pause]` for tiny/small/medium; `[long pause]` for large/huge |
| Other engines or unrecognized ElevenLabs models | Plain speech with cues removed |

The SSML mapping is an application approximation: tiny = 0.1s, small = 0.3s,
medium = 0.6s, large = 1.0s, huge = 2.0s. These are not claimed equivalents of
Yandex's context-dependent timing. v3/v4 audio cues provide coarser control.
No model, voice or user configuration is changed.

Request limits apply **after** rendering. Splitting keeps complete pause tags
together, including SSML attributes containing spaces. Direct synchronous and
asynchronous engine calls also render cues before history lookup and payload
construction. Pause-only fragments are deferred to speech; a pause-only request
is never sent. No silence-only file is synthesized.

Unknown or unfinished `<[...` metadata is hidden. Known malformed HTML wrappers
are recovered to canonical cues. Errors identify `tag=pause` and a fixed reason,
without the surrounding speech or payload. Streaming scans wait for completion
and report truncated metadata only at EOF.

## Provider references

Checked 2026-10-23:

- [Yandex TTS markup](https://yandex.cloud/en/docs/speechkit/tts/markup/tts-markup):
  named context-dependent pauses; markup-only synthesis is unsupported, and
  sentence-edge pauses may be ignored.
- [ElevenLabs pauses](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-can-i-add-pauses):
  v2 break syntax and the three-second limit; v3/v4 do not support SSML breaks.
- [ElevenLabs prompting](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices):
  short/long pause audio tags and their applicability to v3/v4.

Tests intercept the actual HTTP/SDK payloads and simulate fallback, chunking and
mixed-language streaming. They use synthetic neutral text, mocked responses and
no live speech calls. Audible duration and voice-dependent delivery still need
a separately authorized listening check.
