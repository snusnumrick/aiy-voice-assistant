# Yandex TTS API Version Configuration Guide

## Overview

The `YandexTTSEngine` now supports **both API v1 and v3**, switchable via configuration. This allows you to:
- **Save money** by using v1 for short sentences (per-character pricing)
- **Keep compatibility** by using v3 SDK (per-request pricing)
- **Test both** without code changes

## Configuration

### Method 1: In config.json

Add this to your `config.json` file:

```json
{
  "yandex_tts_api_version": "v1"
}
```

Or for v3:
```json
{
  "yandex_tts_api_version": "v3"
}
```

### Method 2: In user.json

Add to `user.json` (takes precedence over config.json):

```json
{
  "yandex_tts_api_version": "v1"
}
```

### Method 3: In .env file

Add to your `.env` file:
```
YANDEX_TTS_API_VERSION=v1
```

## API Comparison

### API v1 (Recommended for Short Sentences) ⭐

**Pricing:** $0.000011 per character
**Best for:** Short sentences, greetings, weather queries
**Pros:**
- ✅ **39% savings** for your usage pattern
- ✅ Fair pricing - pay only for characters used
- ✅ No sentence buffering needed
- ✅ Native async support
- ✅ Direct REST API

**Cons:**
- ❌ Different API endpoint
- ❌ Manual request handling

**Example cost:**
```
"Привет!" (7 chars) = $0.000077
"Как дела?" (10 chars) = $0.000110
```

### API v3 (Default - SDK)

**Pricing:** $0.001333 per 250-character billing unit
**Best for:** Long responses, if you prefer SDK convenience
**Pros:**
- ✅ Easy to use SDK
- ✅ High-level abstraction
- ✅ No need to handle HTTP requests
- ✅ Officially supported

**Cons:**
- ❌ Overpays for short sentences (94% overcharge on greetings)
- ❌ Executor needed for async (not truly async)
- ❌ More expensive with sentence splitting

**Example cost:**
```
"Привет!" (7 chars) = $0.001333 (pays for 250 chars!)
"Как дела?" (10 chars) = $0.001333
```

## Current Usage Analysis

Based on your conversation history:

| Metric | Value |
|--------|-------|
| Total sentences | 7,076 |
| Avg sentence length | 70.2 chars |
| Very short (< 25 chars) | 17.0% |
| Short (26-50 chars) | 23.3% |
| **With v3 (current)** | **$9.43/month** |
| **With v1 (recommended)** | **$5.74/month** |
| **Savings** | **$3.70/month (39%)** |

## Quick Start

### Switch to v1 (Save Money)

1. Add to `user.json`:
```json
{
  "yandex_tts_api_version": "v1"
}
```

2. Restart your application

3. Monitor costs:
```bash
python check_tts_usage.py
```

### Switch back to v3 (If needed)

1. Change `user.json`:
```json
{
  "yandex_tts_api_version": "v3"
}
```

2. Restart your application

## Testing

### Test v1 API Directly

```python
from src.tts_engine import YandexTTSEngine
from src.config import Config

config = Config()
# Force v1 in config
config.data['yandex_tts_api_version'] = 'v1'

engine = YandexTTSEngine(config)

# Test synthesis
engine.synthesize(
    text="Привет! Как дела?",
    filename="test_v1.wav",
    lang=Language.RUSSIAN
)
```

### Check Logs

View the logs to see which API version is being used:

```bash
# See API version in logs
tail -f logs/assistant.log | grep "Using Yandex TTS API"

# Check usage costs
python check_tts_usage.py
```

## Implementation Details

### Code Structure

The engine now has:
- **`__init__`**: Reads config and sets `self.api_version`
- **`_synthesize_v1`**: Private method for v1 API calls
- **`_synthesize_v1_async`**: Private async method for v1
- **`synthesize`**: Routes to v1 or v3 based on config
- **`synthesize_async`**: Routes to v1 or v3 based on config

### Routing Logic

```python
if self.api_version == "v1":
    # Use direct REST API
    self._synthesize_v1(...)
else:
    # Use SDK (v3)
    model = self.voice_model(...)
    result = model.synthesize(...)
```

### Logging

Usage logging now includes API version:
```json
{
  "timestamp": "2026-01-02T16:16:39",
  "engine": "YandexTTSEngine",
  "api_version": "v1",
  "text_length": 142,
  "cost_usd": 0.001562,
  ...
}
```

## Recommendation

**Use v1** because:
- ✅ Saves $3.70/month (39% savings)
- ✅ Better for your sentence-splitting pattern
- ✅ Fair pricing for short texts
- ✅ No impact on user experience
- ✅ Native async support

**Keep v3 if:**
- You have very long responses (500+ chars)
- You prefer SDK abstraction
- You don't want to change API endpoints

## Migration Checklist

- [ ] Add `yandex_tts_api_version` to config
- [ ] Test with a few sentences
- [ ] Monitor usage logs
- [ ] Check cost savings
- [ ] Update documentation
- [ ] Remove old config if switching permanently

## Troubleshooting

### "Invalid API version" warning
- Check spelling: should be `"v1"` or `"v3"` (lowercase)
- Default is v3 if invalid

### v1 not working
- Verify API key is set
- Check network connectivity
- Verify endpoint: `https://tts.api.cloud.yandex.net/v1/synthesize`

### v3 not working
- Check if SDK is installed: `poetry show yandex-speechkit`
- Verify credentials configured
- Check SDK version

### High costs with v1
- v1 should be cheaper - check your usage
- May be from other sources (OpenAI, Google, etc.)
- Run `check_tts_usage.py` to analyze

## Support

For issues:
1. Check logs: `logs/assistant.log`
2. Check usage: `python check_tts_usage.py`
3. Test both APIs with sample text
4. Review this documentation
