# ⚠️ V1 API Setup Requirements

## Important Differences from v3

The v1 API has **completely different requirements** than v3:

| Aspect | v3 (SDK) | v1 (REST) |
|--------|----------|-----------|
| **Auth** | Api-Key | IAM Token (Bearer) |
| **URL** | gRPC SDK | `https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize` |
| **Format** | SDK methods | Form data |
| **Text** | Direct string | URL-encoded |
| **Extra** | None | `folderId` required |
| **Output** | WAV/OGG | OGG-Opus (default) |

## Setup Steps for v1

### 1. Get IAM Token

v1 requires an **IAM token**, not an API key.

**Get IAM token:**
```bash
# Using yc CLI
yc iam create-token

# Or use service account key
export YANDEX_TOKEN=$(yc iam create-token)
```

**Or get from Yandex Cloud Console:**
1. Go to: https://console.cloud.yandex.ru/
2. Service Accounts → Your Service Account
3. Tokens → Create token

### 2. Get Folder ID

v1 requires your cloud folder ID.

**Find folder ID:**
```bash
yc resource-manager folder list
```

**Or in console:** Copy folder ID from folder page.

### 3. Update Configuration

**Option A: Set environment variables**
```bash
export YANDEX_TOKEN="your_iam_token_here"
export YANDEX_FOLDER_ID="your_folder_id_here"
```

**Option B: Add to user.json**
```json
{
  "yandex_tts_api_version": "v1",
  "yandex_folder_id": "b1gxxxxxxxxxxxxxxxxx"
}
```

### 4. Modify tts_engine.py for IAM Token

Currently, v1 uses `self.api_key` which is your API key. We need to:

**Option A: Use environment variable for IAM token**
```python
# In YandexTTSEngine.__init__()
self.iam_token = os.environ.get("YANDEX_TOKEN") or config.get("yandex_token")
self.folder_id = os.environ.get("YANDEX_FOLDER_ID") or config.get("yandex_folder_id")
```

**Option B: Add folderId to request**
```python
# In _synthesize_v1()
data = {
    "text": text,
    "lang": lang_code,
    "voice": voice_name,
    "folderId": self.folder_id,  # Add this!
}
```

### 5. Update Headers

v1 uses **Bearer** auth:
```python
headers = {
    "Authorization": f"Bearer {self.iam_token}",  # Not Api-Key!
}
```

## Code Changes Needed

### Add to `__init__`:
```python
# Add these lines after getting api_key
self.iam_token = os.environ.get("YANDEX_TOKEN")
self.folder_id = config.get("yandex_folder_id")

if not self.iam_token:
    raise ValueError("YANDEX_TOKEN not found (required for v1)")

if not self.folder_id:
    raise ValueError("yandex_folder_id not found in config (required for v1)")
```

### Add to `_synthesize_v1()`:
```python
data = {
    "text": text,
    "lang": lang_code,
    "voice": voice_name,
    "folderId": self.folder_id,  # Add this!
}

headers = {
    "Authorization": f"Bearer {self.iam_token}",  # Use IAM token
}
```

## Why v1 is More Complex

1. **Legacy API** - v1 is older, less convenient
2. **IAM tokens** - Need to be refreshed (expire after 12 hours)
3. **Folder ID** - Additional parameter
4. **Different format** - OGG-Opus instead of WAV

## Recommendation

**Stick with v3** for simplicity, OR:

**Switch to a different TTS provider** that has:
- Simple API keys
- Per-character pricing
- Easy setup

**OR**

**Don't split sentences** to save money with v3:
- Buffer 2-3 sentences together
- Pay $0.001333 for 500+ chars instead of 3 separate requests
- 80% savings without API complexity

## Alternative: Just Buffer Sentences

Instead of dealing with v1 complexity, just modify `ConversationManager` to buffer sentences:

```python
# Instead of:
# send "Привет!" → $0.001333
# send "Как дела?" → $0.001333
# Total: $0.002666

# Do this:
buffered = "Привет! Как дела?"
send buffered → $0.001333
Savings: 50%
```

No API changes needed, no auth complexity, same voice quality!

## Summary

v1 is more trouble than it's worth for your use case. I'd recommend:
1. **Keep v3** (current setup)
2. **Buffer sentences** (simple code change, 50% savings)
3. **Or switch to ElevenLabs** (if you want cheaper per-character pricing)

The v1 implementation is here for reference, but it's not worth the complexity for your sentence-splitting pattern.
