# Qwen3-TTS OpenAI-Compatible FastAPI Server

Serve Qwen3-TTS behind the OpenAI `POST /v1/audio/speech` interface, with optional voice cloning, saved voice profiles, real-time PCM streaming, and CUDA/ROCm backends.

This repository is based on the Qwen3-TTS implementation from the Alibaba Qwen team and adds an API/deployment layer intended for local applications and self-hosted services.

## Highlights

- OpenAI-compatible `POST /v1/audio/speech`
- Model and voice discovery under `/v1/models` and `/v1/voices`
- MP3, Opus, AAC, FLAC, WAV, and signed 16-bit PCM output
- Official and optimized backends
- Base-model voice cloning through `/v1/audio/voice-clone`
- Persistent voice-library profiles through `voice="clone:ProfileName"`
- Lazy model loading, bounded generation concurrency, warmup, and health checks
- Automatic long-text chunking with punctuation-aware boundaries
- Docker, NVIDIA GPU, and AMD ROCm deployment paths
- Optional Gradio Voice Studio and browser interface

## Important: choose the correct model type

Qwen3-TTS exposes different checkpoint families with different generation methods.

| Checkpoint type | Use it for | Do not use it for |
|---|---|---|
| `*-CustomVoice` | Preset speakers such as Vivian, Ryan, Serena, Dylan, and others | Reference-audio voice cloning |
| `*-Base` | `/v1/audio/voice-clone` and saved `clone:` profiles | Preset-speaker `/v1/audio/speech` requests |
| `*-VoiceDesign` | Voice design workflows supported by the underlying model/backend | Assuming preset-speaker or Base-model semantics |

For normal OpenAI-style TTS, start with a **CustomVoice** checkpoint. For voice cloning, run a **Base** checkpoint and call the clone endpoint.

## Requirements

- Python 3.10-3.12 recommended
- FFmpeg for MP3, Opus, AAC, or FLAC responses
- A backend-appropriate PyTorch/CUDA/ROCm installation for GPU use
- Enough RAM/VRAM for the selected checkpoint

WAV and PCM output do not require FFmpeg.

## Quick start

```bash
git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git
cd Qwen3-TTS-Openai-Fastapi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[api]"

# Default backend: official, 1.7B CustomVoice
python -m api.main
```

The server listens on `http://localhost:8880` by default.

Useful URLs:

- Web interface: `http://localhost:8880/`
- Swagger: `http://localhost:8880/docs`
- Health: `http://localhost:8880/health`
- Models: `http://localhost:8880/v1/models`
- Voices: `http://localhost:8880/v1/voices`

The backend loads lazily by default, so `/health` can report `initializing` until the first synthesis request.

## OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="not-needed",
)

response = client.audio.speech.create(
    model="tts-1",
    voice="Ryan",
    input="Hello from a local Qwen3-TTS server.",
    response_format="mp3",
    speed=1.0,
)
response.stream_to_file("speech.mp3")
```

OpenAI voice aliases are accepted:

| Alias | Qwen voice |
|---|---|
| `alloy` | Vivian |
| `echo` | Ryan |
| `fable` | Sophia |
| `nova` | Isabella |
| `onyx` | Evan |
| `shimmer` | Lily |

The exact native speaker list depends on the selected checkpoint and backend. Query `/v1/voices` rather than hard-coding the table above.

## cURL

```bash
curl --fail --show-error \
  http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "Ryan",
    "input": "This response is a WAV file.",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Language-specific model aliases

The API accepts aliases such as `tts-1-es` and `tts-1-hd-fr`. The suffix forces the language passed to the backend.

Supported suffixes:

- `en` English
- `zh` Chinese
- `ja` Japanese
- `ko` Korean
- `de` German
- `fr` French
- `es` Spanish
- `ru` Russian
- `pt` Portuguese
- `it` Italian

Example:

```python
response = client.audio.speech.create(
    model="tts-1-es",
    voice="Ryan",
    input="Hola, esta solicitud fuerza la salida en español.",
    response_format="wav",
)
response.stream_to_file("hola.wav")
```

## Audio formats

| `response_format` | MIME type | Notes |
|---|---|---|
| `mp3` | `audio/mpeg` | Requires FFmpeg |
| `opus` | `audio/opus` | Requires FFmpeg with Opus support |
| `aac` | `audio/aac` | ADTS AAC; requires FFmpeg |
| `flac` | `audio/flac` | Requires an available encoder |
| `wav` | `audio/wav` | PCM WAV container |
| `pcm` | `audio/pcm` | Headerless mono signed 16-bit little-endian PCM |

Encoding errors fail clearly. The server does **not** return WAV bytes while claiming a compressed content type.

## Streaming

Streaming is controlled by the OpenAI `stream_format` request field (not a boolean `stream` flag):

- Omit `stream_format` to receive the complete audio in a single response.
- `stream_format: "audio"` streams raw audio bytes over HTTP chunked transfer. `pcm` and `wav` are yielded **incrementally** as the optimized backend decodes (low latency). Compressed formats (mp3/opus/aac/flac) cannot be produced incrementally, so the model stream is drained, encoded once into a single container, and then byte-chunked — the client still receives one valid container, but without time-to-first-byte savings.
- `stream_format: "sse"` streams OpenAI `speech.audio.*` Server-Sent Events: `speech.audio.delta` (base64 audio per event) and a terminal `speech.audio.done` (with a `usage` object). This is the format OpenAI SDK / SSE-only clients expect.

Streaming requires `speed` to be `1.0` (or omitted). This Qwen3-TTS implementation applies speed adjustment to the fully-generated audio (via `librosa`) and cannot change speed while streaming; a request with `stream_format` set and `speed != 1.0` fails with HTTP 400 (`streaming_speed_unsupported`).

```python
import httpx
import numpy as np
import sounddevice as sd

request = {
    "model": "tts-1",
    "voice": "Ryan",
    "input": "This audio is streamed as signed sixteen-bit PCM.",
    "response_format": "pcm",
    "stream_format": "audio",
}

with httpx.stream(
    "POST",
    "http://localhost:8880/v1/audio/speech",
    json=request,
    timeout=None,
) as response:
    response.raise_for_status()
    pcm = np.frombuffer(b"".join(response.iter_bytes()), dtype="<i2")

sd.play(pcm, samplerate=24000)
sd.wait()
```

SSE example:

```bash
curl -N -X POST http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "Vivian",
    "input": "Streaming as Server-Sent Events.",
    "response_format": "pcm",
    "stream_format": "sse"
  }'
```

## Voice style instructions

Use the OpenAI `instructions` field for voice style/emotion control (forwarded to the model's `instruct` parameter). The older `instruct` request field is no longer accepted — use `instructions`.

```python
response = client.audio.speech.create(
    model="tts-1",
    voice="Vivian",
    input="I am so excited!",
    instructions="Speak with great enthusiasm.",
    response_format="mp3",
)
response.stream_to_file("excited.mp3")
```

## Backend selection

Set `TTS_BACKEND` before starting the server.

| Backend | Value | Recommended use |
|---|---|---|
| Official | `official` | Default, broad feature compatibility |
| Optimized | `optimized` | GPU production, model switching, native PCM streaming, voice library |

### Official backend

```bash
export TTS_BACKEND=official
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
python -m api.main
```

### Optimized backend

The optimized backend reads `config.yaml` from `~/qwen3-tts/config.yaml` unless `TTS_CONFIG` points elsewhere.

```bash
mkdir -p ~/qwen3-tts
cp config.yaml ~/qwen3-tts/config.yaml

TTS_BACKEND=optimized python -m api.main
```

Edit the model entries in `config.yaml` to use local paths or Hugging Face IDs. The configured `type` must match the checkpoint: `customvoice`, `base`, or `voice_design`. Models are loaded **only from local directories** — this build does not use the Hugging Face cache and never downloads. Set `TTS_MODELS_DIR` to a folder containing one subdir per model named after the HF repo without the org prefix (e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `/MODELS/Qwen3-TTS-12Hz-1.7B-Base`). The app resolves each `hf_id` to `<TTS_MODELS_DIR>/<repo-name-without-org>` and raises if it isn't present locally. `verify_models.py` at the repo root compares a `TTS_MODELS_DIR` folder against a Hugging Face cache (size + SHA-256, driven by `config.yaml`'s `models` section) — useful when migrating from a cache to the local folder.

`config.yaml` also accepts an optional `voice_design_model` key naming a `voice_design`-type model. The optimized backend has no `generate_voice_design` API path, so VoiceDesign is run via the **official** backend (`TTS_BACKEND=official` + `TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) or `qwen_tts` directly; the key just makes the model discoverable in one shared config.

## Voice cloning

Run a Base checkpoint:

```bash
TTS_BACKEND=official \
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
python -m api.main
```

Then send base64-encoded reference audio:

```python
import base64
import requests

with open("reference.wav", "rb") as file:
    reference = base64.b64encode(file.read()).decode("ascii")

response = requests.post(
    "http://localhost:8880/v1/audio/voice-clone",
    json={
        "input": "This sentence uses the reference speaker.",
        "ref_audio": reference,
        "ref_text": "The exact transcript spoken in reference.wav.",
        "x_vector_only_mode": False,
        "language": "English",
        "response_format": "wav",
        "speed": 1.0,
    },
    timeout=300,
)
response.raise_for_status()
open("clone.wav", "wb").write(response.content)
```

Modes:

- ICL: `x_vector_only_mode=false`; requires an accurate `ref_text`
- X-vector: `x_vector_only_mode=true`; transcript optional, usually lower fidelity

Only clone voices you have permission to use. Do not use the service to impersonate people deceptively.

## Voice library

Saved profiles are discovered under:

```text
$VOICE_LIBRARY_DIR/profiles/
└── alice/
    ├── meta.json
    └── reference.wav
```

Example `meta.json`:

```json
{
  "name": "Alice",
  "profile_id": "alice",
  "ref_audio_filename": "reference.wav",
  "ref_text": "Transcript of the reference clip.",
  "x_vector_only_mode": false,
  "language": "English"
}
```

Use the profile through the normal OpenAI endpoint:

```python
response = client.audio.speech.create(
    model="tts-1",
    voice="clone:Alice",
    input="This uses the saved Alice profile.",
    response_format="wav",
)
response.stream_to_file("alice.wav")
```

The active backend/checkpoint must support voice cloning. See `docs/voice-library.md` for details.

## Docker

### NVIDIA GPU

```bash
docker compose up --build qwen3-tts-gpu
```

Override the host port or model without editing Compose:

```bash
TTS_PORT=9000 \
TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
docker compose up --build qwen3-tts-gpu
```

The Compose file requests one GPU instead of hard-coding a host GPU index.

### AMD ROCm

```bash
docker compose -f docker-compose.rocm.yml up --build qwen3-tts-rocm
```

Review device mappings in `docker-compose.rocm.yml`; render-node names vary between hosts.

## Configuration reference

| Variable | Default | Purpose |
|---|---:|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8880` | Listen port |
| `WORKERS` | `1` | Uvicorn worker processes; each process loads its own model |
| `TTS_BACKEND` | `official` | Backend selector |
| `TTS_MODEL_ID` / `TTS_MODEL_NAME` (legacy alias) | backend-specific | Hugging Face ID or local model path (`TTS_MODEL_ID` takes precedence) |
| `TTS_LAZY_LOAD` | `true` | Load on first synthesis request |
| `TTS_WARMUP_ON_START` | `false` | Warm regular and supported streaming paths |
| `TTS_WARMUP_MAX_SECONDS` | `10` | Timeout per warmup request |
| `TTS_MAX_CONCURRENT` | `1` | Concurrent generation limit per process |
| `TTS_IDLE_TIMEOUT_SECONDS` | `0` | Opt-in idle shutdown; `0` disables it |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |
| `API_KEY` | *(unset)* | When set, requires `Authorization: Bearer <key>` or `X-API-Key: <key>` on all routes except `/health`. Unset = open |
| `UI_USER` | *(unset)* | Browser UI username (HTTP Basic auth); must be set together with `UI_PASSWORD` |
| `UI_PASSWORD` | *(unset)* | Browser UI password (HTTP Basic auth); must be set together with `UI_USER` |
| `ENABLE_VOICE_STUDIO` | `false` | Mount Gradio at `/voice-studio` |
| `VOICE_LIBRARY_DIR` | `./voice_library` | Saved profile root |
| `TTS_CUSTOM_VOICES` | `./custom_voices` | Legacy/custom voice directory |
| `TTS_CONFIG` | `~/qwen3-tts/config.yaml` | Optimized-backend YAML |
| `TTS_MODELS_DIR` | *(required)* | Folder of local model snapshots (one subdir per model, repo name without org prefix); models are loaded only from here — no HF cache, no downloads |
| `GPU_KEEPALIVE_INTERVAL` | `0` | Optional GPU keepalive interval in seconds |
| `TTS_AUTOCHUNK` | `true` | Enable punctuation-aware input splitting for **non-streaming** requests |
| `TTS_STREAM_AUTOCHUNK` | `true` | Enable chunking for **streaming** requests (independent of `TTS_AUTOCHUNK`; bounds peak VRAM by giving each chunk its own short-lived KV cache) |
| `TTS_MIN_CHUNK_CHARS` | `20` | Soft minimum chunk length (shared by both paths) |
| `TTS_MAX_CHUNK_CHARS` | `70` | Target maximum chunk length (shared by both paths) |
| `TTS_CHUNK_GAP_MS` | `120` | Silence inserted between generated chunks (shared by both paths) |
| `TTS_PARAGRAPH_GAP_MS` | `250` | Silence inserted at Markdown block boundaries (headings, paragraph breaks, lists). Set to `0` to disable |

Invalid float settings fall back to safe defaults instead of crashing module import.

## CORS and network exposure

`CORS_ORIGINS=*` is convenient for local development. For a service exposed beyond localhost, set explicit origins:

```bash
CORS_ORIGINS=https://app.example.com,https://admin.example.com python -m api.main
```

The server has built-in authentication (see [Authentication](#authentication) below) — enable it before exposing the service to the internet. Keep `WORKERS=1` on a single GPU unless you intentionally have enough VRAM for one full model per worker.

## Authentication

By default the server is **open** (no credentials configured) — the same behavior as before, convenient for local and trusted deployments. Configure one or more credentials to require authentication on every route except `/health` (which stays open for orchestrator/load-balancer probes).

You may configure any combination of:

| Variable | Required with | Accepted via |
|---|---|---|
| `API_KEY` | — | `Authorization: Bearer <key>` and `X-API-Key: <key>` (programmatic clients) |
| `UI_USER` + `UI_PASSWORD` | both together | `Authorization: Basic <base64(user:password)>` (browser native login dialog) |

Each credential is **independent** — configure none (open), `API_KEY` only (clients), `UI_USER`+`UI_PASSWORD` only (browser), or both. On every protected request, a presented credential is validated against the store matching its scheme; a credential with no matching configured store is rejected (no fallback), and a request with no credential gets `401`.

### Browser (the web UI)

To log in to `http://127.0.0.1:8880/` (and `/docs`, `/redoc`, `/static/*`) from a browser, set the UI credentials:

```bash
UI_USER=admin
UI_PASSWORD=choose-a-password
```

Browse to `http://127.0.0.1:8880/` → the browser shows its native login dialog → enter `admin` / your password. It caches the credentials for the session, so `/`, `/docs`, and assets all load afterward. The Swagger UI's **Try it out** button is a separate concern: click **Authorize** and paste your `API_KEY` (Bearer) — that authenticates API calls, while the page itself is protected by Basic auth.

### API clients

For programmatic clients (OpenAI SDK, cURL, etc.), set `API_KEY`:

```bash
API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

```bash
curl http://127.0.0.1:8880/v1/models \
  -H "Authorization: Bearer $API_KEY"
```

The OpenAI Python client passes it through with `default_headers={"Authorization": f"Bearer {API_KEY}"}` (or set `X-API-Key`).

### Both at once

For a deployment that humans browse and scripts call, set all three:

```bash
API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
UI_USER=admin
UI_PASSWORD=choose-a-password
```

Browsers use Basic; API clients use Bearer/X-API-Key — each valid on its own.

### Notes

- Comparisons are constant-time (`hmac.compare_digest`) to avoid timing side channels.
- `401` responses include `WWW-Authenticate` advertising every configured scheme (e.g. `Basic realm="Qwen3-TTS API", Bearer`).
- `UI_USER` set without `UI_PASSWORD` logs a warning and leaves Basic auth disabled (no silent weakening).
- `/health` is always unauthenticated.
- **Voice Studio** (`/voice-studio`) manages its own routing and is not covered by this check; its backend calls to `/v1/*` are still protected. Avoid exposing Voice Studio on an untrusted network.

## Development and tests

```bash
pip install -e ".[api,dev]"
pytest -q
```

The regression suite covers URL normalization, currency/unit interactions, PCM/WAV correctness, truthful compressed-encoding failures, invalid environment values, and concurrent cold-start initialization.

## Troubleshooting

### Compressed output fails

Install FFmpeg and confirm the codec is available:

```bash
ffmpeg -version
ffmpeg -encoders | grep -E 'mp3|opus|aac|flac'
```

Use `response_format="wav"` while diagnosing. The API now reports an encoding error rather than silently returning a different format.

### Base model rejects a normal speech request

Base checkpoints clone reference voices. Switch to a `*-CustomVoice` checkpoint for preset speakers, or use `/v1/audio/voice-clone`.

### First request is slow

Model download, load, compilation, and graph capture can make the first request much slower. Use:

```bash
TTS_LAZY_LOAD=false TTS_WARMUP_ON_START=true python -m api.main
```

### Out of memory

- Use the 0.6B checkpoint
- Keep `WORKERS=1`
- Keep `TTS_MAX_CONCURRENT=1`
- Stop other GPU workloads

### Server exits while idle

Idle shutdown is disabled by default. Check that your environment does not set a positive `TTS_IDLE_TIMEOUT_SECONDS`.

## Project layout

```text
api/
├── backends/                 Backend implementations and factory
├── routers/                  OpenAI-compatible endpoints
├── services/                 Text normalization and audio encoding
├── static/                   Browser UI
└── structures/               Pydantic request/response schemas
config.yaml                   Optimized-backend model/performance config
docker-compose.yml            NVIDIA GPU service
docker-compose.rocm.yml       AMD ROCm service
gradio_voice_studio.py        Voice Studio
tests/                        Regression and API tests
```

## Upstream and license

Qwen3-TTS is developed by the Alibaba Qwen team. This repository's API and deployment additions retain the Apache-2.0 license. Review the upstream model cards and licenses for every checkpoint you deploy.
