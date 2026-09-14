# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import inspect
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..security import require_auth

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
)
from ..services.text_processing import normalize_text
from ..services.audio_encoding import (
    encode_audio,
    get_content_type,
    DEFAULT_SAMPLE_RATE,
    pcm_bytes_from_chunk,
    wav_header_bytes,
    iter_encoded_bytes,
)

logger = logging.getLogger(__name__)

# Concurrency cap: prevents simultaneous requests from starving GPU memory.
# Override with TTS_MAX_CONCURRENT env var (default 1 for single-GPU deployments).
try:
    _MAX_CONCURRENT = max(1, int(os.getenv("TTS_MAX_CONCURRENT", "1")))
except ValueError:
    logger.warning("Invalid TTS_MAX_CONCURRENT value; falling back to 1")
    _MAX_CONCURRENT = 1
_generation_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

# --- Auto-chunking -----------------------------------------------------------
# Input is split at punctuation into chunks sized to a [min, max] character
# window, each synthesized separately and the audio concatenated back together.
# This keeps every generation well under the backend's wall-clock cap for
# long sequences, and lowers first-audio latency. Inputs that fit in a
# single chunk take the original code path with zero overhead.
#
# Splitting prefers sentence punctuation (. ! ?), then clause punctuation
# (, ; :), then word boundaries. Chunks are packed greedily up to max_chars and
# kept at/above min_chars where possible (a too-small piece is merged with a
# neighbour as long as the result still fits within max_chars).
#   TTS_AUTOCHUNK=false        disable non-streaming chunking entirely
#   TTS_STREAM_AUTOCHUNK=false disable chunking for streaming requests
#   TTS_MIN_CHUNK_CHARS=20     soft lower bound per chunk (shared)
#   TTS_MAX_CHUNK_CHARS=70     hard upper bound per chunk (shared)
#   TTS_CHUNK_GAP_MS=120       silence inserted between merged chunks (shared)
#   TTS_PARAGRAPH_GAP_MS=250    silence inserted at markdown block boundaries
#                              (headings, paragraph breaks, lists) so they
#                              get a more pronounced pause than the regular
#                              inter-chunk gap. Falls back to TTS_CHUNK_GAP_MS.
try:
    _AUTOCHUNK = os.getenv("TTS_AUTOCHUNK", "true").lower() == "true"
    _STREAM_AUTOCHUNK = os.getenv("TTS_STREAM_AUTOCHUNK", "true").lower() == "true"
    _MIN_CHUNK_CHARS = max(1, int(os.getenv("TTS_MIN_CHUNK_CHARS", "20")))
    _MAX_CHUNK_CHARS = max(_MIN_CHUNK_CHARS, int(os.getenv("TTS_MAX_CHUNK_CHARS", "70")))
    _CHUNK_GAP_MS = max(0, int(os.getenv("TTS_CHUNK_GAP_MS", "120")))
    # Markdown block boundaries (heading, paragraph break, list) get a more
    # pronounced pause than regular inter-chunk gaps. Defaults to 250ms so
    # block transitions read as deliberate pauses; set to 0 to disable or to
    # a custom value to tune. When unset AND TTS_CHUNK_GAP_MS is explicitly 0,
    # this also defaults to 0 (respecting a "no silence anywhere" config).
    _default_paragraph_gap = "0" if _CHUNK_GAP_MS == 0 else "250"
    _PARAGRAPH_GAP_MS = max(0, int(os.getenv("TTS_PARAGRAPH_GAP_MS", _default_paragraph_gap)))
except ValueError:
    logger.warning("Invalid auto-chunk env value; using defaults")
    _AUTOCHUNK, _STREAM_AUTOCHUNK = True, True
    _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS, _CHUNK_GAP_MS = 20, 70, 120
    _PARAGRAPH_GAP_MS = 250


def _pieces(text: str, max_chars: int) -> List[Optional[str]]:
    """Break text into pieces each <= max_chars, splitting on hard newlines
    first (paragraph boundaries), then sentence punctuation (. ! ?), then
    clause punctuation (, ; :), then words.

    Returns a list of piece strings interspersed with ``None`` sentinels that
    mark hard paragraph/block boundaries the chunk packer must never cross, so
    blocks become separate chunks and the merge step inserts real silence
    between them.
    """
    out: List[Optional[str]] = []
    # Hard-split on newlines first so blocks never merge into one piece.
    paragraphs = re.split(r"\n+", text.strip())
    for p_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if p_idx:
            # Hard boundary between blocks.
            out.append(None)
        for sent in re.split(r"(?<=[.!?])\s+", paragraph):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                out.append(sent)
                continue
            for clause in re.split(r"(?<=[,;:])\s+", sent):
                clause = clause.strip()
                if not clause:
                    continue
                if len(clause) <= max_chars:
                    out.append(clause)
                    continue
                buf = ""
                for w in clause.split():
                    if not buf:
                        buf = w
                    elif len(buf) + 1 + len(w) <= max_chars:
                        buf += " " + w
                    else:
                        out.append(buf)
                        buf = w
                if buf:
                    out.append(buf)
    return out


class Chunk(NamedTuple):
    """A synthesized chunk and whether a Markdown block boundary precedes it.

    ``starts_block`` is True when the chunk begins a new Markdown block
    (heading, paragraph, list, etc.). The merge step uses the longer
    ``_PARAGRAPH_GAP_MS`` before such chunks and the regular
    ``_CHUNK_GAP_MS`` before sentence-level chunks.
    """
    text: str
    starts_block: bool = False


def _split_into_chunks(text: str, min_chars: int, max_chars: int) -> List[Chunk]:
    """Split text into chunks within a [min_chars, max_chars] window, breaking
    at punctuation. Pieces are packed greedily up to max_chars; a chunk shorter
    than min_chars is merged into a neighbour when the result still fits.

    Each returned ``Chunk`` records whether it starts a new Markdown block
    (``starts_block=True``) so callers can insert a more pronounced pause there.
    """
    pieces = _pieces(text, max_chars)
    if not pieces:
        return []
    # Greedy pack up to max_chars. ``None`` pieces are hard block boundaries:
    # flush the current buffer (and record the boundary) so blocks never merge
    # into one chunk and the merge step inserts real silence between them.
    # ``pending_block`` marks that the next emitted chunk starts a new block.
    packed: List[Optional[str]] = []
    pending_block = True  # the very first chunk always starts a block
    buf = ""
    for p in pieces:
        if p is None:
            if buf:
                packed.append(buf)
                buf = ""
            packed.append(None)
            pending_block = True
            continue
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += " " + p
        else:
            packed.append(buf)
            buf = p
    if buf:
        packed.append(buf)
    # Soft minimum: fold an undersized chunk into a neighbour if it still fits,
    # but never across a hard block boundary (None). Track which chunks start a
    # block so the merge step can use the right gap size.
    merged: List[Chunk] = []
    for c in packed:
        if c is None:
            pending_block = True
            continue
        if (
            merged
            and not pending_block
            and (len(c) < min_chars or len(merged[-1].text) < min_chars)
            and len(merged[-1].text) + 1 + len(c) <= max_chars
        ):
            merged[-1] = Chunk(merged[-1].text + " " + c, merged[-1].starts_block)
        else:
            merged.append(Chunk(c, pending_block))
            pending_block = False
    return [c for c in merged if c.text and c.text.strip()]
# -----------------------------------------------------------------------------

# Voice library: saved voice profiles used via the "clone:ProfileName" voice prefix.
# Configurable via VOICE_LIBRARY_DIR env var; defaults to ./voice_library.
VOICE_LIBRARY_DIR = Path(
    os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")
).resolve()

# In-process cache for reference audio reads (profile_name -> (audio_np, sample_rate)).
# Avoids re-reading and re-decoding the same WAV file on every request.
_ref_audio_cache: dict = {}

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
    # Enforce configured credentials (API_KEY and/or UI_USER+UI_PASSWORD)
    # on every route in this router when auth is enabled.
    dependencies=[Depends(require_auth)],
)


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Sophia",
    "nova": "Isabella",
    "onyx": "Evan",
    "shimmer": "Lily",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """
    Extract language from model name if it has a language suffix.
    
    Args:
        model_name: Model name (e.g., "tts-1-es", "tts-1-hd-fr")
    
    Returns:
        Language name if suffix found, None otherwise
    """
    # Check if model ends with a language code
    # Only extract language if the model follows the expected pattern
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            # Verify it's a valid language-specific model variant
            # Should be either tts-1-{lang} or tts-1-hd-{lang}
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library.

    Searches ``VOICE_LIBRARY_DIR/profiles/`` for a sub-directory whose
    ``meta.json`` matches the given *name_or_id* (case-insensitive name match
    or exact profile_id match).

    Returns a dict with keys:
        ref_audio_path, ref_text, x_vector_only_mode, language, name

    Raises:
        ValueError: if the profile is not found or its reference audio is missing.
    """
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")

    for child in sorted(profiles_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if (
            meta.get("profile_id") == name_or_id
            or meta.get("name", "").lower() == name_or_id.lower()
        ):
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio filename")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }

    raise ValueError(f"Voice profile not found: '{name_or_id}'")


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed.

    Honors lazy-load: if the backend hasn't been initialized yet, the
    first caller pays the model-load + warmup cost in one shot. The
    warmup is gated by the ``TTS_WARMUP_ON_START`` env var (which the
    factory re-reads on each call). Warmup absorbs the model-load and
    compile cost at first-load time instead of on the user's first
    request.
    """
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        await initialize_backend(warmup=warmup_enabled)

    return backend


def note_speech_activity(app, samples: int = 0) -> None:
    """Reset the idle-shutdown timer on a successful speech request.

    Called from the /v1/audio/speech handler. Read-only endpoints
    like /health do NOT call this, so they don't keep a quiet
    server alive forever.

    Args:
        app: The FastAPI app instance (``request.app`` from the route).
        samples: Number of audio samples generated, for stats.
    """
    import time as _time
    state = getattr(app, "state", None)
    if state is None:
        return
    state.last_speech_at = _time.monotonic()
    state.speech_request_count = getattr(state, "speech_request_count", 0) + 1
    state.speech_total_samples = getattr(state, "speech_total_samples", 0) + int(samples)


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


def _method_accepts_kwarg(method, kwarg: str) -> bool:
    """Return True if a callable accepts a given keyword argument."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    if kwarg in signature.parameters:
        return True

    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using the configured TTS backend.
    
    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    backend = await get_tts_backend()

    # Check custom voice BEFORE applying OpenAI alias mapping,
    # so custom voices with OpenAI alias names remain accessible.
    is_custom = backend.is_custom_voice(voice)
    # Map voice name (OpenAI aliases to internal names) for built-in voices.
    voice_name = voice if is_custom else get_voice_name(voice)

    async def _synth(segment: str) -> tuple[np.ndarray, int]:
        if is_custom:
            return await backend.generate_speech_with_custom_voice(
                text=segment,
                voice=voice,
                language=language,
                speed=speed,
            )
        return await backend.generate_speech(
            text=segment,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    # Decide chunking. A single chunk (or disabled) takes the original path.
    if _AUTOCHUNK:
        chunks = _split_into_chunks(text, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS)
    else:
        chunks = [Chunk(text, starts_block=True)] if text.strip() else []
    if len(chunks) <= 1:
        seg = chunks[0].text if chunks else text
        logger.debug(
            "TTS single chunk (no gaps): %r", seg,
        )
        try:
            return await _synth(seg)
        except Exception as e:
            raise RuntimeError(f"Speech generation failed: {e}")

    # Multi-chunk: synthesize each block, then merge with a gap. Markdown
    # block boundaries (headings, paragraphs, lists) use a more pronounced
    # gap (``_PARAGRAPH_GAP_MS``) than sentence-level splits within a
    # paragraph (``_CHUNK_GAP_MS``).
    logger.info(
        "Auto-chunking %d chars into %d chunks (window=%d-%d, gap=%dms, "
        "block-gap=%dms)",
        len(text), len(chunks), _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS,
        _CHUNK_GAP_MS, _PARAGRAPH_GAP_MS,
    )
    try:
        audios: List[np.ndarray] = []
        sr = DEFAULT_SAMPLE_RATE
        for i, chunk in enumerate(chunks):
            a, sr = await _synth(chunk.text)
            if a is not None and len(a):
                audios.append(np.asarray(a))
        if not audios:
            raise RuntimeError("no audio produced from any chunk")
        merged: List[np.ndarray] = []
        for i, (chunk, a) in enumerate(zip(chunks, audios)):
            if i:
                gap_ms = _PARAGRAPH_GAP_MS if chunk.starts_block else _CHUNK_GAP_MS
                gap_len = int(sr * gap_ms / 1000.0)
                logger.debug(
                    "TTS chunk %d/%d: text=%r silence_before=%dms (%s)",
                    i, len(chunks), chunk.text, gap_ms,
                    "block boundary" if chunk.starts_block else "sentence split",
                )
                if gap_len > 0:
                    merged.append(np.zeros(gap_len, dtype=audios[0].dtype))
            else:
                logger.debug(
                    "TTS chunk %d/%d: text=%r silence_before=0ms (first chunk)",
                    i, len(chunks), chunk.text,
                )
            merged.append(a)
        return np.concatenate(merged), sr
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


# ---------------------------------------------------------------------------
# Streaming helpers (shared by built-in-voice and clone: paths)
# ---------------------------------------------------------------------------
# The optimized (CUDA) backend exposes async generators that yield
# (pcm_chunk_float32, sample_rate) tuples while the model decodes. The helpers
# below turn those into the two OpenAI streaming envelopes:
#   - stream_format="audio" : one HTTP chunked byte stream (raw PCM, a single
#     WAV header + PCM, or a single encoded container byte-chunked).
#   - stream_format="sse"   : text/event-stream of speech.audio.delta /
#     speech.audio.done events (base64 audio per delta).
# Compressed formats (mp3/opus/aac/flac) cannot be produced incrementally, so
# the model stream is drained, encoded ONCE, and then byte-sliced.

_VALID_STREAM_FORMATS = ("pcm", "wav", "mp3", "opus", "aac", "flac")


def _validate_streaming_request(response_format: str, speed: float, stream_format: str):
    """Enforce the OpenAI streaming constraints up-front (raises HTTPException)."""
    if response_format not in _VALID_STREAM_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_format_for_streaming",
                "message": (
                    f"streaming only supports response_format in "
                    f"{list(_VALID_STREAM_FORMATS)}. Got '{response_format}'."
                ),
                "type": "invalid_request_error",
            },
        )
    if speed != 1.0:
        # This TTS implementation applies speed post-hoc via librosa on the
        # fully-assembled audio; it cannot be applied per-chunk on a stream.
        raise HTTPException(
            status_code=400,
            detail={
                "error": "streaming_speed_unsupported",
                "message": (
                    "speed != 1.0 is not supported together with stream_format. "
                    "This TTS implementation applies speed adjustment to the "
                    "fully-generated audio and cannot change speed while "
                    "streaming. Re-send with speed=1.0 (or omit it) for "
                    "streaming, or drop stream_format for non-streaming output."
                ),
                "type": "invalid_request_error",
            },
        )


def _sse_event(event_name: str, data: dict) -> bytes:
    """Serialize one Server-Sent Event (event: + data: JSON) as bytes."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")


def _encode_stream_chunk(
    pcm_chunk: np.ndarray, fmt: str, sr: int, wav_header_emitted: bool
) -> tuple:
    """Encode one PCM chunk for stream_format="audio".

    Returns (bytes_to_emit, wav_header_emitted_now). For WAV, the header is
    emitted exactly once before the first PCM chunk.
    """
    if fmt == "pcm":
        return pcm_bytes_from_chunk(pcm_chunk), wav_header_emitted
    if fmt == "wav":
        out = b""
        if not wav_header_emitted:
            out += wav_header_bytes(sr)
        out += pcm_bytes_from_chunk(pcm_chunk)
        return out, True
    # Compressed formats are not produced per-chunk; caller drains first.
    return b"", wav_header_emitted


async def _drain_and_encode(
    gen, fmt: str, default_sr: int
) -> tuple:
    """Drain a PCM generator to one array and encode once (compressed fmts)."""
    chunks: List[np.ndarray] = []
    sr = default_sr
    async for pcm_chunk, chunk_sr in gen:
        if pcm_chunk is not None and len(pcm_chunk) > 0:
            chunks.append(np.asarray(pcm_chunk))
            sr = chunk_sr
    if not chunks:
        raise RuntimeError("no audio produced")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    audio_bytes = await asyncio.to_thread(encode_audio, audio, fmt, sr)
    return audio_bytes, sr, len(audio)


async def _chunked_stream_drain(
    text_chunks: List[Chunk],
    make_gen,
    fmt: str,
    sr_default: int,
):
    """Drive a backend streaming generator *per text chunk* and unify output.

    Bounds peak VRAM for long streaming requests: each chunk is a separate
    ``stream_generate_*`` call with its own short-lived KV cache, freed before
    the next chunk starts (matching the non-streaming chunking path).

    ``make_gen(text) -> async_generator[(pcm_chunk, sr)]`` builds one backend
    streaming generator for a single chunk's text. The gap between chunks is a
    silence of ``_PARAGRAPH_GAP_MS`` ms (zeros) at Markdown block boundaries and
    ``_CHUNK_GAP_MS`` ms at sentence-level splits — identical to the
    non-streaming chunk path. For compressed formats no gap is inserted (the
    whole audio is encoded once).

    Yields ``(pcm_chunk, sr)`` for ``pcm``/``wav`` (caller encodes/headers).
    For compressed formats, accumulates all PCM into one array and yields a
    single ``(audio_np, sr)`` so the caller can encode it once (multiple
    separate container headers would be corrupt).
    """
    is_raw = fmt in ("pcm", "wav")
    accumulator: List[np.ndarray] = []
    sr = sr_default
    gap_cache: dict = {}
    emitted_any = False  # gate the inter-chunk gap on first *emitted* chunk

    def _gap_for(chunk: Chunk) -> Optional[np.ndarray]:
        if not is_raw:
            return None
        gap_ms = _PARAGRAPH_GAP_MS if chunk.starts_block else _CHUNK_GAP_MS
        if gap_ms <= 0:
            return None
        if gap_ms not in gap_cache:
            gap_cache[gap_ms] = np.zeros(
                int(sr * gap_ms / 1000.0), dtype=np.float32
            )
        return gap_cache[gap_ms]

    total = len(text_chunks)
    for idx, chunk in enumerate(text_chunks):
        seg = chunk.text if isinstance(chunk, Chunk) else chunk
        starts_block = chunk.starts_block if isinstance(chunk, Chunk) else False
        if not seg or not seg.strip():
            continue
        chunk_emitted = False
        async for pcm_chunk, chunk_sr in make_gen(seg):
            if pcm_chunk is None or len(pcm_chunk) == 0:
                continue
            pcm = np.asarray(pcm_chunk, dtype=np.float32)
            sr = chunk_sr if chunk_sr else sr
            if is_raw:
                # Insert the inter-chunk silence gap before the FIRST emitted
                # PCM of each text chunk, except the very first chunk overall.
                if emitted_any and not chunk_emitted:
                    gap = _gap_for(Chunk(seg, starts_block))
                    if gap is not None and len(gap) > 0:
                        yield gap, sr
                yield pcm, sr
                emitted_any = True
                chunk_emitted = True
            else:
                accumulator.append(pcm)
                emitted_any = True
                chunk_emitted = True
        if not chunk_emitted:
            logger.warning("Streaming chunk %d/%d produced no audio", idx + 1, total)
        if emitted_any and is_raw:
            gap_ms = _PARAGRAPH_GAP_MS if starts_block else _CHUNK_GAP_MS
            logger.debug(
                "TTS stream chunk %d/%d: text=%r silence_before=%dms (%s)",
                idx + 1, total, seg, gap_ms,
                "block boundary" if starts_block else "sentence split",
            )
    if not is_raw:
        if not accumulator:
            raise RuntimeError("no audio produced from any chunk")
        audio = (
            np.concatenate(accumulator) if len(accumulator) > 1 else accumulator[0]
        )
        yield audio, sr


def _stream_headers(fmt: str, stream_format: str) -> dict:
    """Build response headers for a streaming speech response."""
    if stream_format == "sse":
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        }
    return {
        "Content-Disposition": f"attachment; filename=speech.{fmt}",
        "Cache-Control": "no-cache",
    }


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """
    OpenAI-compatible endpoint for text-to-speech.

    Generates audio from the input text using the specified voice and model.

    **Voice library:** pass ``voice: "clone:ProfileName"`` to use a saved voice
    profile from the voice library (``VOICE_LIBRARY_DIR/profiles/``).  The
    server automatically switches to the Base model for profile-based cloning.

    **Streaming:** set ``stream_format`` to ``"audio"`` (HTTP chunked raw audio
    bytes) or ``"sse"`` (OpenAI ``speech.audio.*`` Server-Sent Events). When
    omitted, the complete audio is returned in a single response. Real-time,
    incremental streaming is available for ``response_format`` ``pcm`` and
    ``wav``; compressed formats (mp3/opus/aac/flac) are encoded once and then
    byte-chunked. ``speed`` other than 1.0 is not supported together with
    ``stream_format`` (requires the *optimized* backend for incremental
    generation; other backends still work via drain-then-chunk).
    """
    # Validate model
    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )
    
    try:
        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)
        
        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )
        
        # Extract language from model name if present, otherwise use request language
        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")

        # ----------------------------------------------------------------
        # Voice library: "clone:ProfileName" -> load profile + voice clone
        # ----------------------------------------------------------------
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[len("clone:"):].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": (
                            "The 'clone:' prefix requires a profile name, "
                            "e.g. voice='clone:MyVoice'"
                        ),
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as exc:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "profile_not_found",
                        "message": str(exc),
                        "type": "invalid_request_error",
                    },
                )

            backend = await get_tts_backend()

            # Check that voice cloning is supported by the current backend/model
            if not backend.supports_voice_cloning():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "voice_cloning_not_supported",
                        "message": (
                            "Voice library cloning requires a Base model and the "
                            "optimized backend (TTS_BACKEND=optimized), or a backend "
                            "that supports voice cloning."
                        ),
                        "type": "invalid_request_error",
                    },
                )

            # ICL mode (x_vector_only_mode=False) requires a ref_text transcript
            if not profile["x_vector_only_mode"] and not profile["ref_text"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_ref_text",
                        "message": (
                            f"Profile '{profile['name']}' is configured for ICL mode "
                            "(x_vector_only_mode=false) but has no ref_text. "
                            "Add a transcript to meta.json or set x_vector_only_mode=true."
                        ),
                        "type": "invalid_request_error",
                    },
                )

            # Normalize cache key to canonical profile name (case-insensitive safe)
            canonical_key = profile["name"].lower()

            # Cache reference audio reads to avoid repeated disk I/O
            ref_audio_path = profile["ref_audio_path"]
            if canonical_key not in _ref_audio_cache:
                try:
                    ref_audio_np, ref_sr = sf.read(ref_audio_path)
                    if len(ref_audio_np.shape) > 1:
                        ref_audio_np = ref_audio_np.mean(axis=1)
                    ref_audio_np = ref_audio_np.astype(np.float32)
                    _ref_audio_cache[canonical_key] = (ref_audio_np, ref_sr)
                    logger.info(f"Reference audio cached for profile '{profile['name']}'")
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "audio_processing_error",
                            "message": (
                                f"Failed to load reference audio for profile "
                                f"'{profile['name']}': {exc}"
                            ),
                            "type": "invalid_request_error",
                        },
                    )
            ref_audio_np, ref_sr = _ref_audio_cache[canonical_key]

            clone_lang = (
                language if language != "Auto" else profile["language"]
            )
            logger.info(
                f"Voice library clone '{profile['name']}': "
                f"lang={clone_lang}, "
                f"x_vector_only={profile['x_vector_only_mode']}, "
                f"stream_format={request.stream_format}"
            )

            if request.stream_format and hasattr(backend, "generate_voice_clone_streaming"):
                fmt = request.response_format
                _validate_streaming_request(fmt, request.speed, request.stream_format)
                is_sse = request.stream_format == "sse"
                content_type = (
                    "text/event-stream" if is_sse else get_content_type(fmt)
                )

                clone_stream_kwargs = {
                    "ref_audio": ref_audio_np,
                    "ref_audio_sr": ref_sr,
                    "ref_text": profile["ref_text"] or None,
                    "language": clone_lang,
                    "x_vector_only_mode": profile["x_vector_only_mode"],
                }
                if _method_accepts_kwarg(
                    backend.generate_voice_clone_streaming, "cache_key"
                ):
                    clone_stream_kwargs["cache_key"] = canonical_key

                # Decide streaming chunking (independent toggle + shared sizes).
                stream_chunks = (
                    _split_into_chunks(normalized_text, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS)
                    if _STREAM_AUTOCHUNK else [Chunk(normalized_text, starts_block=True)]
                )
                if len(stream_chunks) > 1:
                    logger.info(
                        "Voice clone stream auto-chunking %d chars into %d chunks "
                        "(window=%d-%d, gap=%dms, block-gap=%dms)",
                        len(normalized_text), len(stream_chunks),
                        _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS, _CHUNK_GAP_MS, _PARAGRAPH_GAP_MS,
                    )

                def _clone_make_gen(text: str):
                    """Build one backend streaming generator for a single chunk's text."""
                    return backend.generate_voice_clone_streaming(
                        text=text, **clone_stream_kwargs,
                    )

                async def _clone_stream():
                    gen_start = time.time()
                    first_chunk_logged = False
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    wav_header_emitted = False
                    try:
                        if fmt in ("pcm", "wav"):
                            async with _generation_semaphore:
                                # Source: single-call (today) or chunked driver.
                                if len(stream_chunks) <= 1:
                                    logger.debug(
                                        "TTS clone stream single chunk (no gaps): %r",
                                        normalized_text,
                                    )
                                    source = backend.generate_voice_clone_streaming(
                                        **clone_stream_kwargs, text=normalized_text,
                                    )
                                else:
                                    source = _chunked_stream_drain(
                                        stream_chunks, _clone_make_gen, fmt, 24000,
                                    )
                                async for pcm_chunk, sr in source:
                                    if pcm_chunk is None or len(pcm_chunk) == 0:
                                        continue
                                    if not first_chunk_logged:
                                        logger.info(
                                            f"Voice clone stream TTFB: "
                                            f"{time.time()-gen_start:.3f}s"
                                        )
                                        first_chunk_logged = True
                                    total_samples += len(pcm_chunk)
                                    sample_rate = sr
                                    chunk_count += 1
                                    if is_sse:
                                        payload = pcm_bytes_from_chunk(pcm_chunk)
                                        yield _sse_event("speech.audio.delta", {
                                            "type": "speech.audio.delta",
                                            "audio": base64.b64encode(payload).decode("ascii"),
                                            "response_format": fmt,
                                        })
                                    else:
                                        chunk_bytes, wav_header_emitted = _encode_stream_chunk(
                                            pcm_chunk, fmt, sr, wav_header_emitted
                                        )
                                        if chunk_bytes:
                                            yield chunk_bytes
                                    await asyncio.sleep(0)
                        else:
                            # Compressed: drain all PCM (one chunk or many),
                            # encode ONCE, then byte-chunk. Multi-chunk must
                            # accumulate across chunks to avoid N container headers.
                            async with _generation_semaphore:
                                if len(stream_chunks) <= 1:
                                    audio_bytes, sample_rate, total_samples = await _drain_and_encode(
                                        backend.generate_voice_clone_streaming(
                                            **clone_stream_kwargs, text=normalized_text,
                                        ),
                                        fmt,
                                        24000,
                                    )
                                else:
                                    audio_np = None
                                    async for pcm_chunk, sr in _chunked_stream_drain(
                                        stream_chunks, _clone_make_gen, fmt, 24000,
                                    ):
                                        audio_np = pcm_chunk
                                        sample_rate = sr
                                    if audio_np is None:
                                        raise RuntimeError("no audio produced from any chunk")
                                    total_samples = len(audio_np)
                                    audio_bytes = await asyncio.to_thread(
                                        encode_audio, audio_np, fmt, sample_rate
                                    )
                                chunk_count = 1
                            if is_sse:
                                for slc in iter_encoded_bytes(audio_bytes):
                                    yield _sse_event("speech.audio.delta", {
                                        "type": "speech.audio.delta",
                                        "audio": base64.b64encode(slc).decode("ascii"),
                                        "response_format": fmt,
                                    })
                            else:
                                for slc in iter_encoded_bytes(audio_bytes):
                                    yield slc
                        if is_sse:
                            yield _sse_event("speech.audio.done", {
                                "type": "speech.audio.done",
                                "usage": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                },
                            })
                        gen_time = time.time() - gen_start
                        audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                        rtf = gen_time / audio_dur if audio_dur > 0 else 0
                        logger.info(
                            f"Voice clone stream done: "
                            f"total={gen_time:.2f}s audio={audio_dur:.2f}s "
                            f"RTF={rtf:.2f}x chunks={chunk_count}"
                        )
                        try:
                            note_speech_activity(client_request.app, samples=total_samples)
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.error(f"Voice clone stream error: {exc}")
                        if is_sse:
                            yield _sse_event("speech.audio.error", {
                                "type": "speech.audio.error",
                                "error": {
                                    "message": str(exc),
                                    "type": "server_error",
                                    "param": None,
                                    "code": 500,
                                },
                            })
                        raise

                return StreamingResponse(
                    _clone_stream(),
                    media_type=content_type,
                    headers=_stream_headers(fmt, request.stream_format),
                )
            else:
                # Non-streaming path — honor the requested format (including wav).
                # Chunk the normalized text (same logic as built-in voices) so
                # Markdown block boundaries get a more pronounced pause and so
                # long inputs stay within the model's effective context window.
                gen_start = time.time()
                clone_kwargs = {
                    "ref_audio": ref_audio_np,
                    "ref_audio_sr": ref_sr,
                    "ref_text": profile["ref_text"] or None,
                    "language": clone_lang,
                    "x_vector_only_mode": profile["x_vector_only_mode"],
                    "speed": request.speed,
                }
                if _method_accepts_kwarg(backend.generate_voice_clone, "cache_key"):
                    clone_kwargs["cache_key"] = canonical_key

                if _AUTOCHUNK:
                    chunks = _split_into_chunks(
                        normalized_text, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS
                    )
                else:
                    chunks = (
                        [Chunk(normalized_text, starts_block=True)]
                        if normalized_text.strip() else []
                    )

                async def _clone_synth(segment: str) -> tuple:
                    return await backend.generate_voice_clone(
                        text=segment, **clone_kwargs
                    )

                if len(chunks) <= 1:
                    seg = chunks[0].text if chunks else normalized_text
                    logger.debug(
                        "TTS clone single chunk (no gaps): %r", seg,
                    )
                    async with _generation_semaphore:
                        audio, sample_rate = await _clone_synth(seg)
                else:
                    logger.info(
                        "Voice clone auto-chunking %d chars into %d chunks "
                        "(window=%d-%d, gap=%dms, block-gap=%dms)",
                        len(normalized_text), len(chunks), _MIN_CHUNK_CHARS,
                        _MAX_CHUNK_CHARS, _CHUNK_GAP_MS, _PARAGRAPH_GAP_MS,
                    )
                    audios: List[np.ndarray] = []
                    sample_rate = 24000
                    async with _generation_semaphore:
                        for chunk in chunks:
                            a, sr = await _clone_synth(chunk.text)
                            if a is not None and len(a):
                                audios.append(np.asarray(a))
                                sample_rate = sr
                    if not audios:
                        raise RuntimeError("no audio produced from any chunk")
                    merged: List[np.ndarray] = []
                    for i, (chunk, a) in enumerate(zip(chunks, audios)):
                        if i:
                            gap_ms = (
                                _PARAGRAPH_GAP_MS if chunk.starts_block
                                else _CHUNK_GAP_MS
                            )
                            gap_len = int(sample_rate * gap_ms / 1000.0)
                            logger.debug(
                                "TTS clone chunk %d/%d: text=%r "
                                "silence_before=%dms (%s)",
                                i, len(chunks), chunk.text, gap_ms,
                                "block boundary" if chunk.starts_block
                                else "sentence split",
                            )
                            if gap_len > 0:
                                merged.append(
                                    np.zeros(gap_len, dtype=audios[0].dtype)
                                )
                        else:
                            logger.debug(
                                "TTS clone chunk %d/%d: text=%r "
                                "silence_before=0ms (first chunk)",
                                i, len(chunks), chunk.text,
                            )
                        merged.append(a)
                    audio = (
                        np.concatenate(merged) if len(merged) > 1
                        else merged[0]
                    )

                gen_time = time.time() - gen_start
                audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(
                    f"Voice clone done: gen={gen_time:.2f}s "
                    f"audio={audio_dur:.2f}s RTF={rtf:.2f}x"
                )

                fmt = request.response_format
                audio_bytes = await asyncio.to_thread(encode_audio, audio, fmt, sample_rate)
                content_type = get_content_type(fmt)

                return Response(
                    content=audio_bytes,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # ----------------------------------------------------------------
        # Streaming for built-in voices (stream_format=audio|sse)
        # ----------------------------------------------------------------
        if request.stream_format:
            backend = await get_tts_backend()
            fmt = request.response_format
            _validate_streaming_request(fmt, request.speed, request.stream_format)
            is_sse = request.stream_format == "sse"
            content_type = (
                "text/event-stream" if is_sse else get_content_type(fmt)
            )
            voice_name = get_voice_name(request.voice)

            # Decide streaming chunking (independent toggle + shared sizes).
            stream_chunks = (
                _split_into_chunks(normalized_text, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS)
                if _STREAM_AUTOCHUNK else [Chunk(normalized_text, starts_block=True)]
            )
            if len(stream_chunks) > 1:
                logger.info(
                    "TTS stream auto-chunking %d chars into %d chunks "
                    "(window=%d-%d, gap=%dms, block-gap=%dms)",
                    len(normalized_text), len(stream_chunks),
                    _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS, _CHUNK_GAP_MS, _PARAGRAPH_GAP_MS,
                )

            def _speech_make_gen(text: str):
                """Build one backend streaming generator for a single chunk's text."""
                return backend.generate_speech_streaming(
                    text=text,
                    voice=voice_name,
                    language=language,
                    instruct=request.instructions,
                    model=request.model,
                )

            if hasattr(backend, "generate_speech_streaming"):
                # Optimized backend: real incremental PCM generation.
                async def _speech_stream():
                    gen_start = time.time()
                    first_chunk_logged = False
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    wav_header_emitted = False
                    try:
                        if fmt in ("pcm", "wav"):
                            async with _generation_semaphore:
                                # Source: single-call (today) or chunked driver.
                                if len(stream_chunks) <= 1:
                                    logger.debug(
                                        "TTS stream single chunk (no gaps): %r",
                                        normalized_text,
                                    )
                                    source = backend.generate_speech_streaming(
                                        text=normalized_text,
                                        voice=voice_name,
                                        language=language,
                                        instruct=request.instructions,
                                        model=request.model,
                                    )
                                else:
                                    source = _chunked_stream_drain(
                                        stream_chunks, _speech_make_gen, fmt, 24000,
                                    )
                                async for pcm_chunk, sr in source:
                                    if pcm_chunk is None or len(pcm_chunk) == 0:
                                        continue
                                    if not first_chunk_logged:
                                        logger.info(
                                            f"TTS stream TTFB: "
                                            f"{time.time()-gen_start:.3f}s"
                                        )
                                        first_chunk_logged = True
                                    total_samples += len(pcm_chunk)
                                    sample_rate = sr
                                    chunk_count += 1
                                    if is_sse:
                                        payload = pcm_bytes_from_chunk(pcm_chunk)
                                        yield _sse_event("speech.audio.delta", {
                                            "type": "speech.audio.delta",
                                            "audio": base64.b64encode(payload).decode("ascii"),
                                            "response_format": fmt,
                                        })
                                    else:
                                        chunk_bytes, wav_header_emitted = _encode_stream_chunk(
                                            pcm_chunk, fmt, sr, wav_header_emitted
                                        )
                                        if chunk_bytes:
                                            yield chunk_bytes
                                    await asyncio.sleep(0)
                        else:
                            # Compressed: drain all PCM (one chunk or many),
                            # encode ONCE, then byte-chunk. Multi-chunk must
                            # accumulate across chunks to avoid N container headers.
                            async with _generation_semaphore:
                                if len(stream_chunks) <= 1:
                                    audio_bytes, sample_rate, total_samples = await _drain_and_encode(
                                        backend.generate_speech_streaming(
                                            text=normalized_text,
                                            voice=voice_name,
                                            language=language,
                                            instruct=request.instructions,
                                            model=request.model,
                                        ),
                                        fmt,
                                        24000,
                                    )
                                else:
                                    audio_np = None
                                    async for pcm_chunk, sr in _chunked_stream_drain(
                                        stream_chunks, _speech_make_gen, fmt, 24000,
                                    ):
                                        audio_np = pcm_chunk
                                        sample_rate = sr
                                    if audio_np is None:
                                        raise RuntimeError("no audio produced from any chunk")
                                    total_samples = len(audio_np)
                                    audio_bytes = await asyncio.to_thread(
                                        encode_audio, audio_np, fmt, sample_rate
                                    )
                                chunk_count = 1
                            if is_sse:
                                for slc in iter_encoded_bytes(audio_bytes):
                                    yield _sse_event("speech.audio.delta", {
                                        "type": "speech.audio.delta",
                                        "audio": base64.b64encode(slc).decode("ascii"),
                                        "response_format": fmt,
                                    })
                            else:
                                for slc in iter_encoded_bytes(audio_bytes):
                                    yield slc
                        if is_sse:
                            yield _sse_event("speech.audio.done", {
                                "type": "speech.audio.done",
                                "usage": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                },
                            })
                        gen_time = time.time() - gen_start
                        audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                        rtf = gen_time / audio_dur if audio_dur > 0 else 0
                        logger.info(
                            f"TTS stream done: total={gen_time:.2f}s "
                            f"audio={audio_dur:.2f}s RTF={rtf:.2f}x chunks={chunk_count}"
                        )
                        try:
                            note_speech_activity(client_request.app, samples=total_samples)
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.error(f"TTS stream error: {exc}")
                        if is_sse:
                            yield _sse_event("speech.audio.error", {
                                "type": "speech.audio.error",
                                "error": {
                                    "message": str(exc),
                                    "type": "server_error",
                                    "param": None,
                                    "code": 500,
                                },
                            })
                        raise

                return StreamingResponse(
                    _speech_stream(),
                    media_type=content_type,
                    headers=_stream_headers(fmt, request.stream_format),
                )
            else:
                # Backend without a streaming generator: drain-then-chunk so the
                # streaming envelope/Content-Type still matches the request.
                async def _fallback_stream():
                    async with _generation_semaphore:
                        audio, sample_rate = await generate_speech(
                            text=normalized_text,
                            voice=request.voice,
                            language=language,
                            instruct=request.instructions,
                            speed=1.0,
                        )
                    try:
                        note_speech_activity(client_request.app, samples=len(audio))
                    except Exception:
                        pass
                    audio_bytes = await asyncio.to_thread(
                        encode_audio, audio, fmt, sample_rate
                    )
                    if is_sse:
                        for slc in iter_encoded_bytes(audio_bytes):
                            yield _sse_event("speech.audio.delta", {
                                "type": "speech.audio.delta",
                                "audio": base64.b64encode(slc).decode("ascii"),
                                "response_format": fmt,
                            })
                        yield _sse_event("speech.audio.done", {
                            "type": "speech.audio.done",
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                            },
                        })
                    else:
                        for slc in iter_encoded_bytes(audio_bytes):
                            yield slc

                return StreamingResponse(
                    _fallback_stream(),
                    media_type=content_type,
                    headers=_stream_headers(fmt, request.stream_format),
                )

        # ----------------------------------------------------------------
        # Non-streaming: full audio in a single response
        # ----------------------------------------------------------------
        # Guard against concurrent overload
        async with _generation_semaphore:
            # Generate speech
            audio, sample_rate = await generate_speech(
                text=normalized_text,
                voice=request.voice,
                language=language,
                instruct=request.instructions,
                speed=request.speed,
            )

            # Reset the idle-shutdown timer for this in-process
            # server. Read-only endpoints like /health do not touch
            # this, so a quiet server self-terminates.
            try:
                note_speech_activity(client_request.app, samples=len(audio))
            except Exception:
                pass

        # Get content type
        content_type = get_content_type(request.response_format)

        # Encode audio to requested format (offloaded – pydub MP3 encoding is CPU-heavy)
        audio_bytes = await asyncio.to_thread(encode_audio, audio, request.response_format, sample_rate)

        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech.

    Includes built-in Qwen3-TTS speakers, OpenAI-compatible aliases, and any
    saved voice profiles from the voice library (listed with a ``clone:`` prefix).
    """
    # Default voices (always available)
    default_voices = [
        VoiceInfo(id="Vivian", name="Vivian", language="English", description="Female voice"),
        VoiceInfo(id="Ryan", name="Ryan", language="English", description="Male voice"),
        VoiceInfo(id="Sophia", name="Sophia", language="English", description="Female voice"),
        VoiceInfo(id="Isabella", name="Isabella", language="English", description="Female voice"),
        VoiceInfo(id="Evan", name="Evan", language="English", description="Male voice"),
        VoiceInfo(id="Lily", name="Lily", language="English", description="Female voice"),
    ]
    
    # OpenAI-compatible voice aliases
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Sophia)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Isabella)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Evan)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Lily)"),
    ]
    
    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]

    # Discover voice library profiles (clone: prefix voices)
    clone_voices: List[dict] = []
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if profiles_dir.exists():
        for child in sorted(profiles_dir.iterdir()):
            meta_file = child / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                ref_audio_filename = meta.get("ref_audio_filename")
                name = meta.get("name")
                if ref_audio_filename and isinstance(name, str) and name.strip():
                    clone_name = name.strip()
                    clone_id = f"clone:{clone_name}"
                    clone_voices.append(
                        VoiceInfo(
                            id=clone_id,
                            name=clone_id,
                            description=f"Voice library profile: {clone_name}",
                        ).model_dump()
                    )
                elif ref_audio_filename:
                    logger.warning(
                        "Skipping voice profile at %s due to invalid or missing 'name' in meta.json",
                        meta_file,
                    )
            except Exception:
                pass

    try:
        backend = await get_tts_backend()
        
        # Get supported speakers from the backend
        speakers = backend.get_supported_voices()
        
        # Get supported languages
        languages = backend.get_supported_languages()
        
        # Build voice list from backend
        if speakers:
            voices = []
            for speaker in speakers:
                if hasattr(backend, "is_custom_voice") and backend.is_custom_voice(speaker):
                    description = f"Custom cloned voice: {speaker}"
                else:
                    description = f"Qwen3-TTS voice: {speaker}"
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=description,
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        # OpenAI aliases map to built-in speakers; skip them on Base models
        if backend.get_model_type() != "base":
            voices += [v.model_dump() for v in openai_voices]

        return {
            "voices": voices + clone_voices,
            "languages": languages if languages else default_languages,
        }
        
    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        # Return default voices if backend is not loaded
        return {
            "voices": (
                [v.model_dump() for v in default_voices]
                + [v.model_dump() for v in openai_voices]
                + clone_voices
            ),
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """
    Get voice cloning capabilities of the current backend.

    Returns whether voice cloning is supported and what modes are available.
    Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    """
    try:
        backend = await get_tts_backend()

        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )

    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """
    Clone a voice from reference audio and generate speech.

    This endpoint requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    Set TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable when starting the server.

    Two modes are available:
    - ICL mode (x_vector_only_mode=False): Requires ref_text transcript for best quality
    - X-Vector mode (x_vector_only_mode=True): No transcript needed, good quality
    """
    try:
        backend = await get_tts_backend()

        # Check if voice cloning is supported
        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                               "Set TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable and restart the server.",
                    "type": "invalid_request_error",
                },
            )

        # Validate ICL mode requires ref_text
        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text (transcript of reference audio). "
                               "Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_audio",
                    "message": f"Failed to decode base64 audio: {e}",
                    "type": "invalid_request_error",
                },
            )

        # Load audio using soundfile
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)

            # Convert to mono if stereo
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)

            ref_audio = ref_audio.astype(np.float32)

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "audio_processing_error",
                    "message": f"Failed to process reference audio: {e}. "
                               "Ensure the audio is a valid WAV, MP3, or other supported format.",
                    "type": "invalid_request_error",
                },
            )

        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        # Generate voice clone
        async with _generation_semaphore:
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_sr,
                ref_text=request.ref_text,
                language=request.language or "Auto",
                x_vector_only_mode=request.x_vector_only_mode,
                speed=request.speed,
            )

        # Encode audio to requested format (offloaded – pydub MP3 encoding is CPU-heavy)
        audio_bytes = await asyncio.to_thread(encode_audio, audio, request.response_format, sample_rate)

        # Get content type
        content_type = get_content_type(request.response_format)

        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
