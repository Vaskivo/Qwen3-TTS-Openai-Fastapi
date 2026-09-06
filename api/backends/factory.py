# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Factory and lifecycle management for Qwen3-TTS backends."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from .base import TTSBackend

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_backend_instance: Optional[TTSBackend] = None
_initialization_lock: Optional[asyncio.Lock] = None


def _env_float(name: str, default: float, minimum: float = 0.001) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %.3f; using %.3f", name, minimum, default)
        return default
    return value


def _get_initialization_lock() -> asyncio.Lock:
    global _initialization_lock
    if _initialization_lock is None:
        _initialization_lock = asyncio.Lock()
    return _initialization_lock


def resolve_model_path(hf_id: str, models_dir: Optional[str] = None) -> str:
    """Resolve a model identifier to a **local** model directory on disk.

    This project loads models only from local paths — it does not use the
    HuggingFace cache and never downloads. Resolution order:
      1. If ``hf_id`` is an existing path on disk (absolute or relative to the
         current working directory), return it unchanged.
      2. Else if a models folder is configured (``models_dir`` argument, or the
         ``TTS_MODELS_DIR`` env var), look for
         ``<models_dir>/<repo-name-without-org>`` (i.e. strip any ``<org>/``
         prefix, so ``Qwen/Qwen3-TTS-12Hz-1.7B-Base`` maps to
         ``<models_dir>/Qwen3-TTS-12Hz-1.7B-Base``). If that directory exists,
         return it.
      3. Else raise ``FileNotFoundError`` — there is no HuggingFace fallback.

    This is pure path/string logic with no backend dependencies, so it is safe
    to share between the optimized and official backends.
    """
    if not hf_id:
        return hf_id

    # 1. Existing explicit path wins.
    candidate = Path(hf_id)
    if candidate.exists():
        return str(candidate)

    # 2. TTS_MODELS_DIR folder prefix.
    folder = models_dir if models_dir is not None else os.getenv("TTS_MODELS_DIR")
    if folder:
        repo_name = hf_id.split("/")[-1]
        local_dir = Path(folder) / repo_name
        if local_dir.exists():
            return str(local_dir)

    # 3. No HuggingFace fallback — models must be present locally.
    hint = (
        f" Set TTS_MODELS_DIR to a folder containing '{hf_id.split('/')[-1]}' "
        f"or point the config entry at an existing local path."
        if "/" in hf_id or not Path(hf_id).exists()
        else ""
    )
    raise FileNotFoundError(
        f"Model not found locally: {hf_id!r}. This build does not use the "
        f"HuggingFace cache or download models.{hint}"
    )


def get_backend() -> TTSBackend:
    """Return the process-wide backend instance, creating it lazily."""
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("TTS_BACKEND", "official").strip().lower()
    configured_model = os.getenv("TTS_MODEL_ID") or os.getenv("TTS_MODEL_NAME")
    model_name = (configured_model or _DEFAULT_MODEL).strip()

    logger.info("Creating TTS backend: %s", backend_type)

    # Imports are deliberately local so optional dependencies are not required
    # merely to import the factory module.
    if backend_type == "optimized":
        from .optimized_backend import OptimizedQwen3TTSBackend

        _backend_instance = OptimizedQwen3TTSBackend()
    elif backend_type == "official":
        from .official_qwen3_tts import OfficialQwen3TTSBackend

        _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type!r}. Supported values: "
            "optimized, official"
        )

    logger.info(
        "Using %s backend with model %s",
        _backend_instance.get_backend_name(),
        _backend_instance.get_model_id(),
    )
    return _backend_instance


async def _run_warmup_request(backend: TTSBackend, text: str) -> None:
    custom_names = backend.get_custom_voice_names()
    if backend.get_model_type() == "base" and custom_names:
        await backend.generate_speech_with_custom_voice(
            text=text,
            voice=custom_names[0],
            language="English",
        )
    elif backend.get_model_type() == "base":
        raise LookupError("Base model has no custom voice available for warmup")
    else:
        await backend.generate_speech(
            text=text,
            voice="Vivian",
            language="English",
        )


async def _warmup_backend(backend: TTSBackend) -> None:
    """Warm both regular and streaming paths with real wall-clock timeouts."""
    import time

    max_seconds = _env_float("TTS_WARMUP_MAX_SECONDS", 10.0)
    texts = [
        "Hello.",
        "Hello, this is a warmup test.",
        "Hello, this is a longer warmup test to exercise the full decode pipeline.",
    ]

    logger.info(
        "Performing backend warmup (%d requests, %.1fs timeout each)",
        len(texts),
        max_seconds,
    )
    for index, text in enumerate(texts, 1):
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                _run_warmup_request(backend, text), timeout=max_seconds
            )
        except LookupError as exc:
            logger.info("Skipping warmup: %s", exc)
            return
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Warmup request {index} exceeded {max_seconds:.1f}s"
            ) from exc
        logger.info(
            "Warmup request %d/%d completed in %.2fs",
            index,
            len(texts),
            time.monotonic() - started,
        )

    streaming_method = getattr(backend, "generate_speech_streaming", None)
    if streaming_method is None:
        return

    async def _drain_stream() -> None:
        async for _chunk, _sample_rate in streaming_method(
            text="Streaming warmup.",
            voice="Vivian",
            language="English",
        ):
            pass

    try:
        await asyncio.wait_for(_drain_stream(), timeout=max_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Streaming warmup exceeded {max_seconds:.1f}s"
        ) from exc


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """Initialize the global backend exactly once, even under concurrency."""
    global _backend_instance

    async with _get_initialization_lock():
        backend = get_backend()
        if backend.is_ready():
            return backend

        try:
            await backend.initialize()

            custom_voices_dir = os.getenv(
                "TTS_CUSTOM_VOICES",
                str(Path(__file__).resolve().parent.parent.parent / "custom_voices"),
            )
            try:
                await backend.load_custom_voices(custom_voices_dir)
            except Exception as exc:
                logger.warning("Custom voice loading failed (non-critical): %s", exc)

            if warmup and os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true":
                try:
                    await _warmup_backend(backend)
                except Exception as exc:
                    logger.error("Backend warmup failed: %s", exc)

            return backend
        except Exception:
            # A partially initialized model is unsafe to reuse. The next request
            # constructs a fresh instance rather than retrying corrupted state.
            _backend_instance = None
            raise


def reset_backend() -> None:
    """Reset process globals (primarily for tests)."""
    global _backend_instance, _initialization_lock
    _backend_instance = None
    _initialization_lock = None
