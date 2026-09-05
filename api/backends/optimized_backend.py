# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Optimized Qwen3-TTS backend with dynamic model switching, torch.compile,
CUDA graph captures, voice prompt caching, and real-time streaming.

This backend reads its model roster and optimization knobs from a YAML config
file (default: ~/qwen3-tts/config.yaml, overridable via TTS_CONFIG env var).
It routes each request to the model configured for that purpose:
`default_model` for non-clone requests and `voice_clone_model` (a base model)
for voice-cloning requests (clone: prefix, voice library, /audio/voice-clone).

`load_both_models` controls whether both models stay resident in VRAM at once
(true: no swapping after the initial load; false: unload + reload on each
switch — lower VRAM, higher per-switch latency).
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any

import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Location of the YAML config file (overridable via TTS_CONFIG env var)
_DEFAULT_CONFIG_PATH = Path.home() / "qwen3-tts" / "config.yaml"


def _load_config() -> dict:
    """Load the YAML configuration file, returning an empty dict on failure."""
    config_path = Path(os.environ.get("TTS_CONFIG", str(_DEFAULT_CONFIG_PATH)))
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning(f"Could not load config {config_path}: {exc}")
    return {}


class OptimizedQwen3TTSBackend(TTSBackend):
    """
    Optimized backend with dynamic model switching and real-time streaming.

    Key capabilities:
    - torch.compile + CUDA graph captures (configured via config.yaml)
    - Per-purpose model selection: `default_model` for non-clone requests,
      `voice_clone_model` (base) for voice-cloning requests
    - Optional dual-resident mode (`load_both_models`) keeps both models in
      VRAM; otherwise swaps (unload + load) on demand
    - Voice prompt caching (~0.7 s saved per repeated voice-clone request)
    - Real-time PCM streaming for both CustomVoice and Base (voice-clone) models

    Config file (~/qwen3-tts/config.yaml) example::

        default_model: 0.6B-CustomVoice      # used for non-clone requests
        voice_clone_model: 0.6B-Base          # used for clone: requests (must be base)
        load_both_models: false                # keep both resident? false = swap on demand
        models:
          0.6B-CustomVoice:
            hf_id: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
            type: customvoice          # "customvoice" | "base"
          0.6B-Base:
            hf_id: Qwen/Qwen3-TTS-12Hz-0.6B-Base
            type: base
        optimization:
          attention: flash_attention_2
          use_compile: true
          compile_mode: max-autotune   # "default" | "reduce-overhead" | "max-autotune"
          use_cuda_graphs: true
          use_fast_codebook: true
          compile_codebook_predictor: true
          streaming:
            decode_window_frames: 80   # AMD users: try 72; NVIDIA: 80 is fine
            emit_every_frames: 6       # lower = lower TTFB; higher = better RTF
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = _load_config()
        self._log_config_source()
        self._validate_config()
        # Registry of resident model instances keyed by model_key.  Under
        # `load_both_models: true` more than one entry can live here at once;
        # under `false` at most one is ever resident.
        self._models: Dict[str, Any] = {}
        # Active-model pointers (kept for backward-compat with introspection /
        # get_model_id / get_model_type and code that reads `self.model`).
        self.model: Optional[Any] = None
        self.current_model_key: Optional[str] = None
        self._voice_prompt_cache: Dict[str, Any] = {}  # cache_key -> VoiceClonePromptItem list
        self._ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_config_source(self) -> None:
        """Log which config file was read and the key model-selection values, so a
        misread/stale config is obvious in the startup logs."""
        path = os.environ.get(
            "TTS_CONFIG", str(_DEFAULT_CONFIG_PATH)
        )
        logger.info(
            "Optimized backend config: file=%s default_model=%r "
            "voice_clone_model=%r load_both_models=%r",
            path,
            self.config.get("default_model"),
            self.config.get("voice_clone_model"),
            self.config.get("load_both_models"),
        )

    def _validate_config(self) -> None:
        """Validate the required config keys; raise ValueError with a helpful
        message if any are missing or wrong."""
        models = self.config.get("models", {})
        if not models:
            raise ValueError(
                "config.yaml: `models` section is missing or empty."
            )

        default_key = self.config.get("default_model")
        if not default_key:
            raise ValueError(
                "config.yaml: required key `default_model` is missing "
                "(must name a `customvoice`-type key in `models`)."
            )
        if default_key not in models:
            raise ValueError(
                f"config.yaml: `default_model: {default_key!r}` is not a key in "
                f"`models`. Available: {list(models.keys())}."
            )
        # Non-clone generation uses generate_custom_voice, which only exists on
        # CustomVoice models; reject a base model here so the server fails fast
        # with a clear message instead of crashing on the first request.
        if not isinstance(models[default_key], dict) \
                or models[default_key].get("type") != "customvoice":
            cv_keys = [
                k for k, v in models.items()
                if isinstance(v, dict) and v.get("type") == "customvoice"
            ]
            raise ValueError(
                f"config.yaml: `default_model: {default_key!r}` is not a "
                f"`customvoice`-type model (non-clone generation requires a "
                f"customvoice model). CustomVoice models available: {cv_keys}."
            )

        clone_key = self.config.get("voice_clone_model")
        if not clone_key:
            raise ValueError(
                "config.yaml: required key `voice_clone_model` is missing "
                "(must name a `base`-type key in `models`)."
            )
        if clone_key not in models:
            raise ValueError(
                f"config.yaml: `voice_clone_model: {clone_key!r}` is not a key in "
                f"`models`. Available: {list(models.keys())}."
            )
        if not isinstance(models[clone_key], dict) \
                or models[clone_key].get("type") != "base":
            base_keys = [
                k for k, v in models.items()
                if isinstance(v, dict) and v.get("type") == "base"
            ]
            raise ValueError(
                f"config.yaml: `voice_clone_model: {clone_key!r}` is not a "
                f"`base`-type model (voice cloning requires a base model). "
                f"Base models available: {base_keys}."
            )

        load_both = self.config.get("load_both_models")
        if load_both is None:
            raise ValueError(
                "config.yaml: required key `load_both_models` is missing "
                "(boolean)."
            )
        if not isinstance(load_both, bool):
            raise ValueError(
                f"config.yaml: `load_both_models` must be a boolean, "
                f"got {type(load_both).__name__}."
            )

    def _default_model_key(self) -> str:
        return self.config["default_model"]

    def _voice_clone_model_key(self) -> str:
        """Return the Base model key used for voice cloning."""
        return self.config["voice_clone_model"]

    def _load_both_models(self) -> bool:
        return bool(self.config.get("load_both_models", False))

    def _model_info(self, model_key: str) -> dict:
        return self.config.get("models", {}).get(model_key, {})

    async def _ensure_model_loaded(self, model_key: str) -> None:
        """Ensure *model_key* is resident and make it the active model.

        Under `load_both_models: true` already-loaded models are kept resident;
        only the active pointer moves.  Under `false` (the default) any other
        resident model is unloaded first, preserving today's single-resident
        swap behaviour.
        """
        import torch

        # Already resident: just make it active.
        if model_key in self._models:
            self.model = self._models[model_key]
            self.current_model_key = model_key
            self._ready = True
            return

        model_info = self._model_info(model_key)
        if not model_info:
            raise ValueError(
                f"Unknown model key: '{model_key}'. "
                f"Available: {list(self.config.get('models', {}).keys())}"
            )

        hf_id = model_info["hf_id"]

        # Unload any resident models when not keeping both.  Voice-prompt
        # cache entries are tied to a specific base-model instance, so clear
        # them on unload (today's behaviour).
        if not self._load_both_models() and self._models:
            for old_key in list(self._models.keys()):
                logger.info(f"Unloading {old_key!r}…")
                if self._voice_prompt_cache:
                    logger.info(
                        f"Clearing voice prompt cache "
                        f"({len(self._voice_prompt_cache)} entries)"
                    )
                    self._voice_prompt_cache.clear()
                del self._models[old_key]
                torch.cuda.empty_cache()

        logger.info(f"Loading {model_key!r} ({hf_id})…")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        from qwen_tts import Qwen3TTSModel
        torch.set_float32_matmul_precision("high")

        opt = self.config.get("optimization", {})
        attn_impl = opt.get("attention", "flash_attention_2")

        try:
            loaded = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            logger.info(f"Model loaded with {attn_impl} attention")
        except Exception as exc:
            logger.warning(f"Failed with {attn_impl}: {exc}; retrying with sdpa")
            loaded = Qwen3TTSModel.from_pretrained(
                hf_id,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation="sdpa",
            )
            logger.info("Model loaded with sdpa attention")

        self._models[model_key] = loaded
        # Point the active model at the newly loaded instance while
        # _apply_optimizations runs (it reads self.model).
        self.model = loaded
        self.current_model_key = model_key

        # torch.compile + CUDA graph optimizations
        if opt.get("use_compile", True) and self.device != "cpu":
            await self._apply_optimizations(model_key, model_info, opt)

        self._ready = True
        logger.info(f"Model {model_key!r} ready on {self.device}")

    async def _apply_optimizations(
        self, model_key: str, model_info: dict, opt: dict
    ) -> None:
        """Enable torch.compile and run mandatory warmup passes."""
        streaming_opts = opt.get("streaming", {})
        decode_window = streaming_opts.get("decode_window_frames", 80)
        emit_every = streaming_opts.get("emit_every_frames", 6)

        try:
            self.model.enable_streaming_optimizations(
                decode_window_frames=decode_window,
                use_compile=True,
                use_cuda_graphs=opt.get("use_cuda_graphs", False),
                compile_mode=opt.get("compile_mode", "max-autotune"),
                use_fast_codebook=opt.get("use_fast_codebook", True),
                compile_codebook_predictor=opt.get("compile_codebook_predictor", True),
            )
            logger.info(
                f"torch.compile enabled: mode={opt.get('compile_mode', 'max-autotune')}, "
                f"cuda_graphs={opt.get('use_cuda_graphs', False)}"
            )
        except Exception as exc:
            logger.warning(f"Could not enable streaming optimizations: {exc}")
            return

        # Warmup — triggers actual kernel compilation (torch.compile is lazy)
        model_type = model_info.get("type", "customvoice")
        import numpy as _np

        dummy_audio = _np.sin(
            2 * _np.pi * 440 * _np.arange(24000) / 24000
        ).astype(_np.float32)

        try:
            if model_type == "base":
                await self._warmup_base_model(dummy_audio, emit_every, decode_window)
            else:
                await self._warmup_customvoice_model(emit_every, decode_window)
        except Exception as exc:
            logger.warning(f"Warmup failed (non-critical): {exc}")

    async def _warmup_base_model(
        self,
        dummy_audio: "np.ndarray",
        emit_every: int,
        decode_window: int,
    ) -> None:
        """Three-pass warmup for the Base model (x-vector + ICL + stabilisation)."""
        logger.info("Warmup 1/3: Base — x_vector_only non-streaming…")
        self.model.generate_voice_clone(
            text="Warmup sentence.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
        )
        logger.info("Warmup 1/3: Base — x_vector_only streaming…")
        for _ in self.model.stream_generate_voice_clone(
            text="Streaming warmup for voice clone.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 2/3: Base — ICL mode streaming…")
        for _ in self.model.stream_generate_voice_clone(
            text="Second warmup to compile ICL reference-code path.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            ref_text="Warmup reference text.",
            x_vector_only_mode=False,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 3/3: Base — GPU power stabilisation…")
        for _ in self.model.stream_generate_voice_clone(
            text="Third pass to stabilise GPU power state.",
            language="English",
            ref_audio=(dummy_audio, 24000),
            x_vector_only_mode=True,
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass
        logger.info("Warmup complete (base)")

    async def _warmup_customvoice_model(
        self, emit_every: int, decode_window: int
    ) -> None:
        """Three-pass warmup for the CustomVoice model."""
        logger.info("Warmup 1/3: CustomVoice — non-streaming…")
        self.model.generate_custom_voice(
            text="This is a warmup sentence to trigger torch.compile kernel compilation.",
            language="English",
            speaker="Eric",
        )
        logger.info("Warmup 1/3: CustomVoice — streaming…")
        for _ in self.model.stream_generate_custom_voice(
            text="Streaming warmup sentence for compiling the streaming code path.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 2/3: CustomVoice — medium text…")
        for _ in self.model.stream_generate_custom_voice(
            text="A medium-length sentence to warm up different tensor shapes.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass

        logger.info("Warmup 3/3: CustomVoice — short text…")
        for _ in self.model.stream_generate_custom_voice(
            text="Short test.",
            speaker="Eric",
            language="English",
            emit_every_frames=emit_every,
            decode_window_frames=decode_window,
        ):
            pass
        logger.info("Warmup complete (customvoice)")

    # ------------------------------------------------------------------
    # TTSBackend interface — initialisation
    # ------------------------------------------------------------------

    async def initialize(
        self, model_key: Optional[str] = None, warmup: bool = False
    ) -> None:
        """Load the startup model(s).

        When ``load_both_models`` is true, both ``default_model`` and
        ``voice_clone_model`` are made resident (deduplicated if they are the
        same key).  Otherwise only a single model is loaded — *model_key* if
        given, else ``default_model`` — and the other is loaded lazily on first
        use, swapping out the resident one (today's behaviour).
        """
        if self._load_both_models():
            default_key = self._default_model_key()
            clone_key = self._voice_clone_model_key()
            await self._ensure_model_loaded(default_key)
            if clone_key != default_key:
                await self._ensure_model_loaded(clone_key)
            logger.info(
                f"Loaded models: {self.get_loaded_model_keys()} "
                f"(active={self.current_model_key!r})"
            )
            return

        if model_key is None:
            model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

    async def switch_model(self, model_key: str) -> None:
        """Hot-swap the active model to *model_key*.

        Under ``load_both_models`` this just moves the active pointer if the
        model is already resident; otherwise it loads it (and, when not keeping
        both, unloads the current resident model first).
        """
        await self._ensure_model_loaded(model_key)

    # ------------------------------------------------------------------
    # TTSBackend interface — generation
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> Tuple[np.ndarray, int]:
        """Non-streaming CustomVoice generation."""
        model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=voice,
            instruct=instruct,
        )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            except ImportError:
                pass
        return audio, sr

    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Real token-by-token streaming via stream_generate_custom_voice.

        Yields (pcm_chunk, sample_rate) tuples as the model generates audio.
        """
        model_key = self._default_model_key()
        await self._ensure_model_loaded(model_key)

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 6)

        for chunk, sr in self.model.stream_generate_custom_voice(
            text=text,
            speaker=voice,
            language=language,
            instruct=instruct,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        ):
            yield chunk, sr

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Non-streaming voice cloning (uses the configured voice-clone model)."""
        await self._ensure_model_loaded(self._voice_clone_model_key())

        t0 = time.time()

        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(
                f"Voice clone (cached prompt '{cache_key}'): "
                f"generate={time.time()-t0:.3f}s"
            )
        else:
            t_prompt_start = time.time()
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            t_prompt = time.time() - t_prompt_start
            if cache_key:
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(
                    f"Voice prompt cached for '{cache_key}' "
                    f"(build={t_prompt:.3f}s)"
                )
            t_gen_start = time.time()
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )
            logger.info(
                f"Voice clone: prompt={t_prompt:.3f}s "
                f"generate={time.time()-t_gen_start:.3f}s"
            )

        audio = wavs[0]
        if speed != 1.0:
            try:
                import librosa
                audio = librosa.effects.time_stretch(
                    audio.astype(np.float32), rate=speed
                )
            except ImportError:
                pass
        return audio, sr

    async def generate_voice_clone_streaming(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        cache_key: Optional[str] = None,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """
        Real token-by-token streaming voice cloning (uses the configured
        voice-clone model).

        Yields (pcm_chunk, sample_rate) tuples as the model generates audio.
        """
        await self._ensure_model_loaded(self._voice_clone_model_key())

        streaming_opts = self.config.get("optimization", {}).get("streaming", {})
        decode_window_frames = streaming_opts.get("decode_window_frames", 80)
        emit_every_frames = streaming_opts.get("emit_every_frames", 6)

        # Build or retrieve cached voice clone prompt
        t0 = time.time()
        if cache_key and cache_key in self._voice_prompt_cache:
            prompt_items = self._voice_prompt_cache[cache_key]
        else:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            t_prompt = time.time() - t0
            if cache_key:
                self._voice_prompt_cache[cache_key] = prompt_items
                logger.info(
                    f"Voice prompt cached for '{cache_key}' "
                    f"(build={t_prompt:.3f}s)"
                )
            else:
                logger.info(f"Voice prompt built (no cache): {time.time()-t0:.3f}s")

        for chunk, sr in self.model.stream_generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt_items,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        ):
            yield chunk, sr

    # ------------------------------------------------------------------
    # TTSBackend interface — metadata / introspection
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        return "optimized"

    def get_model_id(self) -> str:
        if self.current_model_key:
            info = self._model_info(self.current_model_key)
            return info.get("hf_id", "unknown")
        return "not-loaded"

    def get_supported_voices(self) -> List[str]:
        """Return voice names listed in config.yaml (voices section)."""
        return [v["name"] for v in self.config.get("voices", [])]

    def get_supported_languages(self) -> List[str]:
        return [
            "English", "Chinese", "Japanese", "Korean", "German",
            "French", "Spanish", "Russian", "Portuguese", "Italian",
        ]

    def is_ready(self) -> bool:
        return self._ready

    def supports_voice_cloning(self) -> bool:
        return True

    def get_model_type(self) -> str:
        if not self.current_model_key:
            return "unknown"
        return self._model_info(self.current_model_key).get("type", "unknown")

    def get_device_info(self) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            return {"device": "unknown", "gpu_available": False}

        info: Dict[str, Any] = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
        }
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total"] = f"{props.total_memory / 1024**3:.1f} GB"
        return info

    def get_available_models(self) -> List[str]:
        return list(self.config.get("models", {}).keys())

    def get_current_model_key(self) -> Optional[str]:
        return self.current_model_key

    def get_loaded_model_keys(self) -> List[str]:
        """Return the keys of all currently resident model instances."""
        return list(self._models.keys())

    def get_config(self) -> dict:
        return self.config
