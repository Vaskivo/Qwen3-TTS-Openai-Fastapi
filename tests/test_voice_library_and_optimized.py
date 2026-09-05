# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the voice library and optimized-backend features added in the
dingausmwald fork port.

These tests do NOT require PyTorch or CUDA — they test the routing and
helper functions at the Python level, using a temporary filesystem for
voice library profiles.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_profile_dir(root: Path, profile_id: str, meta: dict, wav_content: bytes = b"RIFF") -> Path:
    """Create a voice library profile directory with meta.json and a dummy WAV."""
    profile_dir = root / "profiles" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    ref_filename = meta.get("ref_audio_filename", "reference.wav")
    (profile_dir / ref_filename).write_bytes(wav_content)
    return profile_dir


# ---------------------------------------------------------------------------
# _load_voice_profile
# ---------------------------------------------------------------------------


class TestLoadVoiceProfile:
    """Unit tests for the _load_voice_profile() helper."""

    def test_load_by_name(self, tmp_path):
        """Profile can be found by name (case-insensitive)."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "alice",
            {
                "name": "Alice",
                "profile_id": "alice",
                "ref_audio_filename": "reference.wav",
                "ref_text": "Hello world.",
                "x_vector_only_mode": False,
                "language": "English",
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("Alice")

        assert result["name"] == "Alice"
        assert result["ref_text"] == "Hello world."
        assert result["x_vector_only_mode"] is False
        assert result["language"] == "English"
        assert Path(result["ref_audio_path"]).name == "reference.wav"

    def test_load_by_name_case_insensitive(self, tmp_path):
        """Name lookup is case-insensitive."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "bob",
            {
                "name": "Bob",
                "profile_id": "bob",
                "ref_audio_filename": "ref.wav",
                "x_vector_only_mode": True,
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("bob")  # lower-case lookup

        assert result["name"] == "Bob"

    def test_load_by_profile_id(self, tmp_path):
        """Profile can be found by exact profile_id match."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "carol_v2",
            {
                "name": "Carol",
                "profile_id": "carol_v2",
                "ref_audio_filename": "ref.wav",
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("carol_v2")

        assert result["name"] == "Carol"

    def test_not_found_raises_value_error(self, tmp_path):
        """Missing profile raises ValueError."""
        from api.routers import openai_compatible as oc

        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            with pytest.raises(ValueError, match="not found"):
                oc._load_voice_profile("nobody")

    def test_missing_library_raises_value_error(self, tmp_path):
        """Missing voice library directory raises ValueError."""
        from api.routers import openai_compatible as oc

        missing_dir = tmp_path / "does_not_exist"
        with patch.object(oc, "VOICE_LIBRARY_DIR", missing_dir):
            with pytest.raises(ValueError, match="Voice library not found"):
                oc._load_voice_profile("anything")

    def test_missing_ref_audio_raises_value_error(self, tmp_path):
        """Profile with missing ref_audio_filename raises ValueError."""
        from api.routers import openai_compatible as oc

        profile_dir = tmp_path / "profiles" / "empty"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "meta.json").write_text(
            json.dumps({"name": "Empty", "profile_id": "empty"}),
            encoding="utf-8",
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            with pytest.raises(ValueError, match="no reference audio"):
                oc._load_voice_profile("Empty")

    def test_defaults_for_optional_fields(self, tmp_path):
        """Optional meta.json fields fall back to sensible defaults."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "minimal",
            {
                "name": "Minimal",
                "ref_audio_filename": "ref.wav",
                # no ref_text, x_vector_only_mode, language
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("minimal")

        assert result["ref_text"] == ""
        assert result["x_vector_only_mode"] is False
        assert result["language"] == "Auto"


# ---------------------------------------------------------------------------
# list_voices — voice library profiles appear in response
# ---------------------------------------------------------------------------


class TestListVoicesVoiceLibrary:
    """The /v1/voices endpoint must include voice library profiles."""

    @pytest.mark.asyncio
    async def test_clone_voices_included_in_listing(self, tmp_path):
        """Saved profiles appear as 'clone:Name' entries in the voices list."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "dave",
            {
                "name": "Dave",
                "ref_audio_filename": "reference.wav",
            },
        )

        # Patch VOICE_LIBRARY_DIR and a minimal backend
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.get_supported_voices.return_value = []
        mock_backend.get_supported_languages.return_value = ["English"]
        mock_backend.get_model_type.return_value = "customvoice"
        mock_backend.is_custom_voice.return_value = False

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path), \
             patch("api.routers.openai_compatible.get_tts_backend", return_value=mock_backend):
            result = await oc.list_voices()

        voice_ids = [v["id"] for v in result["voices"]]
        assert "clone:Dave" in voice_ids


# ---------------------------------------------------------------------------
# Factory — optimized backend selection
# ---------------------------------------------------------------------------


class TestOptimizedBackendSelection:
    """Factory must return OptimizedQwen3TTSBackend when TTS_BACKEND=optimized."""

    def teardown_method(self):
        try:
            from api.backends.factory import reset_backend
            reset_backend()
        except Exception:
            pass

    def test_optimized_backend_selected(self, tmp_path, monkeypatch):
        """TTS_BACKEND=optimized returns OptimizedQwen3TTSBackend."""
        pytest.importorskip("torch")
        monkeypatch.setenv("TTS_BACKEND", "optimized")
        _write_config(tmp_path, monkeypatch, _base_config())

        from api.backends.factory import get_backend, reset_backend
        reset_backend()

        from api.backends.optimized_backend import OptimizedQwen3TTSBackend
        backend = get_backend()
        assert isinstance(backend, OptimizedQwen3TTSBackend)
        assert backend.get_backend_name() == "optimized"

    def test_optimized_backend_implements_interface(self, tmp_path, monkeypatch):
        """OptimizedQwen3TTSBackend implements the TTSBackend interface."""
        pytest.importorskip("torch")
        _write_config(tmp_path, monkeypatch, _base_config())
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend
        from api.backends.base import TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, "initialize")
        assert hasattr(backend, "generate_speech")
        assert hasattr(backend, "generate_speech_streaming")
        assert hasattr(backend, "generate_voice_clone")
        assert hasattr(backend, "generate_voice_clone_streaming")
        assert hasattr(backend, "supports_voice_cloning")
        assert hasattr(backend, "get_backend_name")
        assert hasattr(backend, "get_model_id")
        assert hasattr(backend, "get_supported_voices")
        assert hasattr(backend, "get_supported_languages")
        assert hasattr(backend, "is_ready")
        assert hasattr(backend, "get_device_info")

    def test_optimized_backend_supports_voice_cloning(self, tmp_path, monkeypatch):
        """OptimizedQwen3TTSBackend reports voice cloning as supported."""
        pytest.importorskip("torch")
        _write_config(tmp_path, monkeypatch, _base_config())
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend.supports_voice_cloning() is True

    def test_optimized_backend_not_ready_initially(self, tmp_path, monkeypatch):
        """OptimizedQwen3TTSBackend is not ready before initialize() is called."""
        pytest.importorskip("torch")
        _write_config(tmp_path, monkeypatch, _base_config())
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend.is_ready() is False

    def test_optimized_backend_loads_config_yaml(self, tmp_path, monkeypatch):
        """OptimizedQwen3TTSBackend reads config.yaml when it exists."""
        pytest.importorskip("torch")
        import yaml

        config = {
            "default_model": "my-model",
            "voice_clone_model": "my-base",
            "load_both_models": False,
            "models": {
                "my-model": {"hf_id": "test/model", "type": "customvoice"},
                "my-base": {"hf_id": "test/base", "type": "base"},
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        monkeypatch.setenv("TTS_CONFIG", str(config_file))

        # Re-import to get a fresh instance
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend._default_model_key() == "my-model"
        assert backend._voice_clone_model_key() == "my-base"
        assert backend._load_both_models() is False
        assert backend.get_loaded_model_keys() == []


# ---------------------------------------------------------------------------
# Config validation + per-purpose model selection + dual-resident behaviour
# ---------------------------------------------------------------------------


def _write_config(tmp_path, monkeypatch, config: dict) -> None:
    """Write a config dict to a temp config.yaml and point TTS_CONFIG at it."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setenv("TTS_CONFIG", str(config_file))


def _base_config() -> dict:
    return {
        "default_model": "cv",
        "voice_clone_model": "base",
        "load_both_models": False,
        "models": {
            "cv": {"hf_id": "test/cv", "type": "customvoice"},
            "base": {"hf_id": "test/base", "type": "base"},
        },
    }


class TestPerPurposeModelConfig:
    """Validates required config keys, the voice_clone_model resolver, and the
    keep-vs-swap behaviour of _ensure_model_loaded under load_both_models."""

    def test_voice_clone_model_key_returns_configured_value(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        _write_config(tmp_path, monkeypatch, _base_config())
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend._voice_clone_model_key() == "base"
        assert backend._default_model_key() == "cv"

    def test_missing_voice_clone_model_raises(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        del cfg["voice_clone_model"]
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="voice_clone_model"):
            OptimizedQwen3TTSBackend()

    def test_voice_clone_model_must_be_base_type(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        cfg["voice_clone_model"] = "cv"  # customvoice, not base
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="base"):
            OptimizedQwen3TTSBackend()

    def test_missing_load_both_models_raises(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        del cfg["load_both_models"]
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="load_both_models"):
            OptimizedQwen3TTSBackend()

    def test_load_both_models_non_bool_raises(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        cfg["load_both_models"] = "yes"
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="boolean"):
            OptimizedQwen3TTSBackend()

    def test_missing_default_model_raises(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        del cfg["default_model"]
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="default_model"):
            OptimizedQwen3TTSBackend()

    def test_default_model_must_be_customvoice_type(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        cfg = _base_config()
        cfg["default_model"] = "base"  # base, not customvoice
        _write_config(tmp_path, monkeypatch, cfg)
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        with pytest.raises(ValueError, match="customvoice"):
            OptimizedQwen3TTSBackend()


class TestEnsureModelLoadedResidency:
    """_ensure_model_loaded keeps vs. swaps resident models based on
    load_both_models, using a mocked Qwen3TTSModel.from_pretrained."""

    def _make_backend(self, tmp_path, monkeypatch, load_both: bool):
        pytest.importorskip("torch")
        cfg = _base_config()
        cfg["load_both_models"] = load_both
        # Disable torch.compile path so _apply_optimizations is skipped.
        cfg.setdefault("optimization", {})["use_compile"] = False
        _write_config(tmp_path, monkeypatch, cfg)

        from api.backends import optimized_backend

        def _fake_from_pretrained(hf_id, **kwargs):
            m = MagicMock()
            m._hf_id = hf_id
            return m

        monkeypatch.setattr(
            optimized_backend, "__import__", __import__, raising=False
        )
        # Patch the from_pretrained via patching the module-level import site.
        import qwen_tts  # noqa: F401  (ensure importable; tests skip if absent)
        monkeypatch.setattr(
            qwen_tts.Qwen3TTSModel, "from_pretrained",
            staticmethod(_fake_from_pretrained),
        )
        backend = optimized_backend.OptimizedQwen3TTSBackend()
        return backend

    def test_swap_unloads_previous_model_when_not_both(self, tmp_path, monkeypatch):
        backend = self._make_backend(tmp_path, monkeypatch, load_both=False)
        import asyncio

        asyncio.run(backend._ensure_model_loaded("cv"))
        assert backend.get_loaded_model_keys() == ["cv"]
        assert backend.current_model_key == "cv"

        asyncio.run(backend._ensure_model_loaded("base"))
        assert backend.get_loaded_model_keys() == ["base"]
        assert backend.current_model_key == "base"
        assert backend.is_ready()

    def test_both_keeps_resident_models(self, tmp_path, monkeypatch):
        backend = self._make_backend(tmp_path, monkeypatch, load_both=True)
        import asyncio

        asyncio.run(backend._ensure_model_loaded("cv"))
        asyncio.run(backend._ensure_model_loaded("base"))
        assert sorted(backend.get_loaded_model_keys()) == ["base", "cv"]

        # Switching back to cv must NOT unload base.
        asyncio.run(backend._ensure_model_loaded("cv"))
        assert sorted(backend.get_loaded_model_keys()) == ["base", "cv"]
        assert backend.current_model_key == "cv"

    def test_already_resident_just_moves_active_pointer(self, tmp_path, monkeypatch):
        backend = self._make_backend(tmp_path, monkeypatch, load_both=True)
        import asyncio

        asyncio.run(backend._ensure_model_loaded("cv"))
        first_model = backend.model
        asyncio.run(backend._ensure_model_loaded("cv"))
        assert backend.model is first_model
        assert backend.current_model_key == "cv"


# ---------------------------------------------------------------------------
# stream_generate_custom_voice — method signature check
# ---------------------------------------------------------------------------


class TestStreamGenerateCustomVoice:
    """stream_generate_custom_voice must exist on Qwen3TTSModel."""

    def test_method_exists(self):
        """Qwen3TTSModel has a stream_generate_custom_voice method."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        assert hasattr(Qwen3TTSModel, "stream_generate_custom_voice")
        assert callable(getattr(Qwen3TTSModel, "stream_generate_custom_voice"))

    def test_method_raises_on_wrong_model_type(self):
        """Calling stream_generate_custom_voice on a non-customvoice model raises ValueError."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # Build a minimal mock model with wrong type
        mock_model = MagicMock()
        mock_model.tts_model_type = "base"  # NOT custom_voice
        mock_processor = MagicMock()

        wrapper = Qwen3TTSModel(model=mock_model, processor=mock_processor)

        with pytest.raises(ValueError, match="custom_voice"):
            list(wrapper.stream_generate_custom_voice(text="hello", speaker="Vivian"))

    def test_method_raises_on_batch_input(self):
        """Passing a list as text raises ValueError (no batching supported)."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        mock_model = MagicMock()
        mock_model.tts_model_type = "custom_voice"
        mock_processor = MagicMock()

        wrapper = Qwen3TTSModel(model=mock_model, processor=mock_processor)

        with pytest.raises(ValueError, match="single text"):
            list(wrapper.stream_generate_custom_voice(text=["a", "b"], speaker="Vivian"))
