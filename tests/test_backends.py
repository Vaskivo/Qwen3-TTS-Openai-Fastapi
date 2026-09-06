# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for backend selection and initialization.
"""

import pytest

from api.backends.factory import get_backend, reset_backend
from api.backends.base import TTSBackend
from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend


class TestBackendSelection:
    """Test backend selection via environment variables."""

    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()

    def test_default_backend_is_official(self, monkeypatch):
        """Test that official backend is selected by default."""
        # Ensure TTS_BACKEND is not set
        monkeypatch.delenv("TTS_BACKEND", raising=False)

        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"

    def test_official_backend_via_env(self, monkeypatch):
        """Test selecting official backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")

        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"

    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend name raises ValueError."""
        monkeypatch.setenv("TTS_BACKEND", "invalid_backend")

        with pytest.raises(ValueError, match="Unknown TTS_BACKEND"):
            get_backend()

    def test_invalid_backend_error_lists_valid_options(self, monkeypatch):
        """The 'unknown backend' error must list only the valid options."""
        monkeypatch.setenv("TTS_BACKEND", "definitely_not_a_real_backend")

        with pytest.raises(ValueError) as exc_info:
            get_backend()
        message = str(exc_info.value)
        assert "official" in message
        assert "optimized" in message
        # Removed backends must no longer appear as valid options.
        assert "vllm" not in message
        assert "pytorch" not in message
        assert "openvino" not in message
        assert "mlx" not in message

    def test_custom_model_name_via_env(self, monkeypatch):
        """Test overriding model name via the legacy TTS_MODEL_NAME env var."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        # TTS_MODEL_ID takes precedence, so it must be unset for the legacy
        # TTS_MODEL_NAME fallback to take effect.
        monkeypatch.delenv("TTS_MODEL_ID", raising=False)
        monkeypatch.setenv("TTS_MODEL_NAME", "custom/model")

        backend = get_backend()
        assert backend.get_model_id() == "custom/model"

    def test_model_id_takes_precedence_over_model_name(self, monkeypatch):
        """TTS_MODEL_ID wins when both TTS_MODEL_ID and TTS_MODEL_NAME are set."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        monkeypatch.setenv("TTS_MODEL_ID", "preferred/model")
        monkeypatch.setenv("TTS_MODEL_NAME", "legacy/model")

        backend = get_backend()
        assert backend.get_model_id() == "preferred/model"

    def test_backend_singleton(self, monkeypatch):
        """Test that get_backend returns the same instance."""
        monkeypatch.setenv("TTS_BACKEND", "official")

        backend1 = get_backend()
        backend2 = get_backend()

        assert backend1 is backend2


class TestBackendInterface:
    """Test that backends implement the required interface."""

    def test_official_backend_implements_interface(self):
        """Test official backend implements TTSBackend interface."""
        backend = OfficialQwen3TTSBackend()

        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')

    def test_backend_names_are_correct(self):
        """Test that backends return correct names."""
        official = OfficialQwen3TTSBackend()

        assert official.get_backend_name() == "official"

    def test_backends_return_voices(self):
        """Test that backends return voice lists."""
        official = OfficialQwen3TTSBackend()

        assert isinstance(official.get_supported_voices(), list)
        assert len(official.get_supported_voices()) > 0

    def test_backends_return_languages(self):
        """Test that backends return language lists."""
        official = OfficialQwen3TTSBackend()

        assert isinstance(official.get_supported_languages(), list)
        assert len(official.get_supported_languages()) > 0

    def test_backends_initially_not_ready(self):
        """Test that backends are not ready before initialization."""
        official = OfficialQwen3TTSBackend()

        assert not official.is_ready()

    def test_backends_return_device_info(self):
        """Test that backends return device info dict."""
        official = OfficialQwen3TTSBackend()

        info = official.get_device_info()

        # Check required keys
        assert "device" in info
        assert "gpu_available" in info


class TestVoiceCloningInterface:
    """Tests for voice cloning interface across all backends."""

    def test_official_backend_has_voice_cloning_methods(self):
        """Test that official backend has voice cloning methods."""
        backend = OfficialQwen3TTSBackend()

        assert hasattr(backend, 'supports_voice_cloning')
        assert hasattr(backend, 'get_model_type')
        assert hasattr(backend, 'generate_voice_clone')

    def test_customvoice_model_does_not_support_cloning(self):
        """Test that CustomVoice models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

        assert not official.supports_voice_cloning()
        assert official.get_model_type() == "customvoice"

    def test_base_model_supports_cloning(self):
        """Test that Base models support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")

        assert official.supports_voice_cloning()
        assert official.get_model_type() == "base"

    def test_voicedesign_model_does_not_support_cloning(self):
        """Test that VoiceDesign models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

        assert not official.supports_voice_cloning()

    def test_model_type_defaults_to_customvoice(self):
        """Test that default model type is customvoice."""
        official = OfficialQwen3TTSBackend()

        assert official.get_model_type() == "customvoice"
