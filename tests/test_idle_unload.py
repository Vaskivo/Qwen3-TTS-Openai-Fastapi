# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Tests for the idle model unload (VRAM release) feature.

Covers:
- factory ``unload_backend()`` (no-op when not ready, unloads when ready)
- both backends' ``unload()`` implementations (official + optimized)
- the idle-unload watchdog in ``api.main`` (fires after idle, disabled at 0,
  skips when nothing is loaded, resets on successful speech, takes the
  generation slot before unloading)
- the manual ``POST /v1/audio/unload`` endpoint (unloaded / already_unloaded /
  busy)
- ``/health`` reporting ``"unloaded"`` vs ``"initializing"`` / ``"healthy"``
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.backends import factory as backend_factory
from api.backends.base import TTSBackend
from api.main import app
from api.routers import openai_compatible as oc
from api.routers.openai_compatible import note_speech_activity

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _CyclicModel:
    """Fake model with an internal reference cycle, like an nn.Module tree
    (parent↔child via ``_modules``). Plain refcounting cannot free such an
    object; only a gc pass can — which is exactly what the unload path must
    trigger *after* dropping every reference."""

    def __init__(self):
        self._modules = {}
        self._modules["self"] = self  # cycle


class _FakeBackend:
    """Minimal backend stand-in for watchdog / endpoint tests."""

    def __init__(self, ready: bool = True):
        self._ready = ready
        self.unload_calls = 0

    def is_ready(self) -> bool:
        return self._ready

    def unload(self) -> bool:
        self.unload_calls += 1
        self._ready = False
        return True

    def get_backend_name(self) -> str:
        return "fake"

    def get_model_id(self) -> str:
        return "fake/model"

    def get_device_info(self) -> dict:
        return {"device": "cpu", "gpu_available": False}


@pytest.fixture(autouse=True)
def _reset_state():
    """Isolate process globals between tests."""
    app.state.backend_unloaded = False
    yield
    app.state.backend_unloaded = False
    backend_factory.reset_backend()


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    """Poll *predicate* (called from the test thread) until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# ---------------------------------------------------------------------------
# release_model_memory + base-class unload default
# ---------------------------------------------------------------------------


class TestReleaseModelMemory:
    def test_base_unload_default_is_noop(self):
        class _Bare(TTSBackend):
            async def initialize(self): ...
            async def generate_speech(self, text, voice, language="Auto",
                                      instruct=None, speed=1.0): ...
            def get_backend_name(self): return "bare"
            def get_model_id(self): return "bare/model"
            def get_supported_voices(self): return []
            def get_supported_languages(self): return []
            def is_ready(self): return False
            def get_device_info(self): return {}

        assert _Bare().unload() is False


# ---------------------------------------------------------------------------
# unload_backend() factory function
# ---------------------------------------------------------------------------


class TestUnloadBackendFactory:
    @pytest.mark.asyncio
    async def test_noop_when_no_backend_exists(self):
        assert await backend_factory.unload_backend() is False

    @pytest.mark.asyncio
    async def test_noop_when_backend_not_ready(self, monkeypatch):
        fake = _FakeBackend(ready=False)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)
        assert await backend_factory.unload_backend() is False
        assert fake.unload_calls == 0

    @pytest.mark.asyncio
    async def test_unloads_ready_backend(self, monkeypatch):
        fake = _FakeBackend(ready=True)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)
        assert await backend_factory.unload_backend() is True
        assert fake.unload_calls == 1
        # Second call: nothing to unload anymore.
        assert await backend_factory.unload_backend() is False


# ---------------------------------------------------------------------------
# Official backend unload / re-init
# ---------------------------------------------------------------------------


class TestOfficialBackendUnload:
    def _official_backend(self):
        from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend

        return OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )

    def test_unload_releases_model_and_voices(self):
        backend = self._official_backend()
        backend.model = object()
        backend._ready = True
        backend.device = "cpu"
        backend._custom_voices = {"myvoice": object()}
        backend._custom_voices_dir = "/tmp/voices"
        backend._custom_voices_loaded = True

        assert backend.unload() is True
        assert backend.model is None
        assert backend.is_ready() is False
        # Custom-voice prompts are GPU tensors; unload must drop them.
        assert backend._custom_voices == {}
        # The remembered dir/flag survive so a re-init can reload them.
        assert backend._custom_voices_dir == "/tmp/voices"
        assert backend._custom_voices_loaded is True

    def test_unload_noop_when_not_loaded(self):
        backend = self._official_backend()
        assert backend.unload() is False

    @pytest.mark.asyncio
    async def test_reinit_reloads_custom_voices(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        backend = self._official_backend()

        # Local model dir so resolve_model_path() succeeds.
        models_dir = tmp_path / "models"
        (models_dir / "Qwen3-TTS-12Hz-1.7B-Base").mkdir(parents=True)
        monkeypatch.setenv("TTS_MODELS_DIR", str(models_dir))

        # Simulate "was loaded, then unloaded".
        backend._custom_voices_dir = str(tmp_path / "voices")
        backend._custom_voices_loaded = True

        spy = AsyncMock()
        backend.load_custom_voices = spy

        import qwen_tts

        monkeypatch.setattr(
            qwen_tts.Qwen3TTSModel,
            "from_pretrained",
            staticmethod(lambda path, **kwargs: MagicMock()),
        )
        # initialize() applies torch.compile() when CUDA is available; compiling
        # a MagicMock would hang dynamo, so stub it out.
        import torch

        monkeypatch.setattr(torch, "compile", lambda m, **kwargs: m)
        await backend.initialize()

        assert backend.is_ready() is True
        spy.assert_awaited_once_with(backend._custom_voices_dir)

    @pytest.mark.asyncio
    async def test_load_custom_voices_is_idempotent_per_dir(self, tmp_path):
        backend = self._official_backend()
        original_voice = object()
        backend._custom_voices = {"v": original_voice}
        backend._custom_voices_dir = str(tmp_path)

        # If the early-return fails, the real load path runs and reaches
        # supports_voice_cloning — make that loud.
        def _boom():
            raise AssertionError("load path must not run when already loaded")

        backend.supports_voice_cloning = _boom

        await backend.load_custom_voices(str(tmp_path))
        assert backend._custom_voices == {"v": original_voice}


# ---------------------------------------------------------------------------
# Optimized backend unload (primary user deployment)
# ---------------------------------------------------------------------------


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


class TestOptimizedBackendUnload:
    def _make_backend(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        import yaml

        cfg = _base_config()
        cfg.setdefault("optimization", {})["use_compile"] = False
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))
        monkeypatch.setenv("TTS_CONFIG", str(config_file))

        # Local model dirs so resolve_model_path() finds them.
        models_dir = tmp_path / "models"
        for info in cfg["models"].values():
            (models_dir / info["hf_id"].split("/")[-1]).mkdir(
                parents=True, exist_ok=True
            )
        monkeypatch.setenv("TTS_MODELS_DIR", str(models_dir))

        import qwen_tts
        from api.backends import optimized_backend

        monkeypatch.setattr(
            qwen_tts.Qwen3TTSModel,
            "from_pretrained",
            staticmethod(lambda hf_id, **kwargs: MagicMock()),
        )
        return optimized_backend.OptimizedQwen3TTSBackend()

    @pytest.mark.asyncio
    async def test_unload_clears_models_and_prompt_cache(self, tmp_path, monkeypatch):
        backend = self._make_backend(tmp_path, monkeypatch)

        await backend._ensure_model_loaded("cv")
        backend._voice_prompt_cache["some-key"] = object()
        assert backend.is_ready() is True
        assert backend.get_loaded_model_keys() == ["cv"]

        assert backend.unload() is True
        assert backend.get_loaded_model_keys() == []
        assert backend.model is None
        assert backend.current_model_key is None
        assert backend._voice_prompt_cache == {}
        assert backend.is_ready() is False

        # Idempotent: nothing left to unload.
        assert backend.unload() is False

    @pytest.mark.asyncio
    async def test_next_request_reloads_after_unload(self, tmp_path, monkeypatch):
        backend = self._make_backend(tmp_path, monkeypatch)

        await backend._ensure_model_loaded("cv")
        backend.unload()
        assert backend.is_ready() is False

        # The self-heal path used by generation: _ensure_model_loaded.
        await backend._ensure_model_loaded("cv")
        assert backend.is_ready() is True
        assert backend.get_loaded_model_keys() == ["cv"]


# ---------------------------------------------------------------------------
# The unload must *actually* free the model (VRAM release)
# ---------------------------------------------------------------------------


class TestUnloadActuallyReleasesModel:
    """Regression: gc.collect()+empty_cache() used to run while the calling
    frame still held a reference to the model, so the weights were never
    collected and lingered in the allocator's reserved pool — nvidia-smi
    kept showing the memory even though the code logged a successful unload.
    """

    def test_optimized_unload_frees_cyclic_model(self, tmp_path, monkeypatch):
        import gc
        import weakref

        backend = TestOptimizedBackendUnload()._make_backend(tmp_path, monkeypatch)
        model = _CyclicModel()
        backend._models["cv"] = model
        backend.model = model
        backend.current_model_key = "cv"
        backend._ready = True
        ref = weakref.ref(model)
        del model

        gc_was_enabled = gc.isenabled()
        gc.disable()  # deterministic: only the explicit gc.collect() can free it
        try:
            assert backend.unload() is True
            # If any reference survived the flush (the old bug), the cycle
            # stays alive and the weakref is still populated.
            assert ref() is None, "model was not actually garbage-collected"
        finally:
            if gc_was_enabled:
                gc.enable()

    def test_official_unload_frees_cyclic_model(self):
        import gc
        import weakref

        backend = TestOfficialBackendUnload()._official_backend()
        model = _CyclicModel()
        backend.model = model
        backend._ready = True
        backend.device = "cpu"
        ref = weakref.ref(model)
        del model

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            assert backend.unload() is True
            assert ref() is None, "model was not actually garbage-collected"
        finally:
            if gc_was_enabled:
                gc.enable()


# ---------------------------------------------------------------------------
# Idle-unload watchdog (api.main lifespan)
# ---------------------------------------------------------------------------


class TestIdleUnloadWatchdog:
    def _install_fake_backend(self, monkeypatch, ready: bool) -> _FakeBackend:
        fake = _FakeBackend(ready=ready)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)
        return fake

    def test_unloads_after_idle(self, monkeypatch):
        monkeypatch.setattr(api_main, "TTS_IDLE_UNLOAD_SECONDS", 1)
        fake = self._install_fake_backend(monkeypatch, ready=True)

        # Record that the watchdog takes the generation slot before unload.
        acquired = []

        @contextlib.asynccontextmanager
        async def recording_slot(as_http: bool = True):
            acquired.append(True)
            yield

        monkeypatch.setattr(oc, "generation_slot", recording_slot)

        with TestClient(app) as client:
            # last_speech_at is set at startup; with a 1s timeout the
            # watchdog should fire within a couple of polls.
            assert _wait_for(lambda: fake.unload_calls == 1), (
                "watchdog did not unload the model"
            )
            assert app.state.backend_unloaded is True
            # /health must distinguish unloaded from initializing.
            assert client.get("/health").json()["status"] == "unloaded"

        assert acquired, "watchdog must hold the generation slot to unload"

    def test_disabled_at_zero(self, monkeypatch):
        monkeypatch.setattr(api_main, "TTS_IDLE_UNLOAD_SECONDS", 0)
        fake = self._install_fake_backend(monkeypatch, ready=True)

        with TestClient(app):
            # Simulate a long idle period; nothing must unload.
            app.state.last_speech_at = time.monotonic() - 9999
            time.sleep(1.5)

        assert fake.unload_calls == 0
        assert app.state.backend_unloaded is False

    def test_skips_when_nothing_loaded(self, monkeypatch):
        monkeypatch.setattr(api_main, "TTS_IDLE_UNLOAD_SECONDS", 1)
        fake = self._install_fake_backend(monkeypatch, ready=False)

        with TestClient(app):
            app.state.last_speech_at = time.monotonic() - 9999
            # Give the watchdog time for at least one poll cycle.
            time.sleep(2.5)

        assert fake.unload_calls == 0

    def test_resets_on_successful_speech(self, monkeypatch):
        monkeypatch.setattr(api_main, "TTS_IDLE_UNLOAD_SECONDS", 1)
        fake = self._install_fake_backend(monkeypatch, ready=True)

        with TestClient(app):
            # Simulate speech activity every 0.4s for ~2.4s (timeout: 1s).
            deadline = time.monotonic() + 2.4
            while time.monotonic() < deadline:
                note_speech_activity(app, samples=10)
                time.sleep(0.4)
            assert fake.unload_calls == 0, (
                "speech activity must reset the idle timer"
            )

            # Now go idle and confirm the unload eventually happens.
            assert _wait_for(lambda: fake.unload_calls == 1)

        assert app.state.backend_unloaded is True


# ---------------------------------------------------------------------------
# Manual unload endpoint
# ---------------------------------------------------------------------------


class TestManualUnloadEndpoint:
    def test_unloads_and_reports_already_unloaded(self, monkeypatch):
        fake = _FakeBackend(ready=True)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)

        client = TestClient(app)
        response = client.post("/v1/audio/unload")
        assert response.status_code == 200
        assert response.json()["status"] == "unloaded"
        assert fake.unload_calls == 1
        assert app.state.backend_unloaded is True

        # Nothing resident anymore.
        response = client.post("/v1/audio/unload")
        assert response.status_code == 200
        assert response.json()["status"] == "already_unloaded"
        assert fake.unload_calls == 1

    def test_busy_when_generation_in_flight(self, monkeypatch):
        fake = _FakeBackend(ready=True)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)

        @contextlib.asynccontextmanager
        async def busy_slot(as_http: bool = True):
            raise oc._GenerationBusyError("generation in progress")
            yield  # pragma: no cover

        monkeypatch.setattr(oc, "generation_slot", busy_slot)

        client = TestClient(app)
        response = client.post("/v1/audio/unload")
        assert response.status_code == 503
        assert response.json()["status"] == "busy"
        assert fake.unload_calls == 0


# ---------------------------------------------------------------------------
# Idle-timer resets on every successful generation path
# ---------------------------------------------------------------------------


class TestIdleTimerResetPaths:
    """Every successful generation path must reset the idle-unload timer.

    Regression: the voice-library clone path (``clone:`` voices inside
    /v1/audio/speech) and the /v1/audio/voice-clone endpoint used to skip
    note_speech_activity, so a successful request was followed by an
    immediate idle unload.
    """

    def _age_last_speech(self) -> float:
        """Pretend the last speech request was an hour ago; return 'now'."""
        app.state.last_speech_at = time.monotonic() - 3600
        return time.monotonic()

    @staticmethod
    def _mock_backend(monkeypatch) -> MagicMock:
        import numpy as np

        mock = MagicMock()
        mock.is_ready.return_value = True
        mock.supports_voice_cloning.return_value = True
        mock.get_model_type.return_value = "base"
        mock.generate_voice_clone = AsyncMock(
            return_value=(np.zeros(8, dtype=np.float32), 24000)
        )
        monkeypatch.setattr(backend_factory, "_backend_instance", mock)
        monkeypatch.setattr(oc, "encode_audio", lambda *_a, **_k: b"audio")
        return mock

    def test_voice_clone_endpoint_resets_timer(self, monkeypatch):
        self._mock_backend(monkeypatch)
        # ref_audio decoding goes through soundfile; stub it.
        monkeypatch.setattr(
            oc.sf, "read", lambda *_a, **_k: (np.zeros(8, dtype=np.float32), 24000)
        )

        client = TestClient(app)
        before = self._age_last_speech()
        response = client.post(
            "/v1/audio/voice-clone",
            json={
                "input": "hello",
                "ref_audio": "dGVzdA==",
                "x_vector_only_mode": True,
                "response_format": "wav",
            },
        )
        assert response.status_code == 200, response.text
        assert app.state.last_speech_at >= before, (
            "a successful voice-clone request must reset the idle timer"
        )

    def test_voice_library_clone_resets_timer(self, monkeypatch, tmp_path):
        import json as _json

        self._mock_backend(monkeypatch)
        # Voice library profile "Alice" with dummy reference audio.
        profile_dir = tmp_path / "profiles" / "alice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "meta.json").write_text(
            _json.dumps(
                {
                    "name": "Alice",
                    "profile_id": "alice",
                    "ref_audio_filename": "reference.wav",
                    "x_vector_only_mode": True,
                    "language": "English",
                }
            ),
            encoding="utf-8",
        )
        (profile_dir / "reference.wav").write_bytes(b"RIFF")
        monkeypatch.setattr(oc, "VOICE_LIBRARY_DIR", tmp_path)
        monkeypatch.setattr(
            oc.sf, "read", lambda *_a, **_k: (np.zeros(8, dtype=np.float32), 24000)
        )
        oc._ref_audio_cache.clear()

        client = TestClient(app)
        before = self._age_last_speech()
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "hello",
                "voice": "clone:Alice",
                "response_format": "wav",
            },
        )
        assert response.status_code == 200, response.text
        assert app.state.last_speech_at >= before, (
            "a successful clone: voice request must reset the idle timer"
        )


# ---------------------------------------------------------------------------
# /health status reporting
# ---------------------------------------------------------------------------


class TestHealthUnloadedStatus:
    def test_healthy_when_ready(self, monkeypatch):
        fake = _FakeBackend(ready=True)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)

        client = TestClient(app)
        assert client.get("/health").json()["status"] == "healthy"

    def test_initializing_when_never_loaded(self, monkeypatch):
        fake = _FakeBackend(ready=False)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)

        app.state.backend_unloaded = False
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "initializing"

    def test_unloaded_after_idle_unload(self, monkeypatch):
        fake = _FakeBackend(ready=False)
        monkeypatch.setattr(backend_factory, "_backend_instance", fake)

        app.state.backend_unloaded = True
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "unloaded"
