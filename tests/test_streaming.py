# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the OpenAI-compatible streaming behavior of /v1/audio/speech.

These tests do NOT require PyTorch or CUDA — they exercise the router's
stream_format handling (audio / sse), the speed constraint, instructions
passthrough, and the removed stream/instruct request fields, using a mock
backend.
"""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.backends import factory


def _make_mock_backend(
    chunks=(np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32) * 0.5),
    sr=24000,
    has_streaming=True,
    captures=None,
):
    """Build a mock optimized-style backend with async streaming generators."""

    class _MockBackend:
        def __init__(self):
            self._ready = True
            self.model_type = "customvoice"
            self.captures = captures if captures is not None else {}

        def is_ready(self):
            return True

        def get_backend_name(self):
            return "mock-optimized"

        def get_model_id(self):
            return "mock/model"

        def get_supported_voices(self):
            return ["Vivian"]

        def get_supported_languages(self):
            return ["English", "Auto"]

        def get_model_type(self):
            return self.model_type

        def supports_voice_cloning(self):
            return False

        def is_custom_voice(self, voice):
            return False

        def get_device_info(self):
            return {"device": "cpu", "gpu_available": False}

        async def generate_speech(self, text, voice, language="Auto",
                                  instruct=None, speed=1.0, model="tts-1"):
            self.captures.setdefault("generate_speech", []).append(
                {"text": text, "voice": voice, "instruct": instruct, "speed": speed}
            )
            audio = np.concatenate([np.asarray(c) for c in chunks])
            return audio, sr

        async def generate_speech_streaming(self, text, voice, language="Auto",
                                            instruct=None, model="tts-1", **kw):
            self.captures.setdefault("generate_speech_streaming", []).append(
                {"text": text, "voice": voice, "instruct": instruct}
            )
            if not has_streaming:
                raise NotImplementedError
            for c in chunks:
                yield np.asarray(c), sr

    return _MockBackend()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_backend_after_test():
    yield
    factory.reset_backend()


def _install_backend(backend):
    factory._backend_instance = backend


class TestSchemaRemovedFields:
    """The custom `stream` and `instruct` request fields are removed."""

    def test_stream_field_no_streaming_effect(self, client):
        backend = _make_mock_backend()
        _install_backend(backend)
        response = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hi", "voice": "Vivian",
                  "stream": True},
        )
        # `stream` is no longer a modeled field; the schema ignores unknown
        # extras (forward-compat with OpenAI clients that send other fields),
        # so the request is handled as NON-streaming — no streaming effect.
        assert response.status_code == 200
        assert response.headers["content-type"] != "text/event-stream"

    def test_instruct_field_rejected_or_ignored(self, client):
        backend = _make_mock_backend(captures={})
        _install_backend(backend)
        # `instruct` is no longer a documented field; `instructions` is.
        response = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hi", "voice": "Vivian",
                  "instructions": "Be happy."},
        )
        assert response.status_code == 200
        cap = backend.captures["generate_speech"][-1]
        assert cap["instruct"] == "Be happy."


class TestStreamFormatAudio:
    """stream_format='audio' produces a single continuous byte stream."""

    def test_pcm_audio_stream(self, client):
        chunks = [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32) * 0.5]
        backend = _make_mock_backend(chunks=chunks, sr=24000)
        _install_backend(backend)

        with client.stream("POST", "/v1/audio/speech", json={
            "model": "tts-1", "input": "hello world", "voice": "Vivian",
            "response_format": "pcm", "stream_format": "audio",
        }) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/pcm"
            body = b"".join(response.iter_bytes())

        # Two int16 chunks of 10 samples each => 20 samples * 2 bytes = 40 bytes.
        assert len(body) == 40
        # First chunk is silence (zeros) -> int16 zeros.
        first = np.frombuffer(body[:20], dtype="<i2")
        assert np.all(first == 0)

    def test_wav_audio_stream_single_header(self, client):
        chunks = [np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32) * 0.5]
        backend = _make_mock_backend(chunks=chunks, sr=24000)
        _install_backend(backend)

        with client.stream("POST", "/v1/audio/speech", json={
            "model": "tts-1", "input": "hello world", "voice": "Vivian",
            "response_format": "wav", "stream_format": "audio",
        }) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "audio/wav"
            body = b"".join(response.iter_bytes())

        # Exactly ONE RIFF/WAVE header, then PCM.
        assert body.count(b"RIFF") == 1
        assert body.count(b"WAVE") == 1
        assert body.count(b"fmt ") == 1
        assert body.count(b"data") == 1
        # 44-byte header + 20 samples * 2 bytes
        assert len(body) == 44 + 40

    def test_compressed_audio_stream_byte_chunked(self, client):
        # Mock encode_audio to a fixed blob by patching it on the router module.
        from api.routers import openai_compatible as oc

        chunks = [np.zeros(10, dtype=np.float32)]
        backend = _make_mock_backend(chunks=chunks, sr=24000)
        _install_backend(backend)

        fake_blob = b"MP3DATA" * 100  # 700 bytes
        orig = oc.encode_audio
        oc.encode_audio = lambda audio, fmt, sample_rate=24000: fake_blob
        try:
            with client.stream("POST", "/v1/audio/speech", json={
                "model": "tts-1", "input": "hello world", "voice": "Vivian",
                "response_format": "mp3", "stream_format": "audio",
            }) as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "audio/mpeg"
                body = b"".join(response.iter_bytes())
        finally:
            oc.encode_audio = orig

        # The full encoded container is delivered (byte-chunked, but reassembled).
        assert body == fake_blob


class TestStreamFormatSse:
    """stream_format='sse' produces speech.audio.* Server-Sent Events."""

    def test_sse_event_shape(self, client):
        chunks = [np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32) * 0.5]
        backend = _make_mock_backend(chunks=chunks, sr=24000)
        _install_backend(backend)

        with client.stream("POST", "/v1/audio/speech", json={
            "model": "tts-1", "input": "hello world", "voice": "Vivian",
            "response_format": "pcm", "stream_format": "sse",
        }) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            text = b"".join(response.iter_bytes()).decode("utf-8")

        events = [e for e in text.split("\n\n") if e.strip()]
        # First events are deltas, last is done.
        delta_events = [e for e in events if e.startswith("event: speech.audio.delta")]
        done_events = [e for e in events if e.startswith("event: speech.audio.done")]
        assert len(delta_events) == 2
        assert len(done_events) == 1

        # Parse the first delta payload.
        first = delta_events[0]
        assert "event: speech.audio.delta" in first
        data_line = [ln for ln in first.splitlines() if ln.startswith("data: ")][0][6:]
        payload = json.loads(data_line)
        assert payload["type"] == "speech.audio.delta"
        assert payload["response_format"] == "pcm"
        decoded = base64.b64decode(payload["audio"])
        # 8 samples * 2 bytes = 16 bytes, and zeros => int16 zeros.
        assert len(decoded) == 16
        assert np.all(np.frombuffer(decoded, dtype="<i2") == 0)

        # done event carries a usage object.
        done = done_events[0]
        done_data = [ln for ln in done.splitlines() if ln.startswith("data: ")][0][6:]
        done_payload = json.loads(done_data)
        assert done_payload["type"] == "speech.audio.done"
        assert "usage" in done_payload
        assert done_payload["usage"]["total_tokens"] == 0


class TestStreamingConstraints:
    """Streaming + speed != 1.0 is rejected."""

    def test_speed_with_streaming_rejected(self, client):
        backend = _make_mock_backend()
        _install_backend(backend)
        for fmt in ("pcm", "wav", "mp3"):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "tts-1", "input": "hi", "voice": "Vivian",
                    "response_format": fmt, "stream_format": "audio", "speed": 2.0,
                },
            )
            assert response.status_code == 400, response.text
            assert response.json()["detail"]["error"] == "streaming_speed_unsupported"


class TestInstructionsPassthrough:
    """`instructions` is forwarded to the backend as `instruct=`."""

    def test_instructions_forwarded_non_streaming(self, client):
        backend = _make_mock_backend(captures={})
        _install_backend(backend)
        response = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hi", "voice": "Vivian",
                  "instructions": "Speak warmly."},
        )
        assert response.status_code == 200
        cap = backend.captures["generate_speech"][-1]
        assert cap["instruct"] == "Speak warmly."

    def test_instructions_forwarded_streaming(self, client):
        backend = _make_mock_backend(captures={})
        _install_backend(backend)
        with client.stream("POST", "/v1/audio/speech", json={
            "model": "tts-1", "input": "hi", "voice": "Vivian",
            "response_format": "pcm", "stream_format": "audio",
            "instructions": "Speak warmly.",
        }) as response:
            assert response.status_code == 200
        cap = backend.captures["generate_speech_streaming"][-1]
        assert cap["instruct"] == "Speak warmly."


class TestNonStreamingDefault:
    """No stream_format => single complete response."""

    def test_default_returns_single_response(self, client):
        chunks = [np.zeros(8, dtype=np.float32)]
        backend = _make_mock_backend(chunks=chunks, sr=24000)
        _install_backend(backend)
        response = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hi", "voice": "Vivian",
                  "response_format": "wav"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")
