import asyncio
import struct
import sys

import numpy as np
import pytest

from api.services.audio_encoding import (
    AudioEncodingError,
    convert_to_pcm,
    convert_to_wav,
    encode_audio,
)
from api.services.text_processing import normalize_text


def test_url_normalization_handles_normal_domains():
    normalized = normalize_text("Visit https://example.com/docs and example.nz now.")
    assert "example dot com slash docs" in normalized
    assert "example dot nz" in normalized


def test_currency_is_not_misread_as_inches():
    normalized = normalize_text("Pay -$1.50 in 1 min.")
    assert "minus" in normalized
    assert "dollar" in normalized
    assert "inch" not in normalized
    assert "minute" in normalized


def test_markdown_headings_get_a_pause_before_body():
    # Heading has no terminal punctuation -> a period is appended so the
    # model takes a breath before the body that follows. A real newline is
    # kept between heading and body so the chunker inserts actual silence.
    normalized = normalize_text("# Quick Title\nBody text right after.")
    assert normalized == "Quick Title.\nBody text right after.", normalized

    # Heading already ends with punctuation -> no doubling.
    normalized = normalize_text("# What is this?\n\nSome answer.")
    assert normalized == "What is this?\nSome answer.", normalized

    # Closing '#'-run is stripped.
    normalized = normalize_text("### Title ###\n\nBody.")
    assert normalized == "Title.\nBody.", normalized

    # Heading marker is only stripped at the start of a line; a '#' mid-line
    # is left for the legacy symbol replacement (-> 'number').
    mid = normalize_text("This is a # hashtag mid-line.")
    assert "number" in mid and "hashtag mid-line" in mid


def test_markdown_paragraph_breaks_become_pauses():
    # Paragraph breaks are preserved as a newline so the chunker splits them
    # into separate chunks and inserts real silence between them.
    normalized = normalize_text("Para one.\n\nPara two.\n\nPara three.")
    assert normalized == "Para one.\nPara two.\nPara three.", normalized


def test_markdown_list_items_get_pauses_between_them():
    from api.routers.openai_compatible import _split_into_chunks

    # Each unordered item is its own chunk (block boundary) so a real pause
    # precedes every item, rather than comma-joining them into one chunk.
    normalized = normalize_text("- a\n- b\n- c")
    assert normalized == "a.\nb.\nc.", normalized
    chunks = _split_into_chunks(normalized, 20, 70)
    assert [c.text for c in chunks] == ["a.", "b.", "c."], [c.text for c in chunks]
    assert all(c.starts_block for c in chunks)

    # Ordered lists speak the ordinal first; each item is its own chunk.
    normalized = normalize_text("1. one\n2. two\n3. three")
    assert normalized == "one, one.\ntwo, two.\nthree, three.", normalized
    chunks = _split_into_chunks(normalized, 20, 70)
    assert [c.text for c in chunks] == [
        "one, one.", "two, two.", "three, three.",
    ], [c.text for c in chunks]
    assert all(c.starts_block for c in chunks)

    # Numbers/units inside list items still get normalized.
    normalized = normalize_text("- I have 3 apples\n- And 10KB of data")
    assert "three apples" in normalized
    assert "kilobytes" in normalized


def test_markdown_inline_syntax_is_stripped():
    normalized = normalize_text("Here is **bold** and *italic* and a [link](https://example.com) and `code`.")
    assert normalized == "Here is bold and italic and a link and code.", normalized


def test_markdown_normalization_can_be_disabled():
    from api.structures.schemas import NormalizationOptions

    opts = NormalizationOptions(markdown_normalization=False)
    # With markdown normalization off, the legacy '#' symbol replacement kicks in
    # and newlines are flattened to spaces (no chunked paragraph silence).
    assert normalize_text("# Heading\n\nBody.", opts) == "number Heading Body."


def test_chunker_keeps_paragraph_boundaries_for_silence():
    from api.routers.openai_compatible import _split_into_chunks, Chunk

    # Short paragraphs that would otherwise pack into one chunk are kept
    # separate so the merge step inserts real silence between them.
    text = "Para one. Short.\nPara two. Short.\nPara three. Short."
    chunks = _split_into_chunks(text, 20, 70)
    assert [c.text for c in chunks] == [
        "Para one. Short.",
        "Para two. Short.",
        "Para three. Short.",
    ], [c.text for c in chunks]
    # Every chunk is a block boundary (separated by newlines).
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.starts_block for c in chunks)

    # A single long paragraph still splits on sentence punctuation as before.
    long_para = (
        "This is a longer paragraph with multiple sentences. "
        "Here is the second sentence. And a third to finish."
    )
    chunks = _split_into_chunks(long_para, 20, 70)
    assert len(chunks) >= 2
    # No chunk spans the (absent) paragraph boundary spuriously.
    assert all("\n" not in c.text for c in chunks)
    # The first chunk starts a block; subsequent sentence splits do not.
    assert chunks[0].starts_block is True
    assert all(not c.starts_block for c in chunks[1:])


def test_normalized_markdown_chunks_per_block():
    from api.routers.openai_compatible import _split_into_chunks

    md = (
        "# Title\n\n"
        "First paragraph.\n\n"
        "- a\n- b\n- c\n\n"
        "1. one\n2. two\n\n"
        "Closing paragraph."
    )
    normalized = normalize_text(md)
    chunks = _split_into_chunks(normalized, 20, 70)
    # Each block becomes its own chunk; each list item is its own chunk (with
    # a pause before it).
    assert [c.text for c in chunks] == [
        "Title.",
        "First paragraph.",
        "a.",
        "b.",
        "c.",
        "one, one.",
        "two, two.",
        "Closing paragraph.",
    ], [c.text for c in chunks]
    # Every chunk is a block boundary -> paragraph gap is used before each.
    assert all(c.starts_block for c in chunks)


def test_list_is_its_own_block_even_after_prose():
    """A list that immediately follows a prose line (no blank line) must still
    become separate chunks (one per item) so real silence precedes each item,
    rather than comma-joining them into the preceding line."""
    from api.routers.openai_compatible import _split_into_chunks

    md = "# Heading\nLead-in line with no blank after heading\nThe things I like\n* carrots\n* video games\n* Rammstein"
    normalized = normalize_text(md)
    chunks = _split_into_chunks(normalized, 20, 70)
    assert [c.text for c in chunks] == [
        "Heading.",
        "Lead-in line with no blank after heading.",
        "The things I like.",
        "carrots.",
        "video games.",
        "Rammstein.",
    ], [c.text for c in chunks]
    assert all(c.starts_block for c in chunks)


def test_nested_list_each_item_its_own_chunk():
    """Nested list items (any depth, any marker) each become their own chunk
    with a pause before them."""
    from api.routers.openai_compatible import _split_into_chunks

    md = (
        "- outer one\n"
        "  - inner a\n"
        "  - inner b\n"
        "- outer two\n"
        "1. first\n"
        "2. second"
    )
    normalized = normalize_text(md)
    chunks = _split_into_chunks(normalized, 20, 70)
    assert [c.text for c in chunks] == [
        "outer one.",
        "inner a.",
        "inner b.",
        "outer two.",
        "one, first.",
        "two, second.",
    ], [c.text for c in chunks]
    assert all(c.starts_block for c in chunks)


def test_pcm_is_signed_16_bit_little_endian():
    payload = convert_to_pcm(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert struct.unpack("<hhh", payload) == (-32768, 0, 32767)


def test_wav_header_matches_stereo_payload():
    audio = np.zeros((10, 2), dtype=np.float32)
    payload = convert_to_wav(audio, sample_rate=24000)
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    assert struct.unpack("<H", payload[22:24])[0] == 2
    assert struct.unpack("<I", payload[40:44])[0] == 40
    assert len(payload) == 84


def test_empty_audio_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        convert_to_pcm(np.array([], dtype=np.float32))


def test_compressed_encoding_never_silently_returns_wav(monkeypatch):
    monkeypatch.setitem(sys.modules, "pydub", None)
    with pytest.raises(AudioEncodingError, match="pydub"):
        encode_audio(np.zeros(32, dtype=np.float32), "mp3", 24000)


@pytest.mark.asyncio
async def test_backend_initializes_only_once_under_concurrency(monkeypatch):
    from api.backends import factory

    class DummyBackend:
        def __init__(self):
            self.ready = False
            self.initialize_calls = 0
            self.load_voice_calls = 0

        def is_ready(self):
            return self.ready

        async def initialize(self):
            self.initialize_calls += 1
            await asyncio.sleep(0.02)
            self.ready = True

        async def load_custom_voices(self, _path):
            self.load_voice_calls += 1

        def get_backend_name(self):
            return "dummy"

        def get_model_id(self):
            return "dummy/model"

    dummy = DummyBackend()
    monkeypatch.setattr(factory, "_backend_instance", dummy)
    monkeypatch.setattr(factory, "_initialization_lock", None)

    first, second = await asyncio.gather(
        factory.initialize_backend(),
        factory.initialize_backend(),
    )

    assert first is dummy and second is dummy
    assert dummy.initialize_calls == 1
    assert dummy.load_voice_calls == 1


def test_invalid_float_environment_falls_back(monkeypatch):
    from api.backends import factory

    monkeypatch.setenv("TTS_WARMUP_MAX_SECONDS", "not-a-number")
    assert factory._env_float("TTS_WARMUP_MAX_SECONDS", 10.0) == 10.0
