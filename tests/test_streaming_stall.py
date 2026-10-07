# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for streaming robustness against stalled or disconnected
clients (the "second request hangs forever" bug).

Background: the response generator used to hold ``_generation_semaphore``
while yielding to the client. A client that stopped reading a stream (without
closing the connection) blocked uvicorn's write flow-control forever while
the semaphore was still held, wedging every subsequent request behind it.

The router now drives generation in a decoupled *producer task* (which holds
the generation slot) and streams to the client from a bounded queue, with two
watchdogs:

  - TTS_STALL_TIMEOUT: abort generation when the client has not consumed
    audio for this many seconds (frees the generation slot).
  - TTS_ACQUIRE_TIMEOUT: bound how long a request may wait for the slot
    before failing (HTTP 503 for non-streaming, an SSE error event for
    streaming).

These tests exercise the machinery directly (no PyTorch / CUDA needed).
"""

import asyncio
import time
import types

import numpy as np
import pytest
from fastapi import HTTPException

import api.routers.openai_compatible as oc


SR = 24000


@pytest.fixture(autouse=True)
def _fresh_semaphore_and_timeouts(monkeypatch):
    """Isolate the module-global generation slot per test.

    The module-level semaphore binds to the first event loop that contends
    it; pytest-asyncio gives each test a fresh loop, so swap in a fresh
    semaphore (and fast watchdog timeouts) for the duration of each test.
    """
    original = oc._generation_semaphore
    oc._generation_semaphore = asyncio.Semaphore(oc._MAX_CONCURRENT)
    monkeypatch.setattr(oc, "_STALL_TIMEOUT", 0.2)
    monkeypatch.setattr(oc, "_STREAM_QUEUE_MAX", 2)
    monkeypatch.setattr(oc, "_ACQUIRE_TIMEOUT", 0.2)
    yield
    oc._generation_semaphore = original


async def _slow_source(n=500, delay=0.01):
    """Async PCM source: many small chunks, slower than any consumer stall."""
    for _ in range(n):
        await asyncio.sleep(delay)
        yield np.ones(100, dtype=np.float32), SR


class TestStallWatchdog:
    def test_stalled_client_aborts_generation_and_releases_slot(self):
        async def run():
            queue = asyncio.Queue(maxsize=oc._STREAM_QUEUE_MAX)
            t0 = time.time()
            # Nobody consumes: the stall watchdog must abort the producer.
            await oc._pcm_stream_producer(_slow_source, queue, "test")
            return time.time() - t0

        elapsed = asyncio.run(run())
        assert elapsed < 3.0
        assert oc._generation_semaphore._value == oc._MAX_CONCURRENT

    def test_stalled_consumers_stream_terminates(self):
        async def run():
            def make_producer(queue):
                return oc._pcm_stream_producer(_slow_source, queue, "test")

            gen = oc._stream_from_producer(
                make_producer, "pcm", False,
                types.SimpleNamespace(app=None), "test",
            )
            chunks = 0
            t0 = time.time()
            try:
                while True:
                    await asyncio.wait_for(gen.__anext__(), timeout=5.0)
                    chunks += 1
                    await asyncio.sleep(0.5)  # consume far too slowly
            except StopAsyncIteration:
                pass
            return time.time() - t0, chunks

        elapsed, chunks = asyncio.run(run())
        # The stream must end on its own (truncated by the stall watchdog)
        # instead of hanging forever.
        assert elapsed < 8.0
        assert chunks >= 1
        assert oc._generation_semaphore._value == oc._MAX_CONCURRENT


class TestDisconnect:
    def test_closing_stream_midway_releases_slot_promptly(self):
        async def run():
            def make_producer(queue):
                return oc._pcm_stream_producer(_slow_source, queue, "test")

            gen = oc._stream_from_producer(
                make_producer, "pcm", False,
                types.SimpleNamespace(app=None), "test",
            )
            item = await gen.__anext__()
            assert item
            await gen.aclose()  # simulates Starlette tearing down the response
            # The generation slot must be free again without waiting for the
            # stall watchdog.
            await asyncio.wait_for(
                oc._generation_semaphore.acquire(), timeout=2.0
            )
            oc._generation_semaphore.release()

        asyncio.run(run())


class TestAcquireTimeout:
    def test_acquire_timeout_raises_http_503(self):
        async def run():
            await oc._generation_semaphore.acquire()
            try:
                with pytest.raises(HTTPException) as excinfo:
                    await oc._acquire_generation_slot(as_http=True)
                assert excinfo.value.status_code == 503
            finally:
                oc._generation_semaphore.release()

        asyncio.run(run())

    def test_acquire_timeout_raises_busy_error_for_streaming(self):
        async def run():
            await oc._generation_semaphore.acquire()
            try:
                with pytest.raises(oc._GenerationBusyError):
                    await oc._acquire_generation_slot(as_http=False)
            finally:
                oc._generation_semaphore.release()

        asyncio.run(run())

    def test_no_permit_leaked_after_acquire_timeout(self):
        async def run():
            # Slot held by "another request"; a waiter times out and gives up.
            await oc._generation_semaphore.acquire()
            try:
                with pytest.raises(oc._GenerationBusyError):
                    await oc._acquire_generation_slot(as_http=False)
            finally:
                oc._generation_semaphore.release()
            # The timed-out waiter must not have consumed the permit — the
            # next acquire succeeds immediately (a leaked permit would hang).
            await asyncio.wait_for(
                oc._acquire_generation_slot(as_http=False), timeout=1.0
            )
            oc._generation_semaphore.release()

        asyncio.run(run())


class TestNextStreamItem:
    def test_none_when_producer_exits_without_sentinel(self):
        async def run():
            queue = asyncio.Queue(maxsize=4)

            async def exit_without_sentinel():
                return  # e.g. stall abort with a full queue: sentinel dropped

            producer = asyncio.ensure_future(exit_without_sentinel())
            item = await oc._next_stream_item(queue, producer)
            assert item is None

        asyncio.run(run())

    def test_items_still_drained_after_producer_exit(self):
        async def run():
            queue = asyncio.Queue(maxsize=4)

            async def push_then_exit():
                queue.put_nowait(("chunk", (np.ones(8, dtype=np.float32), SR)))
                return

            producer = asyncio.ensure_future(push_then_exit())
            item = await oc._next_stream_item(queue, producer)
            assert item is not None and item[0] == "chunk"
            assert await oc._next_stream_item(queue, producer) is None

        asyncio.run(run())


class TestGenerationSlotContext:
    def test_slot_released_on_exception(self):
        async def run():
            with pytest.raises(RuntimeError):
                async with oc._generation_slot():
                    raise RuntimeError("boom")
            assert oc._generation_semaphore._value == oc._MAX_CONCURRENT

        asyncio.run(run())

    def test_slot_serializes_concurrent_holders(self):
        async def run():
            seen = []

            async def worker(i):
                async with oc._generation_slot(as_http=False):
                    seen.append(("enter", i))
                    await asyncio.sleep(0.05)
                    seen.append(("exit", i))

            await asyncio.gather(worker(1), worker(2))
            # Holders never overlapped beyond the configured capacity.
            depth = 0
            max_depth = 0
            for kind, _ in seen:
                depth += 1 if kind == "enter" else -1
                max_depth = max(max_depth, depth)
            assert max_depth == oc._MAX_CONCURRENT
            assert oc._generation_semaphore._value == oc._MAX_CONCURRENT

        asyncio.run(run())
