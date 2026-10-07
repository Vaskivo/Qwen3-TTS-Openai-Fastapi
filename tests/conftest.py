# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures making the test suite hermetic.

Two environment leaks used to break tests on developer machines:

1. Credentials. ``api.security`` reads API_KEY / UI_USER / UI_PASSWORD once at
   import time and enables auth if any is configured. The suite's requests
   carry no credentials, so on a machine with credentials exported (docker
   compose, .env sourcing, ...) every endpoint test failed with 401. The
   autouse fixture below neutralizes the module-level auth config instead.
   If dedicated auth-behavior tests are added later, they can re-enable the
   flag explicitly (e.g. ``monkeypatch.setattr(security, "_AUTH_ENABLED", True)``).

2. onnxruntime telemetry. onnxruntime >= 1.30 collects platform telemetry at
   import time: it fingerprints the machine (``sh -c "echo `blkid; hostname`"``)
   and persists a session/device id as a ``<cwd>/:memory:.ses`` file (a literal
   ":memory:" storage name that native code appends ".ses" to). Importing
   ``qwen_tts`` pulls onnxruntime in, so test runs littered the repo root and
   performed network-adjacent calls. ORT_DISABLE_TELEMETRY=1 turns that off;
   it is set here, before any test module imports qwen_tts.
"""

import os

# Must happen before the first import of qwen_tts/onnxruntime; conftest.py is
# imported by pytest before collecting test modules, so this is early enough.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import pytest

from api import security


@pytest.fixture(autouse=True)
def _open_auth(monkeypatch):
    """Disable credential enforcement for every test, whatever the ambient
    environment has configured (the security module snapshots env vars at
    import time, so patching the environment later would have no effect).

    ``require_auth`` and the gated-ASGI wrapper consult these module globals
    at request time, so patching here covers both route dependencies and
    mounted sub-applications.
    """
    monkeypatch.setattr(security, "_API_KEY", None)
    monkeypatch.setattr(security, "_UI_BASIC_ENABLED", False)
    monkeypatch.setattr(security, "_AUTH_ENABLED", False)
    yield
