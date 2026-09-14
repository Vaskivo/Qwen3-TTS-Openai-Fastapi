# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""API-key authentication for the Qwen3-TTS API.

When the ``API_KEY`` environment variable is set, every protected route
requires a matching key. When it is unset (local/dev), authentication is
disabled and all routes are open — so existing deployments keep working
unless an operator opts in by setting ``API_KEY``.

Two header forms are accepted, so the server interoperates with both the
OpenAI client convention and the common "X-API-Key" convention:

* ``Authorization: Bearer <key>``   (OpenAI / standard bearer)
* ``X-API-Key: <key>``              (simple API-key header)

A request that supplies neither header, or supplies a wrong key, gets a
``401 Unauthorized`` response matching the OpenAI error shape used elsewhere
in this server.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Awaitable, Callable, Optional

from fastapi import Header, HTTPException, status
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Read once at import time. Empty/whitespace => auth disabled (dev mode).
_API_KEY: Optional[str] = os.getenv("API_KEY", "").strip() or None
_AUTH_ENABLED: bool = _API_KEY is not None

if _AUTH_ENABLED:
    # Avoid logging the key itself; just confirm auth is on.
    logger.info("API key authentication enabled (API_KEY is set).")
else:
    logger.warning(
        "API_KEY is not set — API authentication is DISABLED. "
        "Set API_KEY to require a bearer/X-API-Key token on all routes."
    )


def is_auth_enabled() -> bool:
    """Return True if API-key authentication is currently enforced."""
    return _AUTH_ENABLED


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "invalid_api_key",
            "message": detail,
            "type": "invalid_request_error",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency that enforces a valid API key when auth is enabled.

    Accepts the key from either ``Authorization: Bearer <key>`` or
    ``X-API-Key: <key>``. Uses :func:`hmac.compare_digest` for constant-time
    comparison to avoid timing side channels.

    When ``API_KEY`` is unset, this dependency is a no-op (auth disabled).
    """
    if not _AUTH_ENABLED:
        return  # Local/dev: open by default.

    provided: Optional[str] = None

    # Prefer an explicit X-API-Key header, then fall back to Bearer.
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()

    if not provided:
        raise _unauthorized(
            "Missing API key. Provide it via the 'Authorization: Bearer <key>' "
            "or 'X-API-Key: <key>' header."
        )

    # Constant-time comparison to avoid timing attacks.
    if not hmac.compare_digest(provided, _API_KEY):  # type: ignore[arg-type]
        raise _unauthorized("Invalid API key.")

    return


# ---------------------------------------------------------------------------
# ASGI wrapper for gating mounted sub-applications (StaticFiles, etc.)
# ---------------------------------------------------------------------------
# FastAPI/Starlette route-level ``dependencies`` do NOT apply to routes added
# via ``app.mount(...)`` — mounted ASGI sub-apps bypass the router. To gate
# static assets (and any other mounted sub-app) behind the same API key, we
# wrap the sub-app in a tiny ASGI middleware that runs the key check on every
# request before delegating to the wrapped app.


def _unauthorized_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "error": "invalid_api_key",
            "message": (
                "Missing API key. Provide it via the "
                "'Authorization: Bearer <key>' or 'X-API-Key: <key>' header."
            ),
            "type": "invalid_request_error",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_key(scope) -> Optional[str]:
    """Pull the API key from request headers in a raw ASGI scope."""
    headers = dict(scope.get("headers") or [])
    # ASGI header keys are lowercased bytes.
    auth = headers.get(b"authorization")
    if auth:
        try:
            auth_s = auth.decode("latin-1")
        except Exception:
            return None
        parts = auth_s.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    x_key = headers.get(b"x-api-key")
    if x_key:
        try:
            return x_key.decode("latin-1").strip()
        except Exception:
            return None
    return None


def gated_asgi_app(wrapped_app) -> Callable:
    """Wrap an ASGI app so every request requires a valid API key when enabled.

    Use this to gate ``StaticFiles`` and other mounted sub-applications that
    bypass FastAPI's route-level ``dependencies``::

        app.mount("/static", gated_asgi_app(StaticFiles(directory=...)))

    When ``API_KEY`` is unset, the wrapper is a pass-through (auth disabled).
    """

    async def _asgi(scope, receive, send):
        # Only gate HTTP request lifetimes; websocket/lifespan pass through.
        if scope.get("type") != "http" or not _AUTH_ENABLED:
            await wrapped_app(scope, receive, send)
            return

        provided = _extract_key(scope)
        if not provided or not hmac.compare_digest(provided, _API_KEY):  # type: ignore[arg-type]
            response = _unauthorized_response()
            await response(scope, receive, send)
            return

        await wrapped_app(scope, receive, send)

    return _asgi
