"""
OpenJEV Bearer Authentication Middleware
Enforces Bearer token verification per IMPLEMENTATION.md §8 using constant-time comparison.
"""

import hmac
from typing import List, Set
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from core.config import OpenJEVConfig

EXEMPT_PATHS: Set[str] = {
    "/healthz",
    "/readyz",
    "/metrics",
    "/docs",
    "/openapi.json"
}

class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: OpenJEVConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next):
        # 1. Skip if auth is disabled or path is exempt
        if not self.config.server.auth.enabled or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # 2. Extract Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Missing or malformed Authorization header. Expected 'Bearer <token>'."
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = auth_header[7:].strip()
        valid_keys: List[str] = self.config.server.auth.get_resolved_keys()

        # 3. Constant-time comparison to prevent timing attacks
        authenticated = any(
            hmac.compare_digest(token.encode("utf-8"), key.encode("utf-8"))
            for key in valid_keys
            if key
        )

        if not authenticated:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Invalid Bearer token."
                },
                headers={"WWW-Authenticate": "Bearer"}
            )

        return await call_next(request)
