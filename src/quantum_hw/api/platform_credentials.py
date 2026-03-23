"""Centralized credential helpers for multi-platform integrations."""

from __future__ import annotations

import os

# Debug-only fallback. Replace for quick local testing.
# Production usage should prefer environment variable QPU_API_TOKEN.
DEBUG_QPU_API_TOKEN = "FRb5fWkVduBE3VhBcGlfH6DfXZjUfVPJqPWQo`Ii:8T/:KUNxREPzJkMyZEO5B{N3d{OypkJxiY[jxjJyBkPyBkPyFEJ4FUM{BUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"

# Debug-only fallback. Replace for quick local testing.
# Production usage should prefer environment variable CQLIB_LOGIN_KEY.
DEBUG_CQLIB_LOGIN_KEY = "NGlC79z1l66J95ngf7QvRARX4+YDeWcQQ/ns5kHzbjg="


def get_quafu_api_token() -> str:
    """Return Quafu API token from env first, then debug fallback."""
    return os.getenv("QPU_API_TOKEN", DEBUG_QPU_API_TOKEN)


def get_cqlib_login_key() -> str:
    """Return cqlib login key from env first, then debug fallback."""
    return os.getenv("CQLIB_LOGIN_KEY", DEBUG_CQLIB_LOGIN_KEY)
