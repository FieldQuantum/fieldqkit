"""Centralized credential helpers for multi-platform integrations."""

from __future__ import annotations

import os

# Debug-only fallback. Replace for quick local testing.
DEBUG_QUAFU_API_TOKEN = "FRb5fWkVduBE3VhBcGlfH6DfXZjUfVPJqPWQo`Ii:8T/:KUNxREPzJkMyZEO5B{N3d{OypkJxiY[jxjJyBkPyBkPyFEJ4FUM{BUM3JENzJjPjRYZqKDMxpkJtWnemynJtJTcwOnMtmXZueHRu2XcwW4[vWHbkWYfjpkJzW3d2Kzf"
DEBUG_TIANYAN_LOGIN_KEY = "NGlC79z1l66J95ngf7QvRARX4+YDeWcQQ/ns5kHzbjg="
DEBUG_GUODUN_LOGIN_KEY = "YtoTCR9P6p2Rt72yC0WYetJ/X3XFTYftPDe+aHIDmvE="
DEBUG_TENCENT_API_TOKEN = "cAS2u.w4vMT4Vy18lgdyxqP36tMv6b07xXdroaG18CQ5rr3AhSKZMVrwCNFosehawUySH0kOcHIpIjnP8QuFg-ZTRc4RZZeCPOsGEfDAOdnRKDv0M05ukxbo8x6S0dHjzlY7B-VSkIAnJyU3sMdebeAs"


def get_quafu_api_token() -> str:
    """Return Quafu API token from env first, then debug fallback.

    Returns:
        API token string.
    """
    return os.getenv("QUAFU_API_TOKEN", DEBUG_QUAFU_API_TOKEN)


def get_tianyan_login_key() -> str:
    """Return TianYan login key from env first, then debug fallback.

    Returns:
        Login key string.
    """
    return os.getenv("TIANYAN_LOGIN_KEY", DEBUG_TIANYAN_LOGIN_KEY)


def get_guodun_login_key() -> str:
    """Return GuoDun login key from env first, then debug fallback.

    Returns:
        Login key string.
    """
    return os.getenv("GUODUN_LOGIN_KEY", DEBUG_GUODUN_LOGIN_KEY)


def get_tencent_api_token() -> str:
    """Return Tencent API token from env first, then debug fallback.

    Returns:
        API token string.
    """
    return os.getenv("TENCENT_API_TOKEN", DEBUG_TENCENT_API_TOKEN)
