"""Centralized credential helpers for multi-platform integrations.

Credentials are resolved in the following order (first match wins):

1. **Configuration file** ``.quantum_hw.yaml`` discovered from common roots:

    - current working directory and its ancestors
    - package installation directory and its ancestors

    (copy from ``.quantum_hw.example.yaml`` and fill in your tokens).
2. **Environment variables**:

   - ``QUAFU_API_TOKEN``   – 夸父量子云 (https://quafu-sqc.baqis.ac.cn/)
   - ``TIANYAN_API_TOKEN``  – 天衍量子云 (https://qc.zdxlz.com/)
   - ``GUODUN_API_TOKEN``  – 国盾量子云 (https://quantumctek-cloud.com/)
   - ``TENCENT_API_TOKEN``  – 腾讯量子云 (https://quantum.tencent.com/cloud/)
   - ``ORIGIN_API_TOKEN``   – 本源量子云 (https://qcloud.originqc.com.cn/)
   - ``FIELDQUANTUM_API_TOKEN`` – 量坤乾坤云 (https://fieldquantum.tech/)

The ``.quantum_hw.yaml`` file is excluded from Git via ``.gitignore``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = ".quantum_hw.yaml"

# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

_cached_config: Optional[Dict[str, Any]] = None


def _iter_config_candidates() -> list[Path]:
    """Return de-duplicated config file candidates ordered by priority."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    # Optional explicit override for power users.
    env_path = os.getenv("QUANTUM_HW_CONFIG")
    if env_path:
        explicit_path = Path(env_path).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = Path.cwd() / explicit_path
        explicit_path = explicit_path.resolve()
        seen.add(explicit_path)
        candidates.append(explicit_path)

    module_file = Path(__file__).resolve()
    search_starts = [
        Path.cwd(),
        module_file.parent,       # .../quantum_hw/api
        module_file.parents[1],   # .../quantum_hw
    ]

    for start in search_starts:
        for directory in (start, *start.parents):
            path = (directory / _CONFIG_FILENAME).resolve()
            if path in seen:
                continue
            seen.add(path)
            candidates.append(path)

    return candidates


def _load_config(*, force: bool = False) -> Dict[str, Any]:
    """Load and cache the project-local config file.

    Returns an empty dict when no config file exists.
    """
    global _cached_config
    if _cached_config is not None and not force:
        return _cached_config

    try:
        import yaml
    except ImportError:
        _cached_config = {}
        return _cached_config

    # Search current/workspace roots first, then package-install roots.
    for path in _iter_config_candidates():
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict):
                    logger.debug("Loaded credentials config from %s", path)
                    _cached_config = data
                    return _cached_config
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse config %s: %s", path, exc)
                continue

    _cached_config = {}
    return _cached_config


def reload_config() -> None:
    """Force-reload the configuration file (useful after editing)."""
    _load_config(force=True)


# ---------------------------------------------------------------------------
# Credential resolution: config file → env var → error
# ---------------------------------------------------------------------------

_CREDENTIAL_MAP: Dict[str, tuple[str, str, str]] = {
    # key: (config_yaml_path_section, config_yaml_key, env_var)
    "quafu":   ("quafu",   "api_token",  "QUAFU_API_TOKEN"),
    "tianyan":  ("tianyan",  "api_token",  "TIANYAN_API_TOKEN"),
    "guodun":  ("guodun",  "api_token",  "GUODUN_API_TOKEN"),
    "tencent": ("tencent", "api_token",  "TENCENT_API_TOKEN"),
    "origin":  ("origin",  "api_token",  "ORIGIN_API_TOKEN"),
    "fieldquantum": ("fieldquantum", "api_token", "FIELDQUANTUM_API_TOKEN"),
}

_PLATFORM_LABELS: Dict[str, str] = {
    "quafu":   "Quafu (夸父)  – https://quafu-sqc.baqis.ac.cn/",
    "tianyan":  "TianYan (天衢) – https://qc.zdxlz.com/",
    "guodun":  "GuoDun (国盾)  – https://quantumctek-cloud.com/",
    "tencent": "Tencent (腾讯) – https://quantum.tencent.com/cloud/",
    "origin":  "Origin (本源) – https://qcloud.originqc.com.cn/",
    "fieldquantum": "FieldQuantum (量坤) – https://fieldquantum.tech/",
}


def _get_credential(platform: str) -> str:
    """Resolve a credential for *platform*.

    Lookup order: config file ``credentials.<platform>.<key>`` →
    environment variable → ``ValueError``.
    """
    section, key, env_var = _CREDENTIAL_MAP[platform]

    # 1. config file
    cfg = _load_config()
    creds = cfg.get("credentials", {})
    if isinstance(creds, dict):
        plat_section = creds.get(section, {})
        if isinstance(plat_section, dict):
            value = plat_section.get(key)
            if value:
                return str(value)

    # 2. environment variable
    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    # 3. error
    label = _PLATFORM_LABELS[platform]
    raise ValueError(
        f"Credential for {label} not found.\n"
        f"Please set it in one of the following ways:\n"
        f"  1. Copy .quantum_hw.example.yaml to .quantum_hw.yaml and fill in:\n"
        f"     credentials:\n"
        f"       {section}:\n"
        f"         {key}: <your-token>\n"
        f"  2. Set environment variable: export {env_var}=<your-token>\n"
        f"Obtain your token from: {label.split(' – ')[-1]}"
    )


def get_quafu_api_token() -> str:
    """Return Quafu API token (config file → ``QUAFU_API_TOKEN`` env var).

    Returns:
        API token string.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("quafu")


def get_tianyan_api_token() -> str:
    """Return TianYan API token (config file → ``TIANYAN_API_TOKEN`` env var).

    Returns:
        API token string.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("tianyan")


def get_guodun_api_token() -> str:
    """Return GuoDun API token (config file → ``GUODUN_API_TOKEN`` env var).

    Returns:
        API token string.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("guodun")


def get_tencent_api_token() -> str:
    """Return Tencent API token (config file → ``TENCENT_API_TOKEN`` env var).

    Returns:
        API token string.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("tencent")


def get_origin_api_token() -> str:
    """Return Origin (本源) API token (config file → ``ORIGIN_API_TOKEN`` env var).

    Returns:
        API token string.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("origin")


def get_fieldquantum_api_token() -> str:
    """Return FieldQuantum API token (config file → ``FIELDQUANTUM_API_TOKEN`` env var).

    Returns:
        API token string of the form ``fq_<32hex>``.

    Raises:
        ValueError: If no credential is found.
    """
    return _get_credential("fieldquantum")
