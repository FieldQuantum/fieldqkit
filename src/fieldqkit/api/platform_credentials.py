"""Centralized credential helpers for multi-platform integrations.

Credentials are resolved in the following order (first match wins):

1. **Configuration file** ``.quantum_hw.yaml`` discovered from common roots:

    - the path in ``$QUANTUM_HW_CONFIG`` (explicit override)
    - current working directory and its ancestors
    - the per-user locations ``~/.quantum_hw.yaml`` and
      ``~/.config/fieldqkit/credentials.yaml`` (recommended for
      ``pip``-installed users)
    - package installation directory and its ancestors

    Create one quickly with ``fieldqkit-config-init`` (or
    :func:`init_config`), or copy ``.quantum_hw.example.yaml`` from the
    source tree, then fill in your tokens.
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


def _user_config_paths() -> list[Path]:
    """Per-user config locations searched for ``pip``-installed users.

    These give users a stable place to drop credentials when there is no
    project-local ``.quantum_hw.yaml`` (e.g. running from an arbitrary
    directory after ``pip install``).
    """
    home = Path.home()
    return [
        home / _CONFIG_FILENAME,                       # ~/.quantum_hw.yaml
        home / ".config" / "fieldqkit" / "credentials.yaml",
    ]


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

_cached_config: Optional[Dict[str, Any]] = None


def _iter_config_candidates() -> list[Path]:
    """Return de-duplicated config file candidates ordered by priority."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    # 1. Optional explicit override for power users.
    env_path = os.getenv("QUANTUM_HW_CONFIG")
    if env_path:
        explicit_path = Path(env_path).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = Path.cwd() / explicit_path
        _add(explicit_path)

    # 2. Current working directory and its ancestors.
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        _add(directory / _CONFIG_FILENAME)

    # 3. Per-user config locations (recommended for pip-installed users).
    for path in _user_config_paths():
        _add(path)

    # 4. Package installation directory and its ancestors (source/editable installs).
    module_file = Path(__file__).resolve()
    for start in (module_file.parent, module_file.parents[1]):
        for directory in (start, *start.parents):
            _add(directory / _CONFIG_FILENAME)

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
    "tianyan":  "TianYan (天衍) – https://qc.zdxlz.com/",
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
        f"  1. Set environment variable: export {env_var}=<your-token>\n"
        f"  2. Create a config file and fill in your token:\n"
        f"       run `fieldqkit-config-init`  (writes ~/.quantum_hw.yaml)\n"
        f"       then edit it:\n"
        f"         credentials:\n"
        f"           {section}:\n"
        f"             {key}: <your-token>\n"
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


# ---------------------------------------------------------------------------
# Config scaffolding (for pip-installed users without the source-tree template)
# ---------------------------------------------------------------------------

CONFIG_TEMPLATE = """\
# fieldqkit 凭证配置文件 / credentials config
#
# 查找优先级 / lookup order:
#   $QUANTUM_HW_CONFIG -> ./.quantum_hw.yaml (and ancestors)
#   -> ~/.quantum_hw.yaml -> ~/.config/fieldqkit/credentials.yaml -> 环境变量
#
# 也可以改用环境变量 / you may instead use environment variables, e.g.
#   export QUAFU_API_TOKEN=<your-token>
#
# ⚠️  请勿将含真实 token 的文件提交到 Git！/ never commit real tokens.

credentials:
  # 夸父量子云 — https://quafu-sqc.baqis.ac.cn/  (免费、推荐入门)
  quafu:
    api_token: ""

  # 天衍量子云 — https://qc.zdxlz.com/
  tianyan:
    api_token: ""

  # 国盾量子云 — https://quantumctek-cloud.com/
  guodun:
    api_token: ""

  # 腾讯量子云 — https://quantum.tencent.com/cloud/
  tencent:
    api_token: ""

  # 本源量子云 — https://qcloud.originqc.com.cn/
  origin:
    api_token: ""

  # FieldQuantum 云端模拟器 — https://fieldquantum.tech/  (token 形如 fq_<32hex>)
  fieldquantum:
    api_token: ""
"""


def default_user_config_path() -> Path:
    """Return the default per-user config path (``~/.quantum_hw.yaml``)."""
    return Path.home() / _CONFIG_FILENAME


def write_example_config(path: Optional[Path | str] = None, *, force: bool = False) -> Path:
    """Write a credentials template file and return the path written.

    This is the ``pip``-friendly replacement for copying
    ``.quantum_hw.example.yaml`` from the source tree: it materialises the same
    template wherever the user wants it.

    Args:
        path (*Optional[Path | str]*): Target path. Defaults to
            ``~/.quantum_hw.yaml`` (see :func:`default_user_config_path`).
        force (*bool*): Overwrite an existing file when ``True``. Defaults to
            ``False``.

    Returns:
        The ``Path`` that was written.

    Raises:
        FileExistsError: If the target exists and ``force`` is ``False``.
    """
    target = Path(path).expanduser() if path is not None else default_user_config_path()
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass force=True (or --force) to overwrite"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return target


# Public alias — friendlier to call from a notebook/REPL.
init_config = write_example_config


def _config_init_cli(argv: Optional[list[str]] = None) -> int:
    """Console entry point for ``fieldqkit-config-init``.

    Writes a credentials template (default ``~/.quantum_hw.yaml``) for the user
    to fill in. Returns a process exit code.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="fieldqkit-config-init",
        description="Write a fieldqkit credentials template (.quantum_hw.yaml) to fill in.",
    )
    parser.add_argument(
        "-p", "--path", default=None,
        help="target path (default: ~/.quantum_hw.yaml)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="overwrite the file if it already exists",
    )
    args = parser.parse_args(argv)

    try:
        target = write_example_config(args.path, force=args.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote credentials template to: {target}")
    print("Next: edit it and fill in your platform API token(s), then run your program.")
    print("(Alternatively, set an env var such as QUAFU_API_TOKEN=<your-token>.)")
    return 0
