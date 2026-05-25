"""Query each quantum cloud for its hardware list, then update the hardcoded
``*_HARDWARE_NAMES`` sets in ``src/quantum_hw/api/backend.py``.

Usage:
    python update_hardware_lists.py            # query, print, and update file
    python update_hardware_lists.py --dry-run  # query and print only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

from quantum_hw.api import backend as backend_module
from quantum_hw.api.backend import list_available_hardware


PROVIDERS = ["quafu", "tianyan", "guodun", "tencent", "origin"]

# tianyan + guodun share the CQLib stack; we map provider -> variable name in
# backend.py.  CQLIB_HARDWARE_NAMES is derived (union), not queried.
PROVIDER_TO_VAR = {
    "quafu": "QUAFU_HARDWARE_NAMES",
    "tianyan": "TIANYAN_HARDWARE_NAMES",
    "guodun": "GUODUN_HARDWARE_NAMES",
    "tencent": "TENCENT_HARDWARE_NAMES",
    "origin": "ORIGIN_HARDWARE_NAMES",
}

BACKEND_FILE = Path(__file__).resolve().parent / "src" / "quantum_hw" / "api" / "backend.py"


def query_provider(provider: str) -> Set[str]:
    """Call list_available_hardware and extract the hardware_name set.

    No availability/status filtering — every returned row contributes its name.
    """
    rows = list_available_hardware(provider)
    names: Set[str] = set()
    for row in rows:
        name = str(row.get("hardware_name") or "").strip()
        if name:
            names.add(name)
    return names


def existing_names(var: str) -> Set[str]:
    """Read the current set literal from the loaded backend module."""
    value = getattr(backend_module, var, set())
    return {str(n) for n in value} if isinstance(value, (set, frozenset)) else set()


def format_set_literal(names: Set[str]) -> str:
    """Produce a deterministic ``{"a", "b", ...}`` literal."""
    if not names:
        return "set()"
    items = ", ".join(f'"{n}"' for n in sorted(names))
    return "{" + items + "}"


def update_backend_file(provider_to_names: Dict[str, Set[str]]) -> List[str]:
    """Rewrite the hardcoded sets in backend.py.  Returns list of changed vars."""
    text = BACKEND_FILE.read_text(encoding="utf-8")
    changed: List[str] = []
    for provider, var in PROVIDER_TO_VAR.items():
        if provider not in provider_to_names:
            continue
        merged = existing_names(var) | provider_to_names[provider]
        added = merged - existing_names(var)
        if not added:
            continue
        new_literal = format_set_literal(merged)
        pattern = re.compile(rf"^{var}\s*=\s*\{{[^}}]*\}}", re.MULTILINE)
        replacement = f"{var} = {new_literal}"
        new_text, n = pattern.subn(replacement, text, count=1)
        if n == 0:
            print(f"  WARNING: could not find {var} in backend.py")
            continue
        if new_text != text:
            changed.append(f"{var} (+{len(added)}: {', '.join(sorted(added))})")
            text = new_text
    BACKEND_FILE.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="query only, do not modify backend.py")
    parser.add_argument(
        "--provider",
        action="append",
        choices=PROVIDERS,
        help="restrict to one or more providers (default: all)",
    )
    args = parser.parse_args()

    providers = args.provider or PROVIDERS
    provider_to_names: Dict[str, Set[str]] = {}

    for provider in providers:
        print(f"\n=== {provider} ===")
        try:
            names = query_provider(provider)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        provider_to_names[provider] = names
        if names:
            for n in sorted(names):
                print(f"  - {n}")
        else:
            print("  (no hardware returned)")

    if args.dry_run:
        print("\n--dry-run: backend.py left untouched.")
        return

    if not provider_to_names:
        print("\nNo successful queries; backend.py left untouched.")
        return

    print("\nUpdating backend.py...")
    changed = update_backend_file(provider_to_names)
    if changed:
        print(f"  Rewrote: {', '.join(changed)}")
    else:
        print("  No changes (lists already up to date).")


if __name__ == "__main__":
    main()
