"""Shared helpers for calibration cache files."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def cache_file(cache_dir: Path, *, stem: str, chip_name: Optional[str]) -> Path:
	"""Build cache file path from cache directory and chip name."""
	name = chip_name if chip_name is not None else "unknown"
	return cache_dir / f"{stem}_{name}.json"


def load_timestamped_payload(
	path: Path,
	*,
	payload_key: str,
) -> Tuple[Dict[str, str], Dict[str, object]]:
	"""Load timestamped cache payload safely from disk."""
	if not path.exists():
		return {}, {}
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
		if not isinstance(data, dict):
			return {}, {}
		timestamps = data.get("timestamps", {})
		payload = data.get(payload_key, {})
		if not isinstance(timestamps, dict) or not isinstance(payload, dict):
			return {}, {}
		return timestamps, payload
	except Exception:
		return {}, {}


def save_timestamped_payload(
	path: Path,
	*,
	payload_key: str,
	timestamps: Dict[str, str],
	payload: Dict[str, object],
) -> None:
	"""Persist timestamped cache payload to disk."""
	path.write_text(
		json.dumps({"timestamps": timestamps, payload_key: payload}, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def cache_is_fresh(ts_str: Optional[str], *, now: datetime, ttl_hours: int = 12) -> bool:
	"""Return True when timestamp exists and is within TTL."""
	if ts_str is None:
		return False
	try:
		ts = datetime.fromisoformat(ts_str)
	except Exception:
		return False
	return now - ts <= timedelta(hours=ttl_hours)
