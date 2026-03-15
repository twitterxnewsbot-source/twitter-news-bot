"""
storage.py
----------
Tracks article URLs that have already been tweeted.

Improvements over v1
---------------------
* Atomic writes  – data is written to a temp file then renamed into place,
  so a crash mid-write never corrupts the store.
* In-memory set cache  – `is_posted()` no longer hits the disk on every
  call; the cache is loaded once at startup and kept in sync.
* Corrupt-file recovery  – if the JSON is unreadable the bad file is
  renamed to *.bak and a fresh store is started, avoiding a silent empty
  return that would cause duplicate tweets after a restart.
* Thread-safety  – a module-level lock guards the cache so the bot is safe
  if it is ever extended to post from multiple threads.
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

_DEFAULT_PATH = Path(__file__).parent / "posted_links.json"
STORAGE_FILE  = Path(os.getenv("STORAGE_FILE", str(_DEFAULT_PATH)))

# Rolling window: keep the N most-recent URLs.
MAX_STORED_LINKS = 2000

# ── module-level cache ────────────────────────────────────────────────────────

_lock: threading.Lock        = threading.Lock()
_cache: "set[str] | None"    = None   # None means "not yet loaded"
_ordered: "list[str]"        = []     # preserves insertion order for trimming


# ── internal helpers ──────────────────────────────────────────────────────────

def _load_from_disk() -> "list[str]":
    """
    Read the JSON store from disk.
    On corruption, backs up the bad file and returns [] so the bot can
    continue – but logs loudly so the operator notices.
    """
    if not STORAGE_FILE.exists():
        return []

    try:
        with STORAGE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        logger.warning("Storage file had unexpected format; starting fresh.")
        return []

    except (json.JSONDecodeError, OSError) as exc:
        bak = STORAGE_FILE.with_suffix(".json.bak")
        logger.error(
            "Corrupt storage file (%s). Backing up to %s and starting fresh.",
            exc,
            bak,
        )
        try:
            STORAGE_FILE.rename(bak)
        except OSError:
            pass
        return []


def _ensure_loaded() -> None:
    """Populate the in-memory cache from disk (idempotent, called under lock)."""
    global _cache, _ordered
    if _cache is None:
        _ordered = _load_from_disk()
        _cache   = set(_ordered)


def _flush() -> None:
    """
    Atomically write the current in-memory list to disk.
    Writes to a sibling temp file then renames, so a crash during the write
    can never leave the store in a partial/corrupt state.
    """
    try:
        STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir    = STORAGE_FILE.parent,
            prefix = ".posted_links_",
            suffix = ".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(_ordered, fh, indent=2)
            os.replace(tmp_path, STORAGE_FILE)   # atomic on POSIX & Windows
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.error("Could not write storage file %s: %s", STORAGE_FILE, exc)


# ── public API ────────────────────────────────────────────────────────────────

def is_posted(url: str) -> bool:
    """Return True if *url* has already been tweeted.  O(1) set lookup."""
    with _lock:
        _ensure_loaded()
        return url in _cache  # type: ignore[operator]


def mark_posted(url: str) -> None:
    """Record *url* as tweeted and persist to disk atomically."""
    with _lock:
        _ensure_loaded()
        if url in _cache:  # type: ignore[operator]
            return

        _ordered.append(url)
        _cache.add(url)  # type: ignore[union-attr]

        # Trim oldest entries to stay within the rolling window.
        if len(_ordered) > MAX_STORED_LINKS:
            excess  = len(_ordered) - MAX_STORED_LINKS
            removed = _ordered[:excess]
            del _ordered[:excess]
            _cache -= set(removed)  # type: ignore[operator]

        _flush()
        logger.debug("Marked as posted: %s", url)


def count_posted() -> int:
    """Return how many unique links are currently stored."""
    with _lock:
        _ensure_loaded()
        return len(_ordered)
