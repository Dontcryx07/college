"""Candidate data loading, safe accessor functions, and date arithmetic helpers.

This module handles all raw data I/O and dictionary traversal so that
downstream scoring modules work exclusively with typed return values
and never touch raw candidate dictionaries except through these accessors.

Design decisions:
  - Gzip is handled transparently so callers never think about compression.
  - Malformed JSON lines are silently skipped (a single bad record out of
    100k should not abort the entire pipeline).
  - Date parsing returns None for missing/malformed values rather than
    raising, because this dataset is synthetic and upstream callers all
    handle None gracefully.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Optional


# ============================================================================
# Date helpers — year/month arithmetic without datetime
# ============================================================================

def parse_iso_date(value: Optional[str]) -> Optional[tuple[int, int, int]]:
    """Parse a ``YYYY-MM-DD`` string into a ``(year, month, day)`` tuple.

    Returns ``None`` for missing, empty, or malformed input rather than
    raising, so upstream logic degrades gracefully on bad records.
    """
    if not value or not isinstance(value, str):
        return None
    parts = value.strip()[:10].split("-")
    if len(parts) != 3:
        return None
    try:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    return (year, month, day)


def month_index_from_date(value: Optional[str]) -> Optional[int]:
    """Convert a date string to an absolute month index (``year * 12 + month``).

    Enables cheap month-granularity duration and recency arithmetic without
    the locale/timezone surface area of ``datetime``.
    """
    parsed = parse_iso_date(value)
    if parsed is None:
        return None
    year, month, _ = parsed
    return year * 12 + month


def months_between_dates(earlier: Optional[str], later: Optional[str]) -> Optional[int]:
    """Compute whole months from ``earlier`` to ``later`` (may be negative)."""
    start = month_index_from_date(earlier)
    end = month_index_from_date(later)
    if start is None or end is None:
        return None
    return end - start


# ============================================================================
# File I/O — transparent gzip support
# ============================================================================

def _open_text_maybe_gzipped(path: Path):
    """Open ``path`` for text reading, auto-detecting gzip by extension or magic bytes."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    with open(path, "rb") as probe:
        magic_bytes = probe.read(2)
    if magic_bytes == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidate_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate dicts from a (possibly gzipped) JSONL file.

    Skips blank lines and malformed JSON records rather than crashing.
    """
    resolved_path = Path(path)
    with _open_text_maybe_gzipped(resolved_path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def load_all_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Load every candidate record into a list (~100k entries, ~500 MB)."""
    return list(iter_candidate_records(path))


# ============================================================================
# Safe dictionary accessors — return empty defaults for missing keys
# ============================================================================

def get_profile(candidate: dict) -> dict:
    """Return the candidate's profile dict, or an empty dict if missing."""
    return candidate.get("profile") or {}


def get_redrob_signals(candidate: dict) -> dict:
    """Return the candidate's Redrob behavioral signals, or empty dict."""
    return candidate.get("redrob_signals") or {}


def get_career_history(candidate: dict) -> list[dict]:
    """Return the candidate's career history as a list of role dicts.

    Filters out any non-dict entries that might exist in malformed records.
    """
    raw = candidate.get("career_history") or []
    return [role for role in raw if isinstance(role, dict)]


def get_skills(candidate: dict) -> list[dict]:
    """Return the candidate's skills list, filtering out non-dict entries."""
    raw = candidate.get("skills") or []
    return [skill for skill in raw if isinstance(skill, dict)]


def get_education(candidate: dict) -> list[dict]:
    """Return the candidate's education entries, filtering out non-dict items."""
    raw = candidate.get("education") or []
    return [entry for entry in raw if isinstance(entry, dict)]


def extract_candidate_id(candidate: dict) -> str:
    """Extract and strip the candidate's ID string."""
    return str(candidate.get("candidate_id", "")).strip()


def get_evidence_text_blocks(candidate: dict) -> list[str]:
    """Collect ordered free-text blocks that describe the candidate's actual work.

    Deliberately excludes the ``skills[]`` list — per the JD and EDA, the
    skills array is the adversarial keyword-stuffing surface. Career description,
    headline, and summary are the honest signal.
    """
    profile = get_profile(candidate)
    blocks: list[str] = []

    headline = profile.get("headline")
    if isinstance(headline, str):
        blocks.append(headline)

    summary = profile.get("summary")
    if isinstance(summary, str):
        blocks.append(summary)

    for role in get_career_history(candidate):
        description = role.get("description")
        if isinstance(description, str):
            blocks.append(description)

    return blocks
