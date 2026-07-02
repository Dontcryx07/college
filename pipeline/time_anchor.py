"""Deterministic time anchor derived from the candidate pool.

Recency-based features (behavioral recency, role recency discounts) require
a "today" reference date. Using the wall clock would produce silently
different rankings on each re-run, violating the reproducibility requirement
for Stage 3 validation. Instead we derive the anchor from the data itself:
the latest ``last_active_date`` across all candidates.

The fallback date is the observed maximum during exploratory data analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from . import data_access


# Maximum last_active_date observed during EDA of the 100k-pool dataset.
DEFAULT_FALLBACK_ANCHOR_DATE: str = "2026-05-27"


@dataclass(frozen=True)
class TimeAnchor:
    """A deterministic 'today' derived entirely from the data."""

    date: str
    month_index: int


def compute_time_anchor(
    candidates: Iterable[dict],
    fallback: str = DEFAULT_FALLBACK_ANCHOR_DATE,
) -> TimeAnchor:
    """Determine the latest active date across the candidate pool.

    Iterates all candidates to find the maximum ``last_active_date`` from
    their Redrob signals, then returns a ``TimeAnchor`` containing both the
    ISO date string and its month index for convenient arithmetic.

    Args:
        candidates: Iterable of candidate dictionaries.
        fallback: Date string used when no candidate has a parseable
            ``last_active_date``.

    Returns:
        A ``TimeAnchor`` representing the data-derived "today".
    """
    latest_date: Optional[str] = None
    for candidate in candidates:
        active = data_access.get_redrob_signals(candidate).get("last_active_date")
        if isinstance(active, str) and data_access.parse_iso_date(active) is not None:
            if latest_date is None or active > latest_date:
                latest_date = active

    chosen = latest_date or fallback
    month = data_access.month_index_from_date(chosen)
    fallback_month = data_access.month_index_from_date(fallback) or 0
    return TimeAnchor(date=chosen, month_index=month or fallback_month)
