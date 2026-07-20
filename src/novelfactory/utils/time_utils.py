"""ISO 8601 timestamp helpers.

Migrated from DeerFlow utils/time.py.

All timestamp generation should funnel through :func:`now_iso` so the
wire format stays consistent across endpoints.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

__all__ = ["coerce_iso", "is_lease_expired", "now_iso"]


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Example: "2026-04-27T03:19:46.511479+00:00".
    """
    return datetime.now(UTC).isoformat()


def is_lease_expired(lease_expires_at: str | None, *, grace_seconds: int) -> bool:
    """Return True when lease_expires_at has elapsed past grace."""
    if lease_expires_at is None:
        return True
    try:
        dt = datetime.fromisoformat(lease_expires_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return True
    return dt < datetime.now(UTC) - timedelta(seconds=grace_seconds)


_UNIX_TIMESTAMP_PATTERN = re.compile(r"^\d{10}(?:\.\d+)?$")


def coerce_iso(value: object) -> str:
    """Best-effort coerce a stored timestamp to an ISO 8601 string."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return value.isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            return str(value)
    if isinstance(value, str):
        if _UNIX_TIMESTAMP_PATTERN.match(value):
            try:
                return datetime.fromtimestamp(float(value), UTC).isoformat()
            except (ValueError, OverflowError, OSError):
                return value
        return value
    return str(value)