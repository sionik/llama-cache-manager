"""How the tool reads and writes ages.

The cutoff of ``prune --until`` and the age column of ``ls`` are two sides of
one idea, so they share a module. Ages come from the modification time of the
blob, which is when the file was downloaded, and not from the access time,
which mount options such as ``relatime`` make unreliable.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

SECONDS_PER = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 2592000.0,
    "year": 31536000.0,
}

_UNIT_NAMES = {
    "s": "second",
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
    "m": "minute",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "d": "day",
    "day": "day",
    "days": "day",
    "w": "week",
    "week": "week",
    "weeks": "week",
    "mo": "month",
    "mon": "month",
    "month": "month",
    "months": "month",
    "y": "year",
    "yr": "year",
    "yrs": "year",
    "year": "year",
    "years": "year",
}

_RELATIVE = re.compile(r"^(?P<count>\d+)\s*(?P<unit>[a-z]+)(\s+ago)?$", re.IGNORECASE)

_ABSOLUTE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
)


class CutoffError(ValueError):
    """A cutoff the tool cannot read."""


def parse_cutoff(spec: str, now: datetime) -> datetime:
    """Read ``--until SPEC`` as a point in time.

    Relative forms count back from ``now``: ``7d``, ``7 days``, ``2 weeks
    ago``, ``12h``, ``6mo``, ``1y``. Absolute forms are read as local time:
    ``2026-07-01`` or ``2026-07-01 18:30``.

    Raises:
        CutoffError: the text is neither form, or names an unknown unit.
    """
    text = " ".join(spec.split())
    if not text:
        raise CutoffError("empty cutoff")

    relative = _RELATIVE.match(text)
    if relative:
        unit = _UNIT_NAMES.get(relative.group("unit").lower())
        if unit is None:
            raise CutoffError(
                f"unknown unit {relative.group('unit')!r} in cutoff {spec!r}; "
                "use seconds, minutes, hours, days, weeks, months or years"
            )
        count = int(relative.group("count"))
        try:
            return now - timedelta(seconds=count * SECONDS_PER[unit])
        except (OverflowError, ValueError, OSError) as error:
            raise CutoffError(f"cutoff {spec!r} reaches beyond the calendar") from error

    for fmt in _ABSOLUTE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise CutoffError(
        f"cannot read cutoff {spec!r}; expected a span such as '7d' or '2 weeks', "
        "or a date such as '2026-07-01'"
    )


def cutoff(spec: str, now: datetime) -> float:
    """Read ``--until SPEC`` as a Unix timestamp.

    Raises:
        CutoffError: the text cannot be read, or names a moment the platform
            cannot express as a timestamp.
    """
    moment = parse_cutoff(spec, now)
    try:
        return moment.timestamp()
    except (OverflowError, ValueError, OSError) as error:
        raise CutoffError(f"cutoff {spec!r} reaches beyond the calendar") from error


def describe(seconds: float) -> str:
    """A short age, as ``ls`` prints it in the revision line."""
    if seconds < 0:
        return "just now"
    # Weeks are left out on purpose. "27 days ago" says more about a download
    # than "3 weeks ago", and the listing has room for the longer word.
    for unit in ("year", "month", "day", "hour", "minute"):
        size = SECONDS_PER[unit]
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    return "just now"
