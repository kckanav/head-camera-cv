"""Human-readable today's-date helper, e.g. '25th April 2026'."""

from datetime import date
from typing import Optional


def today_pretty(d: Optional[date] = None) -> str:
    d = d or date.today()
    if 11 <= d.day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return f"{d.day}{suffix} {d.strftime('%B %Y')}"
