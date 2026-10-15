"""
scrapers/opportunity_normalizer.py — Adapts the dict shapes produced by
hackathons.py / internships.py / jobs.py / events.py into the canonical
row shape db/opportunity_operations.upsert_opportunity() expects.

Your four scraper modules each return dicts with a *mostly* shared shape
but domain-specific field names:
  hackathons.py:  title,url,image,prize,date,start_date,end_date,tags,
                  participants,source,type,mode,location,organizer,
                  status,description
  internships.py: title,company,url,location,stipend,duration,tags,
                  source,type,mode,image,date
  jobs.py:        title,company,url,location,salary,experience,tags,
                  source,type,mode,image,date
  events.py:      title,organizer,url,date,location,prize|fee,tags,
                  source,type,mode,image

This module is the ONE place that reconciles those differences — same
role github_client/parser.py plays for roadmap react-flow/frontmatter,
and the same role I originally gave scrapers/normalizer.py before I'd
seen your actual scraper output.

Design choices worth knowing about:
  - external_id: none of these scrapers expose a stable source-native ID
    (Devpost's numeric "id" gets discarded before the dict leaves
    hackathons.py). We derive a stable id by hashing the URL — the same
    field your own aggregators already dedupe on, so this is at least
    as reliable as your existing dedup logic.
  - compensation: prize (hackathons/techfests), stipend (internships),
    salary (jobs), fee (workshops) all collapse into one column — they're
    mutually exclusive per type, so no information is lost by unifying
    the column name.
  - duration_info: duration (internships) or experience (jobs) —
    same reasoning.
  - deadline: only hackathons/events carry a real end date. Internships
    and jobs report "Open"/"Rolling"/relative-time strings instead — we
    do NOT try to force those into a parsed date (that would produce
    garbage). deadline stays NULL for those; date_label preserves the
    original text for display.
"""
import hashlib
import logging
import re
from datetime import datetime
from typing import Optional

from dateutil import parser as dateutil_parser

log = logging.getLogger(__name__)

VALID_OPP_TYPES = {"hackathon", "internship", "job", "techfest", "workshop"}

_RANGE_SPLIT_RE = re.compile(r"\s+(?:-|–|—|to|through)\s+", re.IGNORECASE)

# Vague cadence/status words carry no extractable date at all — checked
# as a substring match since fallback_data.py uses phrases like
# "Self-paced", "Year Round", "Quarterly" that exact-match won't catch.
_VAGUE_LABEL_RE = re.compile(
    r"\b(tba|ongoing|open|rolling|just now|recent|today|self-paced|"
    r"year round|quarterly|monthly)\b",
    re.IGNORECASE,
)

# "3 days ago", "12 days left", "2 weeks ago" — dateutil's fuzzy parser
# happily extracts the bare number and misreads it as a day-of-month,
# producing a plausible but FABRICATED date (verified: "3 days ago"
# silently parsed as day=3 of the current month). Reject outright.
_RELATIVE_TIME_RE = re.compile(r"\b(ago|left|remaining)\b", re.IGNORECASE)

# "Apr 4-6, 2026" or "Apr 4 - 6, 2026" — same month, compact or spaced
# hyphen. Mirrors the same_m pattern hackathons.py already uses for MLH.
_SAME_MONTH_RANGE_RE = re.compile(
    r"([A-Za-z]+)\s+(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*,?\s*(\d{4})"
)
# "Apr 4 - May 1, 2026" — cross-month range. Mirrors hackathons.py's
# cross_m pattern for MLH.
_CROSS_MONTH_RANGE_RE = re.compile(
    r"([A-Za-z]+)\s+(\d{1,2})\s*[-–—]\s*([A-Za-z]+)\s+(\d{1,2})\s*,?\s*(\d{4})"
)


def _derive_external_id(url: str, title: str) -> str:
    """Stable id from URL (falls back to title if URL is missing/placeholder —
    a few event-scraper paths use the listing page URL as a stand-in)."""
    basis = (url or "").strip().rstrip("/") or title.strip().lower()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:20]


def _try_ymd(month_str: str, day: str, year: str) -> Optional[datetime]:
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{month_str} {day} {year}", fmt)
        except ValueError:
            continue
    return None


def _parse_single(text: str) -> Optional[datetime]:
    """Only for genuinely single (non-range) date strings — do not call
    this on compound ranges, it will confidently produce wrong years/days
    for shapes it wasn't built to handle (verified against real scraper
    output: "March 28 - 29, 2026" -> wrongly parsed as 2029-03-28)."""
    try:
        return dateutil_parser.parse(text, fuzzy=True, default=datetime.utcnow())
    except (ValueError, OverflowError):
        return None


def normalize_deadline(raw: Optional[str]) -> Optional[str]:
    """Best-effort parse of a date/date-range string into 'YYYY-MM-DD'
    (the END of a range, i.e. the actual deadline). Returns None rather
    than guessing whenever the string is vague, relative, or ambiguous —
    a wrong deadline is worse than a missing one."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if _VAGUE_LABEL_RE.search(raw) or _RELATIVE_TIME_RE.search(raw):
        return None

    # Cross-month compact range: "Apr 4 - May 1, 2026" — check before
    # same-month, since same-month's pattern would otherwise partially
    # match the first half and miss the actual end date.
    m = _CROSS_MONTH_RANGE_RE.search(raw)
    if m:
        end = _try_ymd(m.group(3), m.group(4), m.group(5))
        if end:
            return end.date().isoformat()

    # Same-month compact range: "Apr 4-6, 2026" / "Apr 4 - 6, 2026"
    m = _SAME_MONTH_RANGE_RE.search(raw)
    if m:
        end = _try_ymd(m.group(1), m.group(3), m.group(4))
        if end:
            return end.date().isoformat()

    # Whitespace-delimited full-date range: "Jul 13, 2026 - Jul 21, 2026"
    # or "2026-07-09 to 2026-07-21" — each side is already a complete,
    # unambiguous date, so splitting and parsing the last one is safe.
    segments = _RANGE_SPLIT_RE.split(raw)
    if len(segments) > 1:
        dt = _parse_single(segments[-1].strip())
        if dt:
            return dt.date().isoformat()
        return None  # ambiguous range we don't have a specific pattern for — don't guess

    # No range detected — safe to fuzzy-parse as a single date.
    dt = _parse_single(raw)
    if dt is None:
        log.debug("Could not parse deadline: %r", raw)
        return None
    return dt.date().isoformat()


def _is_remote(mode: str, location: str) -> bool:
    text = f"{mode} {location}".lower()
    return "remote" in text or "online" in text or "work from home" in text


def normalize(item: dict) -> Optional[dict]:
    """
    Converts one raw dict from any of the four scraper modules into the
    canonical row shape. Returns None (and logs) for malformed input —
    callers should filter None results before persisting.
    """
    title = (item.get("title") or "").strip()
    url = (item.get("url") or "").strip()
    source = (item.get("source") or "unknown").strip()
    opp_type = (item.get("type") or "").strip().lower()

    if not title or not url:
        log.warning("[%s] dropping listing — missing title/url: %r", source, item)
        return None
    if opp_type not in VALID_OPP_TYPES:
        log.warning("[%s] dropping listing — unrecognized type %r for %r",
                    source, opp_type, title)
        return None

    org = (item.get("organizer") or item.get("company") or "").strip()
    location = (item.get("location") or "").strip()
    mode = (item.get("mode") or "").strip()

    # prize (hackathon/techfest) | stipend (internship) | salary (job) | fee (workshop)
    compensation = (
        item.get("prize") or item.get("stipend") or
        item.get("salary") or item.get("fee") or ""
    ).strip()

    duration_info = (item.get("duration") or item.get("experience") or "").strip()

    # Prefer an explicit end_date (hackathons carry this); fall back to
    # the general "date" field, which for events/hackathons is often a
    # real date and for internships/jobs is a status word (correctly
    # discarded by normalize_deadline's _NON_DATE_LABELS check).
    deadline_raw = item.get("end_date") or item.get("date") or ""
    deadline = normalize_deadline(deadline_raw)
    date_label = (item.get("date") or "").strip()

    return {
        "source": source,
        "external_id": _derive_external_id(url, title),
        "title": title[:300],
        "org": org[:200],
        "opp_type": opp_type,
        "url": url,
        "location": location[:200],
        "work_mode": mode[:50],
        "is_remote": _is_remote(mode, location),
        "date_label": date_label[:100],
        "deadline": deadline,
        "compensation": compensation[:200],
        "duration_info": duration_info[:100],
        "description": (item.get("description") or "").strip()[:2000],
        "thumbnail_url": (item.get("image") or "").strip(),
        "skills_raw": [t for t in (item.get("tags") or []) if isinstance(t, str) and t.strip()],
        "is_fallback": bool(item.get("_is_fallback", False)),
    }


def normalize_many(items: list) -> list:
    out = []
    for item in items:
        n = normalize(item)
        if n:
            out.append(n)
    return out