"""
scrapers/circuit_breaker.py — Per-host fail-fast breaker for every scraper's
requests.Session.

THE PROBLEM (seen directly in production logs): each scraper's request loop
retries serially across many URLs/pages/categories, each with a 12-15s
connect timeout. When a host is genuinely unreachable — network block,
firewall, DNS issue, the host itself down — that turns into MINUTES of
wall-clock time waiting out timeouts on requests that were never going to
succeed. One observed sync run took 14 minutes for exactly this reason
(Devpost, Internshala, api.unstop.com, and Wellfound all timed out on
every attempt, but each one was retried across many URLs before giving up).

THE FIX: CircuitBreakerSession is a drop-in replacement for requests.Session.
Swap `requests.Session()` for `CircuitBreakerSession()` at the small number
of places each scraper module creates a session — nothing else needs to
change. Every `.get()`/`.post()` call automatically routes through
`request()` below (that's how requests.Session works internally), which:

  1. Before each request, checks whether this host has already failed
     FAILURE_THRESHOLD times this sync run. If so, raises immediately
     instead of waiting out another full connect timeout.
  2. On a connection-level failure (timeout, DNS failure, connection
     refused), records it against the host.
  3. On ANY successful connection — including a 403/404, which still
     proves the host is reachable — resets that host's failure count.
     Only connection-level failures trip the breaker; HTTP error status
     codes do not, since those mean the host answered.

Existing `except Exception as e: logger.error(...); break` blocks in every
scraper's loop keep working completely unchanged — they still catch the
same exception types, they just fail in milliseconds instead of 12-15
seconds once a host is known to be down this run.

State is in-memory, per-process, and reset once per sync run (see
sync/opportunity_engine.py calling reset() at the top of _run()) — a host
that's down right now might be back in 6 hours when the next scheduled
sync happens, so failures must never persist across runs.
"""
import logging
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

FAILURE_THRESHOLD = 2   # consecutive connection-level failures before tripping
_state: dict = {}       # host -> {"failures": int, "tripped": bool}


def reset():
    """Call once at the start of each sync run so a host that failed
    6 hours ago gets a fresh chance this run."""
    _state.clear()


def _host(url: str) -> str:
    return urlparse(url).netloc


def is_open(url: str) -> bool:
    """True if this host's circuit is open (i.e. further requests to it
    should be skipped for the rest of this run)."""
    return _state.get(_host(url), {}).get("tripped", False)


def record_failure(url: str):
    h = _host(url)
    entry = _state.setdefault(h, {"failures": 0, "tripped": False})
    entry["failures"] += 1
    if entry["failures"] >= FAILURE_THRESHOLD and not entry["tripped"]:
        entry["tripped"] = True
        log.warning(
            "[circuit-breaker] %s unreachable after %d consecutive connection "
            "failures — skipping remaining requests to this host for the "
            "rest of this sync run",
            h, entry["failures"],
        )


def record_success(url: str):
    h = _host(url)
    if h in _state:
        _state[h] = {"failures": 0, "tripped": False}


class CircuitBreakerSession(requests.Session):
    """Drop-in requests.Session replacement — see module docstring.
    Use exactly like requests.Session(); every .get()/.post()/etc call
    is automatically guarded, since they all route through .request()."""

    def request(self, method, url, *args, **kwargs):
        if is_open(url):
            raise requests.exceptions.ConnectionError(
                f"[circuit-breaker open] skipping {url} — {_host(url)} "
                f"already failed {FAILURE_THRESHOLD}+ times this sync run"
            )
        try:
            resp = super().request(method, url, *args, **kwargs)
            record_success(url)
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            record_failure(url)
            raise