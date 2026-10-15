"""
sync/opportunity_engine.py — Orchestrates Shashank's existing scraper
modules (scrapers/hackathons.py, internships.py, jobs.py, events.py),
normalizes their output, and persists it to SQLite.

DELIBERATELY a separate scheduler from sync/engine.py (roadmap sync),
not merged:
  - Cadence differs: roadmaps are stable for weeks, opportunities have
    deadlines and go stale in hours (Config.OPPORTUNITY_SYNC_INTERVAL_HOURS).
  - Fault isolation: if this scheduler's thread dies, roadmap sync is
    unaffected, and vice versa.

Layering, outside-in:
  scrapers/{hackathons,internships,jobs,events}.py  — live scraping (yours, untouched)
  scrapers/cache_manager.py                          — 1h file cache in front of live scraping (yours, untouched)
  scrapers/fallback_data.py                          — curated static data if a whole source-family is down (yours, untouched)
  scrapers/opportunity_normalizer.py                 — reconciles the four dict shapes into one row shape (new)
  db/opportunity_operations.py                        — SQLite persistence + skill-matched queries (new)
  sync/opportunity_engine.py (this file)              — orchestration + scheduling (new)

Each "get_all_X" aggregator already dedupes internally by url/title
(that logic is untouched); this module dedupes ACROSS sync runs via the
DB's UNIQUE(source, external_id) upsert, and across the whole DB via
that same constraint if the same listing shows up through two routes
(e.g. Internshala's job scraper and the general jobs aggregator).
"""
import logging
import threading
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from careercompass.config import Config
from careercompass.db import init_db
from careercompass.db import opportunity_operations as ops
from careercompass.scrapers import cache_manager
from careercompass.scrapers import circuit_breaker
from careercompass.scrapers.hackathons import get_all_hackathons
from careercompass.scrapers.internships import get_all_internships
from careercompass.scrapers.jobs import get_all_jobs
from careercompass.scrapers.events import get_all_techfests, get_all_workshops
from careercompass.scrapers.fallback_data import (
    FALLBACK_HACKATHONS, FALLBACK_INTERNSHIPS, FALLBACK_JOBS,
    FALLBACK_TECHFESTS, FALLBACK_WORKSHOPS,
)
from careercompass.scrapers.opportunity_normalizer import normalize_many

log = logging.getLogger(__name__)

# ── Source registry ─────────────────────────────────────────────────────
# (cache_key, live_scrape_fn, fallback_list) — one entry per domain.
# Adding a new domain later = one new scraper module + one line here.
_SOURCES = [
    ("opp_hackathons", get_all_hackathons, FALLBACK_HACKATHONS),
    ("opp_internships", get_all_internships, FALLBACK_INTERNSHIPS),
    ("opp_jobs", get_all_jobs, FALLBACK_JOBS),
    ("opp_techfests", get_all_techfests, FALLBACK_TECHFESTS),
    ("opp_workshops", get_all_workshops, FALLBACK_WORKSHOPS),
]

# ── Shared state (mirrors sync/engine.py's _state dict shape) ──────────────
_lock = threading.Lock()
_state: dict = {
    "running": False, "trigger": None, "started_at": None,
    "current_source": None, "sources_done": 0, "sources_total": 0,
    "last_status": None, "last_completed": None, "error": None,
    "last_fallback_sources": [],   # which domains fell back to curated data last run
}
_run_lock = threading.Lock()
_timer: Optional[threading.Timer] = None


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


# ── Public API ────────────────────────────────────────────────────────────

def start_scheduler():
    init_db()
    log.info("Opportunity sync scheduler starting — %d source domains", len(_SOURCES))
    _schedule(3, "startup")   # small offset from roadmap sync's 2s so they don't collide


def trigger_manual() -> bool:
    if _state["running"]:
        return False
    _schedule(0, "manual")
    return True


def stop_scheduler():
    global _timer
    if _timer:
        _timer.cancel()
        _timer = None


def _schedule(delay: float, trigger: str):
    t = threading.Timer(delay, _run, kwargs={"trigger": trigger})
    t.daemon = True
    t.start()


def _schedule_next():
    global _timer
    if _timer:
        _timer.cancel()
    secs = Config.OPPORTUNITY_SYNC_INTERVAL_HOURS * 3600
    _timer = threading.Timer(secs, _run, kwargs={"trigger": "scheduled"})
    _timer.daemon = True
    _timer.start()
    log.info("Next opportunity sync in %.1f h", Config.OPPORTUNITY_SYNC_INTERVAL_HOURS)


# ── Per-source fetch with cache + fallback ──────────────────────────────────

def _fetch_domain(cache_key: str, scrape_fn: Callable, fallback_list: List[dict]) -> Tuple[List[dict], bool]:
    """
    Returns (items, used_fallback).
    1. Try the 1h file cache first (protects against back-to-back manual
       triggers hammering live sites within the same hour).
    2. On cache miss, scrape live. Your aggregator functions already
       catch and log per-scraper-function exceptions internally, so a
       single broken source (e.g. MLH HTML structure changes) degrades
       that one source to fewer results rather than raising.
    3. If the live scrape comes back completely empty (every source in
       that domain is down/blocked), fall back to the curated static
       list from fallback_data.py rather than showing the user nothing.
    """
    cached = cache_manager.get(cache_key)
    if cached is not None:
        log.info("[%s] served from cache (age: %s)", cache_key, cache_manager.get_age(cache_key))
        return cached, False

    try:
        items = scrape_fn()
    except Exception as e:
        log.error("[%s] live scrape raised: %s", cache_key, e, exc_info=True)
        items = []

    if items:
        cache_manager.set(cache_key, items)
        return items, False

    log.warning("[%s] live scrape returned nothing — using curated fallback data", cache_key)
    fallback_items = [dict(item, _is_fallback=True) for item in fallback_list]
    return fallback_items, True


# ── Core sync ─────────────────────────────────────────────────────────────

def _run(trigger: str = "scheduled"):
    if not _run_lock.acquire(blocking=False):
        log.warning("Opportunity sync already running — skipping (%s)", trigger)
        return

    circuit_breaker.reset()   # fresh chance for every host at the start of each run —
                               # a host down 6h ago may well be back by now
    _set(running=True, trigger=trigger, started_at=_now(),
         current_source=None, sources_done=0, sources_total=len(_SOURCES),
         error=None, last_fallback_sources=[])

    sync_id = ops.start_opp_sync(trigger=trigger, sources_total=len(_SOURCES))
    hist_stats = {"status": "running", "total_fetched": 0, "total_upserted": 0,
                  "pruned_expired": 0, "pruned_stale": 0, "fallback_sources": [], "error": None}

    total_fetched, total_upserted, fallback_domains, errors = 0, 0, [], []

    try:
        for i, (cache_key, scrape_fn, fallback_list) in enumerate(_SOURCES):
            domain_label = cache_key.replace("opp_", "")
            _set(current_source=domain_label, sources_done=i)

            try:
                items, used_fallback = _fetch_domain(cache_key, scrape_fn, fallback_list)
                total_fetched += len(items)
                if used_fallback:
                    fallback_domains.append(domain_label)

                normalized = normalize_many(items)
                for opp in normalized:
                    ops.upsert_opportunity(opp)
                    total_upserted += 1

            except Exception as e:
                log.error("Domain '%s' failed entirely: %s", domain_label, e, exc_info=True)
                errors.append(f"{domain_label}: {e}")

        pruned_expired = ops.prune_expired(Config.OPPORTUNITY_RETENTION_DAYS)
        pruned_stale = ops.prune_stale_scrapes(Config.OPPORTUNITY_STALE_SCRAPE_DAYS)
        log.info(
            "Opportunity sync: fetched=%d upserted=%d pruned_expired=%d "
            "pruned_stale=%d fallback_domains=%s errors=%d",
            total_fetched, total_upserted, pruned_expired, pruned_stale,
            fallback_domains, len(errors),
        )

        status = "success" if not errors else ("partial" if total_upserted > 0 else "failed")
        _set(last_status=status, error="; ".join(errors) if errors else None,
             last_fallback_sources=fallback_domains)

        hist_stats.update(
            status=status, total_fetched=total_fetched, total_upserted=total_upserted,
            pruned_expired=pruned_expired, pruned_stale=pruned_stale,
            fallback_sources=fallback_domains,
            error="; ".join(errors) if errors else None,
        )

    except Exception as e:
        log.exception("Opportunity sync crashed: %s", e)
        _set(last_status="failed", error=str(e))
        hist_stats.update(status="failed", total_fetched=total_fetched,
                           total_upserted=total_upserted, error=str(e))
    finally:
        ops.finish_opp_sync(sync_id, hist_stats)
        _set(running=False, current_source=None, last_completed=_now())
        _run_lock.release()
        _schedule_next()


def _now() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")