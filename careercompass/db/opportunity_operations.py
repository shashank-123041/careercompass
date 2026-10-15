"""
db/opportunity_operations.py — Every opportunity-related database
operation lives here. No SQL is written anywhere else in this domain.

REVISION NOTE: schema widened from the first draft after integrating
Shashank's existing scrapers/{hackathons,internships,jobs,events}.py.
Their dicts carry richer, source-shaped fields (prize/stipend/salary/fee,
duration/experience, a Remote/On-Site/Hybrid mode string, images, and a
free-text date/status like "Open" or "3 days ago") that a narrower schema
would have silently discarded. See scrapers/opportunity_normalizer.py for
the mapping from their raw dict shape to this canonical row shape.

Same conventions as db/operations.py:
  - atomic ON CONFLICT upserts (avoids the SELECT->INSERT race the
    roadmaps table hit under Flask's debug reloader)
  - dataclass-free dict rows (kept simple — this data is read-mostly
    and consumed directly as JSON by the API layer)
  - the one shared get_conn() connection pool from db/__init__.py
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from careercompass.db import get_conn
from careercompass.taxonomy.matcher import match_skills

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")


# ── upsert ────────────────────────────────────────────────────────────────

def upsert_opportunity(opp: dict) -> int:
    """
    Atomically insert-or-update one opportunity, then replace its
    matched-skill rows. `opp` is the canonical dict produced by
    scrapers/opportunity_normalizer.normalize().
    """
    c = get_conn()
    now = _now()

    c.execute("""
        INSERT INTO opportunities
            (source, external_id, title, org, opp_type, url, location,
             work_mode, is_remote, date_label, deadline, compensation,
             duration_info, description, thumbnail_url, is_fallback, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, external_id) DO UPDATE SET
            title          = excluded.title,
            org            = excluded.org,
            opp_type       = excluded.opp_type,
            url            = excluded.url,
            location       = excluded.location,
            work_mode      = excluded.work_mode,
            is_remote      = excluded.is_remote,
            date_label     = excluded.date_label,
            deadline       = excluded.deadline,
            compensation   = excluded.compensation,
            duration_info  = excluded.duration_info,
            description    = excluded.description,
            thumbnail_url  = excluded.thumbnail_url,
            is_fallback    = excluded.is_fallback,
            scraped_at     = excluded.scraped_at
    """, (
        opp["source"], opp["external_id"], opp["title"], opp.get("org", ""),
        opp["opp_type"], opp["url"], opp.get("location", ""),
        opp.get("work_mode", ""), int(opp.get("is_remote", False)),
        opp.get("date_label", ""), opp.get("deadline"),
        opp.get("compensation", ""), opp.get("duration_info", ""),
        opp.get("description", ""), opp.get("thumbnail_url", ""),
        int(opp.get("is_fallback", False)), now,
    ))
    c.commit()

    row = c.execute(
        "SELECT id FROM opportunities WHERE source=? AND external_id=?",
        (opp["source"], opp["external_id"]),
    ).fetchone()
    opp_id = row["id"]

    _replace_skills(opp_id, opp.get("skills_raw", []))
    return opp_id


def _replace_skills(opportunity_id: int, skills_raw: List[str]):
    """Fuzzy-matches raw tags to canonical slugs and replaces the join rows.
    Non-skill tags ('National', 'Flagship', 'Beginner') simply won't clear
    the match threshold and are dropped — that's intended, not a bug."""
    c = get_conn()
    c.execute("DELETE FROM opportunity_skills WHERE opportunity_id=?", (opportunity_id,))
    slugs = match_skills(skills_raw)
    for slug in slugs:
        c.execute(
            "INSERT OR IGNORE INTO opportunity_skills (opportunity_id, skill_slug) VALUES (?, ?)",
            (opportunity_id, slug),
        )
    c.commit()


# ── read ──────────────────────────────────────────────────────────────────

def get_opportunities(
    opp_type: Optional[str] = None,
    is_remote: Optional[bool] = None,
    skill_slug: Optional[str] = None,
    source: Optional[str] = None,
    exclude_fallback: bool = False,
    limit: Optional[int] = None,
) -> List[dict]:
    c = get_conn()
    where, params = ["1=1"], []

    if opp_type:
        where.append("o.opp_type=?"); params.append(opp_type)
    if is_remote is not None:
        where.append("o.is_remote=?"); params.append(int(is_remote))
    if source:
        where.append("o.source=?"); params.append(source)
    if exclude_fallback:
        where.append("o.is_fallback=0")

    join = ""
    if skill_slug:
        join = "JOIN opportunity_skills os ON os.opportunity_id = o.id"
        where.append("os.skill_slug=?"); params.append(skill_slug)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    sql = f"""
        SELECT DISTINCT o.* FROM opportunities o
        {join}
        WHERE {' AND '.join(where)}
        ORDER BY (o.deadline IS NULL), o.deadline ASC, o.scraped_at DESC
        {limit_clause}
    """
    rows = c.execute(sql, params).fetchall()
    return [_attach_skills(dict(r)) for r in rows]


def _attach_skills(opp: dict) -> dict:
    c = get_conn()
    rows = c.execute(
        "SELECT skill_slug FROM opportunity_skills WHERE opportunity_id=?",
        (opp["id"],),
    ).fetchall()
    opp["skills"] = [r["skill_slug"] for r in rows]
    return opp


def get_opportunity(opportunity_id: int) -> Optional[dict]:
    row = get_conn().execute(
        "SELECT * FROM opportunities WHERE id=?", (opportunity_id,)
    ).fetchone()
    return _attach_skills(dict(row)) if row else None


def match_by_skills(skill_slugs: List[str], limit: Optional[int] = None) -> List[dict]:
    """
    Simple overlap-count ranking — no ML, consistent with deferring the
    CatBoost recommendation engine until there's real usage data.
    """
    if not skill_slugs:
        return []
    c = get_conn()
    ph = ",".join("?" * len(skill_slugs))
    params = list(skill_slugs)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    rows = c.execute(f"""
        SELECT o.*, COUNT(os.skill_slug) AS match_count
        FROM opportunities o
        JOIN opportunity_skills os ON os.opportunity_id = o.id
        WHERE os.skill_slug IN ({ph})
        GROUP BY o.id
        ORDER BY match_count DESC, (o.deadline IS NULL), o.deadline ASC
        {limit_clause}
    """, params).fetchall()
    return [_attach_skills(dict(r)) for r in rows]


# ── maintenance ───────────────────────────────────────────────────────────

def prune_expired(retention_days: int = 3) -> int:
    """Deletes LIVE-SCRAPED opportunities whose parsed deadline passed
    more than retention_days ago. Fallback rows (is_fallback=1) are
    excluded on purpose: fallback_data.py exists specifically to show
    something rather than nothing when live scraping fails, and its
    entries carry manually-curated dates that go stale over time
    (verified: 2025-dated fallback techfest/hackathon entries were being
    silently deleted the moment they were inserted, since 'today' had
    already passed them — defeating the entire point of having a
    fallback). Fallback rows are replaced wholesale on the next
    successful live scrape of that domain, not pruned by date."""
    c = get_conn()
    cutoff = (datetime.utcnow().date() - timedelta(days=retention_days)).isoformat()
    cur = c.execute(
        "DELETE FROM opportunities WHERE deadline IS NOT NULL AND deadline < ? AND is_fallback=0",
        (cutoff,),
    )
    c.commit()
    return cur.rowcount


def prune_stale_scrapes(stale_after_days: int = 14) -> int:
    """For listings with no parsable deadline (rolling internships/jobs),
    drop rows that haven't been re-confirmed by a scrape in a while —
    the listing may have been taken down and just isn't showing up as
    an explicit expiry anywhere."""
    c = get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=stale_after_days)).isoformat(sep=" ", timespec="seconds")
    cur = c.execute(
        "DELETE FROM opportunities WHERE deadline IS NULL AND scraped_at < ? AND is_fallback=0",
        (cutoff,),
    )
    c.commit()
    return cur.rowcount


def get_opportunity_stats() -> dict:
    c = get_conn()
    total = c.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    by_type = c.execute(
        "SELECT opp_type, COUNT(*) n FROM opportunities GROUP BY opp_type"
    ).fetchall()
    by_source = c.execute(
        "SELECT source, COUNT(*) n FROM opportunities GROUP BY source"
    ).fetchall()
    fallback_count = c.execute(
        "SELECT COUNT(*) FROM opportunities WHERE is_fallback=1"
    ).fetchone()[0]
    return {
        "total": total,
        "by_type": {r["opp_type"]: r["n"] for r in by_type},
        "by_source": {r["source"]: r["n"] for r in by_source},
        "fallback_count": fallback_count,
    }


# ── sync history ──────────────────────────────────────────────────────────
# Separate table from the roadmap domain's sync_history — different
# cadence, different fields (per-source fallback tracking has no roadmap
# equivalent), and keeping the two domains' history independent means a
# migration or cleanup on one never risks the other.

def start_opp_sync(trigger: str = "scheduled", sources_total: int = 0) -> int:
    c = get_conn()
    cur = c.execute(
        "INSERT INTO opportunity_sync_history (started_at, status, trigger, sources_total) "
        "VALUES (?, 'running', ?, ?)",
        (_now(), trigger, sources_total),
    )
    c.commit()
    return cur.lastrowid


def finish_opp_sync(sync_id: int, stats: dict):
    c = get_conn()
    c.execute("""
        UPDATE opportunity_sync_history
        SET completed_at=?, status=?, total_fetched=?, total_upserted=?,
            pruned_expired=?, pruned_stale=?, fallback_sources=?, error_message=?
        WHERE id=?
    """, (
        _now(), stats.get("status", "success"),
        stats.get("total_fetched", 0), stats.get("total_upserted", 0),
        stats.get("pruned_expired", 0), stats.get("pruned_stale", 0),
        json.dumps(stats.get("fallback_sources", [])),
        stats.get("error"), sync_id,
    ))
    c.commit()


def get_opp_sync_history(limit: int = 20) -> List[dict]:
    rows = get_conn().execute(
        "SELECT * FROM opportunity_sync_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["fallback_sources"] = json.loads(d.get("fallback_sources") or "[]")
        except (TypeError, ValueError):
            d["fallback_sources"] = []
        out.append(d)
    return out


def get_latest_opp_sync() -> Optional[dict]:
    h = get_opp_sync_history(1)
    return h[0] if h else None