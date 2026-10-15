"""
db/interaction_operations.py — logs impressions and interactions across the
app in ONE shared, extensible table, rather than a domain-separated table
per feature.

WHY SHARED, NOT SEPARATED (unlike opportunity_sync_history vs the roadmap
domain's sync_history, which ARE deliberately kept apart): sync runs are
operationally independent per domain and gain nothing from being queried
together. Interaction events are the opposite — the whole point of this
table, decided early in the project, is that the future contextual bandit
needs ONE unified event stream to learn from, and COSMOS should be able to
log into this SAME table later via a different event_type/source rather
than requiring a new table. Same underlying principle ("separate what's
operationally independent, share what needs joint analysis"), different
conclusion for a different kind of data.

TWO PROPERTIES THIS TABLE MUST HAVE, PER THE ORIGINAL DESIGN DECISION:
  1. REWARD-SHAPED: context_snapshot is a FROZEN JSON blob captured at
     impression time (skill_score, urgency_score, roadmap_score, combined
     score, which skills actually matched) — NOT recomputed later from
     current state. A user's skills, an opportunity's deadline, and the
     heuristic's weights can all change after the fact; a valid reward
     signal needs to reflect what the system actually knew and showed the
     user AT THAT MOMENT, not what's true now.
  2. SOURCE-EXTENSIBLE: item_type/event_type/source are free-text, not
     enums tied to opportunities specifically. COSMOS, roadmap-topic
     interactions, or anything else added later reuses this same table.

DESIGN CHOICE: append-only, immutable rows — no impression row ever gets
UPDATEd with a later reward. This matches the atomic-upsert-over-SELECT-
then-UPDATE philosophy already used elsewhere in this codebase (see the
roadmaps table's race-condition fix) — an UPDATE-based "attach reward to
this specific impression" design would need the frontend to echo back an
impression_id on every click, which I haven't verified exists in the
current UI. Instead, reward is COMPUTED at read time by joining an
impression to any interaction event for the same (user_id, item_type,
item_id) within a following time window — see compute_rewards(). This is
a deliberate simplification, not an oversight: it's slightly less precise
than exact impression-to-click linkage, but requires no frontend changes
and has zero risk of write-time races.

SCHEMA PLACEMENT: this file assumes a CREATE TABLE has been run somewhere
during app startup (same pattern as db/__init__.py's init_db() running
models.py's SCHEMA_SQL). INTERACTION_SCHEMA_SQL below is the DDL to splice
into wherever your current schema init actually lives — I don't have
visibility into whether that's still models.py or has moved, so this is
provided standalone rather than assumed to be wired in automatically.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from careercompass.db import get_conn

log = logging.getLogger(__name__)

INTERACTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS interaction_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    item_type        TEXT    NOT NULL,   -- 'opportunity' | 'roadmap' | 'topic' | future types
    item_id          TEXT    NOT NULL,   -- stored as TEXT deliberately — item_type disambiguates
                                          -- which domain table it logically refers to, so this
                                          -- table isn't locked to one item id type/format forever
    event_type       TEXT    NOT NULL,   -- 'impression' | 'clicked' | 'bookmarked' |
                                          -- 'dismissed' | 'applied' | future: 'cosmos_query' etc
    source           TEXT    NOT NULL,   -- WHICH SYSTEM logged this: 'opportunity_ranker' |
                                          -- 'cosmos' | future sources. Distinct from
                                          -- opportunities.source (scraper origin) — do not confuse
                                          -- the two "source" concepts across tables.
    context_snapshot TEXT    DEFAULT '{}', -- JSON, frozen at write time — see module docstring
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_interaction_user       ON interaction_events(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_item        ON interaction_events(item_type, item_id);
CREATE INDEX IF NOT EXISTS idx_interaction_event_type  ON interaction_events(event_type);
CREATE INDEX IF NOT EXISTS idx_interaction_created     ON interaction_events(created_at);
"""

# Simple, documented, ADJUSTABLE reward mapping — not a fixed law. Revisit
# once real usage data shows what actually predicts continued engagement.
REWARD_MAP = {
    "clicked": 1.0,
    "bookmarked": 1.0,
    "applied": 1.0,
    "dismissed": -1.0,
    # "impression" alone (no follow-up event) resolves to 0.0 — see compute_rewards()
}

# How long after an impression a follow-up event still counts as "caused by"
# that impression, for reward-joining purposes. Chosen as a reasonable
# single-session window — long enough for someone to browse and come back,
# short enough that an unrelated bookmark of the same item three weeks
# later doesn't get misattributed to an old impression.
REWARD_ATTRIBUTION_WINDOW_HOURS = 2


def init_interaction_schema():
    conn = get_conn()
    conn.executescript(INTERACTION_SCHEMA_SQL)
    conn.commit()
    log.info("interaction_events schema ready")


def _now() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")


def log_event(user_id: str, item_type: str, item_id, event_type: str,
              source: str, context_snapshot: Optional[dict] = None) -> int:
    """
    THE single write path for this table — every impression, click,
    bookmark, dismissal, or future COSMOS event goes through this one
    function. context_snapshot should be the FROZEN feature state at the
    moment of the event (e.g. straight from heuristic_scorer's
    score_opportunity() output for an 'impression') — never pass something
    that will be recomputed differently later.
    """
    c = get_conn()
    cur = c.execute("""
        INSERT INTO interaction_events
            (user_id, item_type, item_id, event_type, source, context_snapshot, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(user_id), item_type, str(item_id), event_type, source,
        json.dumps(context_snapshot or {}), _now(),
    ))
    c.commit()
    return cur.lastrowid


def log_impression(user_id: str, item_type: str, item_id, context_snapshot: dict,
                    source: str = "opportunity_ranker") -> int:
    """Convenience wrapper — the common case of logging that something was
    shown to a user, with its scoring context frozen at that moment."""
    return log_event(user_id, item_type, item_id, "impression", source, context_snapshot)


def log_interaction(user_id: str, item_type: str, item_id, event_type: str,
                     source: str = "opportunity_ranker") -> int:
    """Convenience wrapper for click/bookmark/dismiss/applied — these don't
    need a context_snapshot of their own; the reward gets attributed back
    to the ORIGINAL impression's snapshot at read time (see compute_rewards)."""
    if event_type not in REWARD_MAP:
        log.warning("log_interaction called with unmapped event_type %r — "
                    "it will be logged but compute_rewards() won't know its "
                    "reward value until REWARD_MAP is updated", event_type)
    return log_event(user_id, item_type, item_id, event_type, source, context_snapshot=None)


def compute_rewards(item_type: Optional[str] = None, source: Optional[str] = None,
                     since: Optional[str] = None) -> List[dict]:
    """
    THE actual training-data extraction function for Phase B/C and the
    future bandit — turns the raw event log into (context, reward) pairs.

    For every 'impression' event, looks for any interaction event on the
    SAME (user_id, item_type, item_id) within REWARD_ATTRIBUTION_WINDOW_HOURS
    afterward. Reward = REWARD_MAP value of the first such event found, or
    0.0 if no follow-up interaction occurred in the window (a real,
    meaningful signal — "shown but ignored" — not a missing value).

    Returns a list of {context_snapshot (dict), reward (float), user_id,
    item_type, item_id, impression_at} — ready to hand to a training script
    without it needing to know anything about the underlying event-log
    representation.
    """
    c = get_conn()
    where, params = ["event_type = 'impression'"], []
    if item_type:
        where.append("item_type = ?"); params.append(item_type)
    if source:
        where.append("source = ?"); params.append(source)
    if since:
        where.append("created_at >= ?"); params.append(since)

    impressions = c.execute(f"""
        SELECT id, user_id, item_type, item_id, context_snapshot, created_at
        FROM interaction_events
        WHERE {' AND '.join(where)}
        ORDER BY created_at
    """, params).fetchall()

    results = []
    for imp in impressions:
        window_end = (
            datetime.strptime(imp["created_at"], "%Y-%m-%d %H:%M:%S")
            + timedelta(hours=REWARD_ATTRIBUTION_WINDOW_HOURS)
        ).strftime("%Y-%m-%d %H:%M:%S")

        follow_up = c.execute("""
            SELECT event_type FROM interaction_events
            WHERE user_id = ? AND item_type = ? AND item_id = ?
              AND event_type != 'impression'
              AND created_at >= ? AND created_at <= ?
            ORDER BY created_at
            LIMIT 1
        """, (imp["user_id"], imp["item_type"], imp["item_id"], imp["created_at"], window_end)).fetchone()

        reward = REWARD_MAP.get(follow_up["event_type"], 0.0) if follow_up else 0.0

        try:
            context = json.loads(imp["context_snapshot"])
        except (TypeError, json.JSONDecodeError):
            context = {}

        results.append({
            "user_id": imp["user_id"],
            "item_type": imp["item_type"],
            "item_id": imp["item_id"],
            "context_snapshot": context,
            "reward": reward,
            "impression_at": imp["created_at"],
        })

    return results


def get_event_counts(item_type: Optional[str] = None) -> Dict[str, int]:
    """Quick sanity-check / admin-dashboard helper — counts per event_type."""
    c = get_conn()
    where, params = ["1=1"], []
    if item_type:
        where.append("item_type = ?"); params.append(item_type)
    rows = c.execute(f"""
        SELECT event_type, COUNT(*) n FROM interaction_events
        WHERE {' AND '.join(where)}
        GROUP BY event_type
    """, params).fetchall()
    return {r["event_type"]: r["n"] for r in rows}