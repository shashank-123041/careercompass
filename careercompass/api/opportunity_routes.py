"""
api/opportunity_routes.py — REST API v2 for the opportunity finder.

PATCHED: wires in the full Phase A heuristic (sync/heuristic_scorer.py —
skill overlap + deadline urgency + gating, not just raw overlap-count),
prefers the precomputed canonical_skill_slugs field over redundant
re-matching, and logs impressions for the future CatBoost bootstrap.
See inline notes at each change for the "why".

Serves data from the same local SQLite knowledge base as api/routes.py,
reusing its `cached()` TTL-cache decorator rather than duplicating it —
one cache dict, one eviction policy, for the whole API layer.

Endpoint shape deliberately mirrors api/routes.py's roadmap endpoints
(list / detail / stats / sync status / sync trigger) so the frontend
and any future API consumer only has to learn one pattern.
"""
import json
import logging

from flask import Blueprint, jsonify, request, session

from careercompass.db import opportunity_operations as ops
from careercompass.db import operations as core_ops
from careercompass.db import interaction_operations as interaction_ops
from careercompass.sync.opportunity_engine import get_state, trigger_manual
from careercompass.ranking.heuristic_scorer import rank_opportunities
from careercompass.taxonomy.matcher import match_skills
from careercompass.api.routes import cached, bust_cache
from careercompass.auth.routes import login_required, admin_required

log = logging.getLogger(__name__)

opportunities_api = Blueprint("opportunities_api", __name__, url_prefix="/api/v2/opportunities")

VALID_TYPES = {"hackathon", "internship", "job", "techfest", "workshop"}


# ── List / filter ────────────────────────────────────────────────────────────

@opportunities_api.get("")
@opportunities_api.get("/")
@cached(60)
def list_opportunities():
    """
    Filters:
      ?type=hackathon|internship|job|techfest|workshop
      ?remote=true|false
      ?skill=react              (canonical skill slug — see taxonomy/skill_taxonomy.py)
      ?source=Devpost
      ?include_fallback=true    (default false — hide curated fallback data once live data exists)
      ?limit=N                  (optional — omit for no limit, returns every matching row)
    """
    opp_type = request.args.get("type")
    if opp_type and opp_type not in VALID_TYPES:
        return jsonify({"error": f"invalid type — must be one of {sorted(VALID_TYPES)}"}), 400

    remote_arg = request.args.get("remote")
    is_remote = None
    if remote_arg is not None:
        is_remote = remote_arg.lower() in ("true", "1", "yes")

    include_fallback = request.args.get("include_fallback", "false").lower() in ("true", "1", "yes")
    limit_arg = request.args.get("limit")
    limit = int(limit_arg) if limit_arg is not None else None

    results = ops.get_opportunities(
        opp_type=opp_type,
        is_remote=is_remote,
        skill_slug=request.args.get("skill"),
        source=request.args.get("source"),
        exclude_fallback=not include_fallback,
        limit=limit,
    )
    return jsonify({"count": len(results), "opportunities": results})


@opportunities_api.get("/<int:opportunity_id>")
def get_opportunity(opportunity_id):
    opp = ops.get_opportunity(opportunity_id)
    if not opp:
        return jsonify({"error": "not found"}), 404
    return jsonify({"opportunity": opp})


# ── Skill-based matching ──────────────────────────────────────────────────────

def _get_user_learning_gap_skills(user_id) -> set:
    """
    Builds the set of canonical skill slugs the user is ACTIVELY LEARNING
    right now, for sync/heuristic_scorer.py's roadmap_alignment_score.
    Combines db/operations.py (schema access) with taxonomy/matcher.py
    (title -> slug matching) — this orchestration belongs at the route
    layer, not inside heuristic_scorer.py itself (see that module's
    docstring on why it stays DB/taxonomy-agnostic) and not inside
    db/operations.py either (matching isn't SQL).

    Uses the user's PRIMARY roadmap only (the strongest, least-diluted
    signal of current focus — see db/operations.py's is_primary/
    maybe_advance_primary). Falls back to the first-selected roadmap if
    none is marked primary (shouldn't normally happen given
    maybe_advance_primary's logic, but handled rather than assumed).

    Returns an empty set (not None) when the user has no roadmaps selected
    at all — heuristic_scorer.py's roadmap_alignment_score treats an empty
    set as "nothing incomplete to align with" (neutral), same as None
    ("no roadmap context available"). Both resolve to NEUTRAL_SCORE, so
    this distinction doesn't currently change behavior — kept as two
    return paths anyway since they mean different things and a future
    change might want to treat them differently.
    """
    roadmaps = core_ops.get_user_roadmaps(user_id)
    if not roadmaps:
        return set()

    primary = next((r for r in roadmaps if r.get("is_primary")), roadmaps[0])
    incomplete_titles = core_ops.get_incomplete_topic_titles(user_id, primary["id"])
    if not incomplete_titles:
        return set()

    return set(match_skills(incomplete_titles))


def _rank_and_limit(slugs: list, limit, user_id=None) -> list:
    """
    Shared by both /match and /match/me. FIX: previously called
    ops.match_by_skills(slugs, limit) directly — raw overlap-COUNT
    ranking only, no urgency, no relevance gating, no roadmap alignment
    (see conversation history: raw overlap-count let a zero-skill-match
    but urgent opportunity outrank a genuinely relevant one).

    Pulls the FULL candidate pool first (limit=None — any opportunity
    with at least one matching skill, via match_by_skills's existing
    SQL join), THEN applies the full heuristic (urgency + gate + roadmap
    alignment) via rank_opportunities(), THEN slices to the requested
    limit. Limiting before re-ranking would risk returning fewer than
    `limit` results once gating removes low-relevance candidates from an
    already-small pre-limited set.

    user_id is optional — None for the anonymous /match endpoint (no
    roadmap-alignment signal available, handled as neutral by
    heuristic_scorer, not as a penalty).
    """
    candidates = ops.match_by_skills(slugs, limit=None)
    gap_skills = _get_user_learning_gap_skills(user_id) if user_id is not None else None
    ranked = rank_opportunities(user_skills=slugs, opportunities=candidates, user_gap_skills=gap_skills)
    return ranked[:limit] if limit else ranked

@opportunities_api.get("/match")
def match_opportunities():
    """
    ?skills=react,python,docker   — free-text, fuzzy-matched to canonical slugs first
    ?skill_slugs=react,python     — already-canonical slugs, skips fuzzy matching
    ?limit=N                     (optional — omit for no limit)

    Anonymous endpoint (no login) — ranked with the full heuristic
    (urgency + gate) but NOT impression-logged, since there's no user_id
    to attribute the log to. See /match/me for the personalized,
    logged version.
    """
    limit_arg = request.args.get("limit")
    limit = int(limit_arg) if limit_arg is not None else None

    raw_skills = request.args.get("skills", "")
    slug_param = request.args.get("skill_slugs", "")

    slugs = []
    if slug_param:
        slugs = [s.strip() for s in slug_param.split(",") if s.strip()]
    elif raw_skills:
        slugs = match_skills([s.strip() for s in raw_skills.split(",") if s.strip()])

    if not slugs:
        return jsonify({"error": "provide ?skills= (free text) or ?skill_slugs= (canonical)"}), 400

    results = _rank_and_limit(slugs, limit)  # user_id=None — no roadmap context for anonymous requests
    return jsonify({"matched_skills": slugs, "count": len(results), "opportunities": results})


def _get_canonical_skill_slugs(user) -> list:
    """
    FIX: previously always re-derived skills via _flatten_resume_skills()
    + match_skills() on every single request, even though resume_parser.py
    now computes canonical_skill_slugs ONCE at parse time. Prefers that
    precomputed field; falls back to the old flatten+match path only for
    records parsed before this field existed (backward compatible with
    resumes already in the DB from before this change).

    Returns [] if neither path yields anything — caller decides how to
    respond to that (see match_my_opportunities below).
    """
    raw_skills = getattr(user, "resume_skills", None) or {}
    if isinstance(raw_skills, str):
        try:
            raw_skills = json.loads(raw_skills)
        except (TypeError, ValueError):
            raw_skills = {}
    if not isinstance(raw_skills, dict):
        return []

    precomputed = raw_skills.get("canonical_skill_slugs")
    if precomputed:
        return [s for s in precomputed if isinstance(s, str) and s.strip()]

    # Fallback path for older records — same logic as before this patch
    flat = _flatten_resume_skills(raw_skills)
    if not flat:
        return []
    return match_skills(flat)


def _flatten_resume_skills(resume_skills) -> list:
    """
    Pulls a flat list of skill strings out of the onboarding-flow's resume
    JSON shape: {"skills": {"technical": {...}, "domain_knowledge": {...}},
    "all_skills_flat": [...]}. UNCHANGED from before this patch — kept as
    the fallback path for records without canonical_skill_slugs yet.

    Prefers all_skills_flat (already deduplicated by the onboarding flow).
    Falls back to manually flattening technical + domain_knowledge only if
    all_skills_flat is missing or empty — e.g. an older stored record from
    before that field existed.
    """
    if isinstance(resume_skills, str):
        try:
            resume_skills = json.loads(resume_skills)
        except (TypeError, ValueError):
            resume_skills = {}
    if not isinstance(resume_skills, dict):
        return []

    flat = resume_skills.get("all_skills_flat")
    if flat:
        return [s for s in flat if isinstance(s, str) and s.strip()]

    skills_block = resume_skills.get("skills") or {}
    technical = skills_block.get("technical") or {}
    domain = skills_block.get("domain_knowledge") or {}

    out = list(technical.keys()) + list(domain.keys())
    for subs in technical.values():
        if subs:
            out.extend(subs)
    return [s for s in out if isinstance(s, str) and s.strip()]


@opportunities_api.get("/match/me")
@login_required
def match_my_opportunities():
    """
    Personalized version of /match — pulls skills from the logged-in
    user's stored resume_skills instead of a ?skills= query param.
    Requires an active session (see auth/routes.py's login_required).

    PATCHED: now uses the full heuristic (urgency + gate) via
    _rank_and_limit(), prefers precomputed canonical_skill_slugs via
    _get_canonical_skill_slugs(), and logs an impression event per
    result shown — the training-data source for the future Phase B/C
    CatBoost bootstrap (see db/interaction_operations.py).
    """
    user = core_ops.get_user_by_id(session["user_id"])
    if not user:
        return jsonify({"error": "user not found"}), 404

    slugs = _get_canonical_skill_slugs(user)
    if not slugs:
        return jsonify({
            "matched_skills": [], "count": 0, "opportunities": [],
            "reason": "no_resume_skills_or_no_taxonomy_matches",
        })

    limit_arg = request.args.get("limit")
    limit = int(limit_arg) if limit_arg is not None else None
    results = _rank_and_limit(slugs, limit, user_id=user.id)

    # Impression logging — best-effort. A logging failure should never
    # break the actual response the user is waiting on; log and move on.
    for opp in results:
        try:
            interaction_ops.log_impression(
                user_id=user.id, item_type="opportunity", item_id=opp["id"],
                context_snapshot={
                    "skill_score": opp.get("_skill_score"),
                    "urgency_score": opp.get("_urgency_score"),
                    "roadmap_score": opp.get("_roadmap_score"),
                    "combined_score": opp.get("_combined_score"),
                },
                source="opportunity_ranker",
            )
        except Exception as e:
            log.warning("Failed to log impression for opportunity %s: %s", opp.get("id"), e)

    return jsonify({"matched_skills": slugs, "count": len(results), "opportunities": results})


@opportunities_api.post("/<int:opportunity_id>/interact")
@login_required
def log_interaction_event(opportunity_id):
    """
    NEW endpoint — the frontend calls this when a user clicks, bookmarks,
    dismisses, or applies to an opportunity that was shown via /match/me.
    Without this, impressions get logged but nothing ever completes the
    (context, reward) pair compute_rewards() needs — this is the other
    half of that loop, not previously wired up anywhere.

    Body: {"event_type": "clicked" | "bookmarked" | "dismissed" | "applied"}
    """
    event_type = (request.json or {}).get("event_type", "").strip()
    valid_events = {"clicked", "bookmarked", "dismissed", "applied"}
    if event_type not in valid_events:
        return jsonify({"error": f"event_type must be one of {sorted(valid_events)}"}), 400

    interaction_ops.log_interaction(
        user_id=session["user_id"], item_type="opportunity", item_id=opportunity_id,
        event_type=event_type, source="opportunity_ranker",
    )
    return jsonify({"status": "ok"})


# ── Stats ─────────────────────────────────────────────────────────────────────

@opportunities_api.get("/stats")
def stats():
    return jsonify(ops.get_opportunity_stats())


# ── Sync ─────────────────────────────────────────────────────────────────────

@opportunities_api.get("/sync/status")
def sync_status():
    latest = ops.get_latest_opp_sync()
    return jsonify({
        "live": get_state(),
        "latest": latest,
    })


@opportunities_api.get("/sync/history")
def sync_history():
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({"history": ops.get_opp_sync_history(limit)})


@opportunities_api.post("/sync/trigger")
@admin_required
def sync_trigger():
    bust_cache()
    ok = trigger_manual()
    if ok:
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"}), 409


# ── Admin ─────────────────────────────────────────────────────────────────────

@opportunities_api.get("/admin/sources")
@admin_required
def admin_sources():
    """Per-source breakdown for the admin dashboard — count and how many
    of each source's rows are curated fallback data vs. genuinely live."""
    from careercompass.db import get_conn
    c = get_conn()
    rows = c.execute("""
        SELECT source,
               COUNT(*) AS total,
               SUM(CASE WHEN is_fallback=1 THEN 1 ELSE 0 END) AS fallback_count,
               MAX(scraped_at) AS last_scraped_at
        FROM opportunities
        GROUP BY source
        ORDER BY total DESC
    """).fetchall()
    return jsonify({"sources": [dict(r) for r in rows]})