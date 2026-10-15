"""
sync/heuristic_scorer.py — Phase A opportunity ranking: skill-overlap +
deadline-urgency scoring, no ML. This is what match_by_skills() in
db/opportunity_operations.py already does PART of (raw overlap count) —
this module extends it into the full Phase A design (weighted, normalized,
urgency-aware) rather than replacing the existing function.

WHY THIS MATTERS BEYOND JUST RANKING TODAY'S OPPORTUNITY LIST: the raw
component scores this produces (skill_score, urgency_score, not just the
final combined rank) are the intended PSEUDO-LABEL SOURCE for the future
Phase B/C CatBoost bootstrap — see project history. Keep the component
scores attached to each result, not just the sort order, so that future
step doesn't need to recompute this logic from scratch.

VOCABULARY ALIGNMENT (already correct, worth stating explicitly): both
sides of the skill-overlap computation go through taxonomy/matcher.py's
match_skills() — opportunity_skills via db/opportunity_operations.py's
_replace_skills() at scrape time, user skills via resume_parser.py's
canonical_skill_slugs field. This module assumes both inputs are ALREADY
canonical slugs (e.g. "react", not "React.js") — it does not do any
matching itself.

VOCABULARY ALIGNMENT (already correct, worth stating explicitly): both
sides of the skill-overlap computation go through taxonomy/matcher.py's
match_skills() — opportunity_skills via db/opportunity_operations.py's
_replace_skills() at scrape time, user skills via resume_parser.py's
canonical_skill_slugs field. This module assumes both inputs are ALREADY
canonical slugs (e.g. "react", not "React.js") — it does not do any
matching itself.

ROADMAP ALIGNMENT (now implemented, given the real schema from
db/operations.py — user_roadmaps/user_topic_progress): this module still
does NOT touch the database or taxonomy matcher directly, matching the
same separation principle db/operations.py's own get_topic_tree() already
states ("kept separate here to avoid coupling the base tree-builder to
auth"). roadmap_alignment_score() takes an already-computed SET of
canonical skill slugs the user is actively learning (their incomplete
topics' titles, matched to slugs) — the caller (api/opportunity_routes.py)
is responsible for building that set via db/operations.py +
taxonomy/matcher.py and passing it in. This keeps this module pure and
unit-testable without a live DB, exactly as it's been tested so far.
"""
import logging
from datetime import date, datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Tunable weights (kept as named constants, not magic numbers scattered
#    through the scoring function, so these are the ONE place to adjust
#    the heuristic's behavior without touching scoring logic) ──────────────
# Rebalanced now that roadmap_alignment is real, not a stub. Skill overlap
# still dominates (it's the most direct relevance signal); urgency and
# roadmap-alignment split the remainder. Adjustable, not a fixed law —
# revisit once real interaction data (see db/interaction_operations.py)
# shows what actually predicts engagement.
SKILL_OVERLAP_WEIGHT = 0.5
URGENCY_WEIGHT = 0.25
ROADMAP_ALIGNMENT_WEIGHT = 0.25

# Neutral score used when a signal genuinely can't be computed (e.g. an
# opportunity with no matched skill tags at all, or no deadline) — NOT 0.
# Treating "unknown" as "bad" would systematically bury legitimate
# opportunities that just have thin metadata, which is a real, common case
# (e.g. a rolling internship with no deadline, or a workshop whose scraped
# tags were all non-skill words like "Beginner"/"Free" that correctly
# didn't clear match_skills()'s threshold).
NEUTRAL_SCORE = 0.5

# GATE, not blend: opportunities scoring below this on skill_overlap are
# EXCLUDED from ranking entirely, regardless of how urgent their deadline
# is. Chosen over an additive/multiplicative blend after validation showed
# additive weighting let a zero-skill-match opportunity (a "Unrelated Job"
# test case, cobol/mainframe) narrowly outrank a partial-match opportunity
# purely because its deadline was closer — urgency should never be able to
# make an irrelevant opportunity look competitive. Opportunities with
# skill_score == NEUTRAL_SCORE (no matched tags at all, genuinely unknown
# rather than confirmed-irrelevant) are NOT gated out — "we don't know"
# and "we know it doesn't match" are different situations and shouldn't be
# treated the same way. See rank_opportunities() for the exact gate logic.
SKILL_MATCH_GATE = 0.15

# Deadlines beyond this many days out get the lowest (but non-zero)
# urgency — a linear decay, deliberately simple and explainable over a
# curve-fit exponential, since this is a heuristic meant to be auditable,
# not a trained function. Worth revisiting once real interaction data
# shows whether users actually respond to opportunities this far out.
URGENCY_HORIZON_DAYS = 30


def skill_overlap_score(user_skills: list, opportunity_skills: list) -> float:
    """
    Recall-oriented, not precision-oriented: "what fraction of what THIS
    opportunity is asking for does the user have" — not "what fraction of
    the user's total skills are relevant here". A user with 40 skills and
    a user with 4 skills should score identically on an opportunity if
    both have every skill it's asking for; precision-style scoring would
    unfairly penalize the broader user for having unrelated skills.

    Returns NEUTRAL_SCORE (not 0.0) when the opportunity has no matched
    skill tags at all — see module docstring on why "unknown" isn't "bad".
    """
    if not opportunity_skills:
        return NEUTRAL_SCORE
    if not user_skills:
        return 0.0  # opportunity clearly wants specific skills; user profile has none matched

    user_set = set(user_skills)
    opp_set = set(opportunity_skills)
    overlap = len(user_set & opp_set)
    return overlap / len(opp_set)


def urgency_score(deadline: Optional[str], today: Optional[date] = None) -> float:
    """
    deadline: 'YYYY-MM-DD' string (as produced by opportunity_normalizer's
    normalize_deadline()) or None (rolling/no parseable deadline).

    No deadline -> NEUTRAL_SCORE (rolling opportunities aren't "urgent"
    but shouldn't be buried below urgent ones either — still legitimate,
    open opportunities).

    Already-past deadline -> 0.0. Shouldn't normally occur (prune_expired
    removes these), but scored defensively rather than assuming the
    pruning job has definitely run recently.
    """
    if not deadline:
        return NEUTRAL_SCORE

    today = today or datetime.utcnow().date()
    try:
        deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        log.warning("urgency_score got an unparseable deadline: %r", deadline)
        return NEUTRAL_SCORE

    days_left = (deadline_date - today).days
    if days_left < 0:
        return 0.0
    if days_left == 0:
        return 1.0

    return max(0.0, 1.0 - (days_left / URGENCY_HORIZON_DAYS))


def roadmap_alignment_score(user_gap_skills: Optional[set], opportunity_skills: list) -> float:
    """
    Bonus for opportunities matching skills the user is ACTIVELY LEARNING
    right now (per their primary roadmap's incomplete topics), not just
    skills they already have — this is what lets the ranker say "you're
    currently on the Docker topic in your DevOps roadmap, here's a
    hackathon that wants Docker" rather than only matching a static skill
    list from their resume.

    user_gap_skills: a SET of canonical skill slugs already computed by
    the caller (see api/opportunity_routes.py's _get_user_learning_gap_skills)
    from the user's primary roadmap's incomplete topic titles, matched
    through taxonomy/matcher.py. This function does no DB/taxonomy work
    itself — see module docstring.

    Returns NEUTRAL_SCORE (not 0.0) when gap skills are unavailable — no
    roadmap selected, no primary set, or the caller is the anonymous
    /match endpoint with no user context at all. "We don't know what
    they're learning" is not the same as "this doesn't align with what
    they're learning" and shouldn't be scored as if it were.

    Recall-oriented like skill_overlap_score, for the same reason: "what
    fraction of what this opportunity wants IS something the user is
    actively learning" — not diluted by how large their overall gap set is.
    """
    if user_gap_skills is None:
        return NEUTRAL_SCORE
    if not opportunity_skills:
        return NEUTRAL_SCORE
    if not user_gap_skills:
        return NEUTRAL_SCORE  # roadmap selected but nothing incomplete (or nothing matched) — not a penalty

    opp_set = set(opportunity_skills)
    overlap = len(user_gap_skills & opp_set)
    return overlap / len(opp_set)


def skill_overlap_count(user_skills: list, opportunity_skills: list) -> int:
    """
    Raw intersection size — deliberately separate from skill_overlap_score's
    NORMALIZED ratio. This is the tie-breaker for cases where the ratio
    saturates to 1.0 for very differently-sized matches: an opportunity
    tagged with a single skill the user has (ratio 1.0, count 1) and one
    tagged with four skills the user all has (ratio 1.0, count 4) are NOT
    equally strong signals — the second is corroborated by more independent
    tag matches. Confirmed as a real, common case on production data: a
    large share of scraped opportunities (particularly Devpost hackathons)
    carry only one matched tag ("machine-learning"), which otherwise makes
    wildly different opportunities (a $180,000 Google hackathon and a
    $0-prize student hackathon) score identically indistinguishable.
    """
    return len(set(user_skills) & set(opportunity_skills))


def score_opportunity(user_skills: list, opportunity: dict, user_gap_skills: Optional[set] = None) -> dict:
    """
    Scores ONE opportunity dict (as returned by db/opportunity_operations.py's
    get_opportunities/match_by_skills — must already have "skills" (list of
    canonical slugs, via _attach_skills) and "deadline" fields).

    user_gap_skills: optional precomputed set from
    api/opportunity_routes.py's _get_user_learning_gap_skills() — pass None
    for anonymous/no-roadmap-data contexts (roadmap_alignment_score handles
    that as "unknown", not "zero").

    Returns the original opportunity dict with score fields attached —
    non-destructive, doesn't mutate the input.
    """
    s_score = skill_overlap_score(user_skills, opportunity.get("skills", []))
    u_score = urgency_score(opportunity.get("deadline"))
    r_score = roadmap_alignment_score(user_gap_skills, opportunity.get("skills", []))
    match_count = skill_overlap_count(user_skills, opportunity.get("skills", []))

    combined = (
        SKILL_OVERLAP_WEIGHT * s_score
        + URGENCY_WEIGHT * u_score
        + ROADMAP_ALIGNMENT_WEIGHT * r_score
    )

    out = dict(opportunity)
    out["_skill_score"] = round(s_score, 4)
    out["_urgency_score"] = round(u_score, 4)
    out["_roadmap_score"] = round(r_score, 4)
    out["_combined_score"] = round(combined, 4)
    out["_match_count"] = match_count  # tie-breaker only, see skill_overlap_count docstring
    return out


def rank_opportunities(user_skills: list, opportunities: list, user_gap_skills: Optional[set] = None,
                        apply_gate: bool = True) -> list:
    """
    THE actual Phase A ranking entry point — scores every opportunity,
    GATES OUT ones with confirmed-low skill relevance (see SKILL_MATCH_GATE
    docstring above — "no signal" and "confirmed irrelevant" are treated
    differently), and returns the rest sorted by combined score descending,
    with raw match_count as a secondary sort key (see skill_overlap_count)
    to break ties within score bands that saturate to the same ratio.

    user_gap_skills is computed ONCE by the caller and passed through here
    unchanged for every opportunity in the batch — NOT re-fetched/re-matched
    per opportunity, which would repeat the same DB + fuzzy-matching work
    dozens of times per ranking request for no benefit.

    apply_gate=False is available for debugging/inspection (e.g. an admin
    view that wants to see what got filtered and why) — production callers
    should leave it True.
    """
    scored = [score_opportunity(user_skills, opp, user_gap_skills) for opp in opportunities]

    if apply_gate:
        kept = []
        for o in scored:
            # Gate only on CONFIRMED low relevance — an opportunity with no
            # matched skill tags at all scores exactly NEUTRAL_SCORE, which
            # must never be caught by a gate below NEUTRAL_SCORE. This is
            # an explicit inequality, not just "below the gate", so a
            # future change to NEUTRAL_SCORE's value can't silently start
            # gating out "unknown" opportunities by accident.
            if o["_skill_score"] < SKILL_MATCH_GATE and o["_skill_score"] != NEUTRAL_SCORE:
                continue
            kept.append(o)
        scored = kept

    return sorted(scored, key=lambda o: (o["_combined_score"], o["_match_count"]), reverse=True)