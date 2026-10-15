"""
ml/opportunity_bootstrap.py — Phase B pseudo-label bootstrap for the
opportunity-ranking CatBoost model.

WHY THIS EXISTS: real reward data via db/interaction_operations.py's
compute_rewards() will be thin for a long time after launch — a brand-new
CatBoost model can't train on a few dozen labeled rows. This script
generates a full pseudo-labeled training set FROM sync/heuristic_scorer.py's
already-validated component scores, run across the SAME (user, candidate
opportunity) pairs the heuristic would actually see in production —
same candidate generation (opp_ops.match_by_skills), same scoring, same
gate — so the bootstrap distribution matches what /match/me actually
returns, not some broader/different sample.

TARGET DESIGN — mirrors, does not copy: the pseudo-label is NOT trained
on the heuristic's own component scores as features (see module note
above) — only on their raw, pre-heuristic inputs. Two targets are
produced side by side so the training script can pick either:
  - pseudo_label_reg:       heuristic's _combined_score (regression warm-start)
  - pseudo_label_relevant:  1 if _combined_score >= RELEVANCE_THRESHOLD,
                             else 0 (classification warm-start)

GATE CONSISTENCY: pairs gated out by SKILL_MATCH_GATE are excluded from
the training set entirely, not included with a low label — a confirmed-
irrelevant pair teaches the model nothing about ranking among relevant
ones and would just be noise (see heuristic_scorer.py's own reasoning
for why gating beats additive blending).

BLENDING WITH REAL DATA (next step, NOT implemented here): once
db/interaction_operations.py's compute_rewards() has enough volume, real
reward rows should be CONCATENATED to this pseudo-labeled set with a
sample_weight column favoring real rows over pseudo rows — not a
straight swap, so opportunities/users with no real interactions yet
don't silently drop out of training.
"""
import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from careercompass.config import Config
from careercompass.db import operations as core_ops
from careercompass.db import opportunity_operations as opp_ops
from careercompass.ranking.heuristic_scorer import score_opportunity, SKILL_MATCH_GATE, NEUTRAL_SCORE
from careercompass.taxonomy.matcher import match_skills

log = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.5   # combined_score cutoff for pseudo_label_relevant —
                            # tune after eyeballing the printed score distribution below
OUTPUT_PATH = Path(Config.OPP_PSEUDO_LABELS_PATH)


# ── skill/gap resolution — same precedence as api/opportunity_routes.py ───────
# Duplicated rather than imported to keep this ml/ script decoupled from
# api/ (route-layer concerns like session/auth shouldn't leak into a batch
# job). If this logic changes in opportunity_routes.py, mirror the change
# here too — small enough surface area that a shared helper isn't worth
# the coupling yet.

def _get_canonical_skill_slugs(user) -> list:
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
    flat = raw_skills.get("all_skills_flat") or []
    return match_skills([s for s in flat if isinstance(s, str)]) if flat else []


def _get_user_gap_skills(user_id) -> set:
    roadmaps = core_ops.get_user_roadmaps(user_id)
    if not roadmaps:
        return set()
    primary = next((r for r in roadmaps if r.get("is_primary")), roadmaps[0])
    titles = core_ops.get_incomplete_topic_titles(user_id, primary["id"])
    return set(match_skills(titles)) if titles else set()


def _days_to_deadline(deadline: Optional[str]) -> Optional[int]:
    if not deadline:
        return None
    try:
        d = datetime.strptime(deadline, "%Y-%m-%d").date()
        return (d - date.today()).days
    except (ValueError, TypeError):
        return None


# ── row construction ───────────────────────────────────────────────────────

def build_row(user, opp: dict, user_skills: list, gap_skills: set, user_source: str = "real") -> Optional[dict]:
    scored = score_opportunity(user_skills, opp, user_gap_skills=gap_skills)

    if scored["_skill_score"] < SKILL_MATCH_GATE and scored["_skill_score"] != NEUTRAL_SCORE:
        return None  # excluded, not low-labeled — see module docstring

    opp_skills = opp.get("skills", []) or []
    days_left = _days_to_deadline(opp.get("deadline"))

    return {
        "user_id": user.id,
        "user_source": user_source,  # 'real' | 'synthetic' — see SYNTHETIC BOOTSTRAP section.
                                      # NEVER drop this column when concatenating — a future
                                      # trainer needs to be able to down-weight or exclude
                                      # synthetic rows once real interaction volume exists.
        "opportunity_id": opp.get("id"),
        # raw features — NOT the heuristic's own component scores
        "match_count": scored["_match_count"],
        "opp_skill_count": len(opp_skills),
        "user_skill_count": len(user_skills),
        "gap_overlap_count": len(gap_skills & set(opp_skills)) if gap_skills else 0,
        "days_to_deadline": days_left if days_left is not None else -1,  # -1 sentinel; has_deadline flag disambiguates from a real 0
        "has_deadline": int(days_left is not None),
        "opp_type": opp.get("opp_type", ""),
        "source": opp.get("source", ""),
        "is_remote": int(bool(opp.get("is_remote"))),
        "is_fallback": int(bool(opp.get("is_fallback"))),
        "has_compensation": int(bool(opp.get("compensation"))),
        # pseudo-labels
        "pseudo_label_reg": scored["_combined_score"],
        "pseudo_label_relevant": int(scored["_combined_score"] >= RELEVANCE_THRESHOLD),
    }


def generate(limit_users: Optional[int] = None) -> list:
    """
    Iterates every user with resume data, builds their candidate opportunity
    pool the SAME way api/opportunity_routes.py's /match/me does (via
    opp_ops.match_by_skills — skill-overlap SQL join, no limit), scores
    each candidate through the Phase A heuristic, and emits one training
    row per (user, opportunity) pair that clears the relevance gate.
    """
    users = core_ops.get_all_users_with_resume_skills()
    if limit_users:
        users = users[:limit_users]

    rows = []
    skipped_no_skills = skipped_no_candidates = 0

    for user in users:
        user_skills = _get_canonical_skill_slugs(user)
        if not user_skills:
            skipped_no_skills += 1
            continue

        candidates = opp_ops.match_by_skills(user_skills, limit=None)
        if not candidates:
            skipped_no_candidates += 1
            continue

        gap_skills = _get_user_gap_skills(user.id)
        for opp in candidates:
            row = build_row(user, opp, user_skills, gap_skills, user_source="real")
            if row:
                rows.append(row)

    log.info("Bootstrap (real): %d rows from %d users (%d skipped: no skills, %d skipped: no candidates)",
              len(rows), len(users), skipped_no_skills, skipped_no_candidates)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC USER BOOTSTRAP — Step 2
#
# WHY: with a single-digit count of real signed-up users, generate()'s
# output shares one skill profile across hundreds of rows — CatBoost would
# just memorize that one person, not learn anything about skill-to-
# opportunity relevance in general. Real signups will take time to
# accumulate; this fills the gap WITHOUT inventing arbitrary data.
#
# GROUNDING: skill profiles are sampled from final_dataset_v3.csv — the
# SAME tiered (real + reviewed-synthetic) dataset the role classifier was
# trained on, not fresh made-up skill lists. This keeps provenance honest:
# these are real skill co-occurrence patterns, applied to a synthetic
# "this is a signed-up user" wrapper, not synthetic skills on top of
# synthetic skills.
#
# VOCABULARY BRIDGE: final_dataset_v3.csv's skill_tags use the role
# classifier's granular vocabulary (e.g. "react.js", "postgresql") — a
# DIFFERENT, wider space than the ~65-term canonical taxonomy opportunities
# are matched against (see resume_parser.py's module docstring on this
# exact distinction). Every synthetic profile's tags are run through
# taxonomy.matcher.match_skills() before use — the SAME canonicalization
# step a real resume goes through via canonical_skill_slugs — so synthetic
# and real rows are apples-to-apples once they reach build_row().
#
# GAP SKILLS: synthetic users have no real user_roadmaps/user_topic_progress
# row to query (see _get_user_gap_skills above, which needs a real DB
# user). Gap skills are instead derived from what OTHER profiles sharing
# the SAME role_slug in the full dataset have that this profile doesn't —
# a real, role-correlated stand-in for "what a person on this path is
# likely learning next", not arbitrary noise. (First version of this
# sampled uniformly from all 65 canonical skills regardless of role —
# verified via the printed score distribution that this was wrong: every
# synthetic row capped at exactly combined_score=0.750, the exact value
# you get when roadmap_alignment_score is stuck at 0 for effectively every
# row, since a DevOps profile's "gap skills" could land on things like
# figma/solidity/ruby with near-zero chance of ever overlapping a DevOps
# opportunity's tags. Role-correlated sampling fixes this at the source.)
# ═══════════════════════════════════════════════════════════════════════════
import ast
import random
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd

from careercompass.taxonomy.skill_taxonomy import CANONICAL_SKILLS

# Grounding dataset — the SAME file the role classifier trains on
# (Config.FINAL_DATASET_CSV, data/final/final_dataset_v7.csv by default).
FINAL_DATASET_CSV = str(Config.FINAL_DATASET_CSV)

N_SYNTHETIC_USERS = 150   # sample size — bump once you've eyeballed output quality
SYNTHETIC_SEED = 42       # reproducible sampling, same discipline as train_role_classifier.py's CV seed
GAP_SKILLS_MAX = 3        # max simulated "actively learning" skills per synthetic user


def _build_role_skill_pools(path: str) -> dict:
    """
    Scans the FULL dataset (not just the n sampled for profiles) once,
    canonicalizes every row's skill_tags, and groups the union of
    canonical skills seen per role_slug. This is the population gap-skills
    get drawn from — built from all ~2,429 rows for a stable per-role
    picture, independent of which small subset ends up sampled as actual
    synthetic profiles.

    SAFE ON MISSING FILE: returns {} rather than raising — this function
    is called from a background scheduler path (see run_scheduled_bootstrap
    below) that must never crash the app just because the grounding
    dataset isn't present in a given deployment/environment. An empty
    pool falls back cleanly in _synthetic_gap_skills() (which already has
    a CANONICAL_SKILLS fallback for unseen roles).
    """
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        log.warning("Synthetic bootstrap grounding dataset not found/empty at %r (%s) — "
                    "synthetic profile generation will be skipped this run.", path, e)
        return {}

    pools = defaultdict(set)
    for _, row in df.iterrows():
        role = row.get("role_slug", "")
        if not role:
            continue
        try:
            raw_tags = ast.literal_eval(row["skill_tags"])
        except (ValueError, SyntaxError):
            continue
        canonical = match_skills([t for t in raw_tags if isinstance(t, str)])
        pools[role].update(canonical)
    return dict(pools)


def _load_synthetic_profiles(path: str, n: int, seed: int) -> tuple:
    """
    Samples n rows from final_dataset_v3.csv, canonicalizes each row's
    skill_tags through match_skills(), and drops any row whose tags don't
    map to ANY canonical taxonomy term (a real, expected outcome for some
    niche Tier 3 role skill sets — not an error, just not usable for
    opportunity matching, which only knows the ~65-term canonical space).

    Returns (profiles, n_sampled) — n_sampled may be less than n if the
    dataset itself has fewer rows than requested.

    SAFE ON MISSING FILE: returns ([], 0) rather than raising — see
    _build_role_skill_pools's docstring for why this matters for the
    scheduled backend path.
    """
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        log.warning("Synthetic bootstrap grounding dataset not found/empty at %r (%s) — "
                    "skipping synthetic profile generation this run.", path, e)
        return [], 0

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    n_sampled = len(df)

    profiles = []
    for i, row in df.iterrows():
        try:
            raw_tags = ast.literal_eval(row["skill_tags"])
        except (ValueError, SyntaxError):
            continue
        canonical = match_skills([t for t in raw_tags if isinstance(t, str)])
        if not canonical:
            continue  # tags didn't map to the canonical taxonomy space — unusable, not an error
        profiles.append({
            "id": f"synth_{i:04d}",
            "role_slug": row.get("role_slug", ""),
            "canonical_skills": canonical,
        })
    return profiles, n_sampled


def _synthetic_gap_skills(role_slug: str, role_pools: dict, rng: random.Random) -> set:
    """
    Draws gap skills from the role's typical skill pool. Deliberately does
    NOT exclude the profile's own user_skills from this pool — verified
    that an earlier version which did (pool = role_pool - user_skills)
    created an artificial ceiling: it made gap_skills and user_skills
    mutually exclusive by construction, so roadmap_alignment_score could
    never be high on the SAME opportunity where skill_overlap_score was
    also high (every synthetic row capped at combined_score=0.750 as a
    result). The real system has no such constraint — _get_user_gap_skills
    pulls incomplete roadmap TOPIC titles independently of resume_skills,
    so a real user's "gap" and "known" skills can and do overlap (they
    have a skill in their resume AND an unchecked roadmap topic for the
    same thing). Matching that real behavior here, quirk included, is
    correct — the goal is fidelity to what /match/me actually produces,
    not an idealized version of it.

    Falls back to the full canonical space only if the role has no pool
    at all (e.g. an unroutable/unseen role_slug).
    """
    candidate_pool = list(role_pools.get(role_slug, set())) or list(CANONICAL_SKILLS)
    if not candidate_pool:
        return set()
    k = rng.randint(0, min(GAP_SKILLS_MAX, len(candidate_pool)))
    return set(rng.sample(candidate_pool, k)) if k else set()


def generate_synthetic(n: int = N_SYNTHETIC_USERS, seed: int = SYNTHETIC_SEED,
                        dataset_path: str = FINAL_DATASET_CSV) -> list:
    role_pools = _build_role_skill_pools(dataset_path)
    profiles, n_sampled = _load_synthetic_profiles(dataset_path, n, seed)
    n_dropped_no_canonical = n_sampled - len(profiles)
    rng = random.Random(seed)

    rows = []
    skipped_no_candidates = 0

    for profile in profiles:
        user_skills = profile["canonical_skills"]
        user = SimpleNamespace(id=profile["id"])  # only .id is used past this point

        candidates = opp_ops.match_by_skills(user_skills, limit=None)
        if not candidates:
            skipped_no_candidates += 1
            continue

        gap_skills = _synthetic_gap_skills(profile["role_slug"], role_pools, rng)
        for opp in candidates:
            row = build_row(user, opp, user_skills, gap_skills, user_source="synthetic")
            if row:
                rows.append(row)

    log.info("Bootstrap (synthetic): %d rows from %d usable profiles out of %d sampled "
              "(%d skipped: no matching candidates, %d dropped earlier: no canonical skill match)",
              len(rows), len(profiles), n_sampled, skipped_no_candidates, n_dropped_no_canonical)
    return rows


SYNTHETIC_OUTPUT_PATH = Path(Config.OPP_PSEUDO_LABELS_SYNTHETIC_PATH)
COMBINED_OUTPUT_PATH = Path(Config.OPP_PSEUDO_LABELS_COMBINED_PATH)


def write_csv(rows: list, path: Path):
    if not rows:
        log.warning("No rows to write for %s — skipping", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def _print_stats(label: str, rows: list):
    if not rows:
        print(f"\n{label}: 0 rows")
        return
    n_relevant = sum(r["pseudo_label_relevant"] for r in rows)
    scores = sorted(r["pseudo_label_reg"] for r in rows)
    print(f"\n{label}: {len(rows)} rows")
    print(f"  {n_relevant} relevant ({n_relevant/len(rows):.1%}) at threshold {RELEVANCE_THRESHOLD}")
    print(f"  combined_score range: {scores[0]:.3f} – {scores[-1]:.3f}, median {scores[len(scores)//2]:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    real_rows = generate()
    write_csv(real_rows, OUTPUT_PATH)

    synthetic_rows = generate_synthetic()
    write_csv(synthetic_rows, SYNTHETIC_OUTPUT_PATH)

    combined = real_rows + synthetic_rows
    write_csv(combined, COMBINED_OUTPUT_PATH)

    _print_stats("REAL", real_rows)
    _print_stats("SYNTHETIC", synthetic_rows)
    _print_stats("COMBINED", combined)