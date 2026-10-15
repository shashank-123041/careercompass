"""
taxonomy/matcher.py — Fuzzy-match free text to canonical skill/role slugs.

Used to turn each opportunity's raw "tags" list (a mix of real skills
like "React"/"Python" and non-skill descriptors like "National"/
"Flagship"/"Beginner") into skill_slug rows. Non-skill tags simply won't
clear the confidence threshold and are dropped — that's intended.
"""
import logging
from functools import lru_cache
from typing import Optional

from rapidfuzz import process, fuzz, utils

from careercompass.config import Config
from careercompass.taxonomy.skill_taxonomy import all_slugs_with_aliases

log = logging.getLogger(__name__)


def _normalize_for_scoring(s: str) -> str:
    """Preprocessor applied to both the query and every candidate before
    scoring (not before lookup — table keys/values stay as-written).
    Normalizes hyphens/slashes to spaces so a hyphenated slug like
    'machine-learning' tokenizes the same way as the space-separated
    'machine learning' a person would actually type. Without this,
    token_set_ratio treats 'machine-learning' as a single unsplittable
    token and badly under-scores it against 'machine learning' (verified:
    76.2 without this normalizer, comfortably below threshold; 100.0 with
    it)."""
    return utils.default_process(s.replace("-", " ").replace("/", " "))


@lru_cache(maxsize=1)
def _lookup_table() -> dict:
    exact = {}
    for slug, aliases in all_slugs_with_aliases().items():
        for a in aliases:
            exact[a.lower().strip()] = slug
    return exact


def match_skill(raw_text: str, threshold: float = None) -> Optional[str]:
    if not raw_text or not raw_text.strip():
        return None
    threshold = threshold if threshold is not None else Config.SKILL_MATCH_THRESHOLD
    text = raw_text.lower().strip()
    table = _lookup_table()

    if text in table:
        return table[text]

    # Exclude very short aliases (e.g. the single-char "c" for the C
    # language) from the fuzzy candidate pool. rapidfuzz's WRatio scores
    # a short string as a near-perfect "partial" match against almost any
    # longer text containing similar characters — verified concretely:
    # WRatio("cobol", "c") = 90, comfortably clearing the default 82
    # threshold and incorrectly tagging COBOL as the C language. Short
    # aliases remain reachable through the exact-match check above; they
    # just can't be reached via typos/fuzziness, which is the right
    # tradeoff since fuzzy matching on 1-2 character strings isn't
    # meaningful in either direction.
    fuzzy_candidates = [k for k in table.keys() if len(k) >= 3]

    # NOTE on scorer choice: fuzz.WRatio was tried first and rejected.
    # WRatio blends in partial_ratio, which scores a SHORT string as a
    # near-perfect match whenever it happens to appear as a substring of
    # a longer candidate — regardless of whether that containment is
    # semantically meaningful. Concretely verified against real scraper
    # tag values: 'CS' (Internshala's computer-science category label)
    # scored 90.0 against 'csharp' purely because "cs" is a literal
    # substring of "csharp"; 'Web Dev' scored 85.5 against 'amazon web
    # services' for the same reason. Both cleared the 82 threshold and
    # produced visibly wrong tags in production.
    #
    # Plain fuzz.ratio was tried next — it fixed those false positives,
    # but broke a legitimate case: 'Backend Development' (a real domain-
    # knowledge label from resume parsing) stopped matching 'backend',
    # since plain Levenshtein ratio penalizes the extra word
    # "Development" too harshly.
    #
    # fuzz.token_set_ratio (bag-of-words comparison, immune to raw
    # substring inflation) plus the hyphen-normalizing preprocessor above
    # resolves both: 'Backend Development' correctly matches 'backend'
    # (100.0, since "backend" is a subset of its tokens) while every
    # false-positive case from the WRatio era stays comfortably below
    # threshold (worst case 'cs'->'css' at 80.0, still under 82).
    result = process.extractOne(
        text, fuzzy_candidates, scorer=fuzz.token_set_ratio, processor=_normalize_for_scoring
    )
    if result is None:
        return None
    matched_text, score, _ = result
    if score >= threshold:
        return table[matched_text]

    log.debug("No confident skill match for %r (best=%r score=%.1f)",
              raw_text, matched_text, score)
    return None


def match_skills(raw_texts: list, threshold: float = None) -> list:
    seen, out = set(), []
    for t in raw_texts:
        slug = match_skill(t, threshold)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out