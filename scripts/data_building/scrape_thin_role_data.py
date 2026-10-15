"""
scrape_thin_role_data.py — scrapes REAL job postings for role classes
identified as thin/unstable by check_class_stability.py, as training
data for the role classifier. This is a DIFFERENT, stronger category of
fix than the Tier 3 Gemini-generation approach tried earlier: real
current employer skill requirements, not LLM-imagined profiles.

WHY KEYWORD-FILTERING INSTEAD OF PER-ROLE URLS: I cannot verify exact
Internshala/Unstop URL slugs for roles like "network-engineer" or
"devrel" against the live site from this environment. Guessing a slug
that happens to be wrong would silently return zero results for that
role with no indication why. Instead, this scrapes the SAME general
job-listing feeds jobs.py already confirmed work, then classifies each
posting by matching its TITLE against a synonym list per role — robust
to whichever exact URL structure the site actually uses, and every
role's match count is reported individually so a bad synonym list is
immediately visible rather than silently returning nothing.

REUSES, DOES NOT REIMPLEMENT: _session(), _card_scope(), _skills_in_scope()
are imported directly from scrapers.jobs — the same battle-tested card-
parsing logic already validated against live Internshala/Unstop pages
(see jobs.py's own docstrings on why _card_scope replaced an earlier,
broken ancestor-climbing heuristic). Re-deriving this logic here would
risk reintroducing bugs jobs.py already fixed.

DISTRIBUTION MISMATCH, STATED PLAINLY: a job posting describes what an
EMPLOYER wants, not what one candidate actually HAS — this is real,
useful signal, but not identical in character to a resume. Rows from
this source are tagged with their own distinct provenance
(source="job_posting_scraped_v1") specifically so their effect on
per-class F1 can be checked empirically (same repeated-CV discipline
as every other data change this session) rather than assumed helpful.

SCHEMA NOTE: job postings don't reveal a candidate's degree or years of
experience — degree_level/degree_field/years_experience are set to
"unknown" for every row from this source, which the model already
handles natively (see train_role_classifier.py's NUM_FEATURES handling
of NaN, and CAT_FEATURES' existing "unknown" values elsewhere in the
dataset).

USAGE
  python scrape_thin_role_data.py                       # thin roles only (default, recommended)
  python scrape_thin_role_data.py --roles data-analyst,product-manager
  python scrape_thin_role_data.py --all-roles           # attempt all 22 — NOT recommended;
                                                        # large, established classes don't need this
"""
import argparse
import csv
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from careercompass.config import Config
from careercompass.scrapers import circuit_breaker
from careercompass.scrapers.jobs import _session, _card_scope, _skills_in_scope, HEADERS  # noqa: reuse, see docstring

logger = logging.getLogger(__name__)

OUTPUT_CSV = str(Config.INTERIM_DATA_DIR / "thin_roles_job_postings_v1.csv")
SOURCE_TAG = "job_posting_scraped_v1"

# Classes flagged as thin/unstable by check_class_stability.py (support
# under ~100 rows, or std >= 0.10 in the repeated-CV stability analysis).
# This is the DEFAULT, evidence-based target set — not "all 22 classes",
# per the finding that large established classes (backend, data-engineer,
# frontend, devops, cyber-security) already have hundreds of stable rows
# and gain little from this.
DEFAULT_THIN_ROLES = [
    "data-analyst", "product-manager", "network-engineer", "devsecops",
    "game-developer", "mobile", "engineering-manager", "software-architect",
    "blockchain", "technical-writer", "devrel", "postgresql-dba", "mlops",
    "ux-design", "full-stack",
]

# Title-matching synonyms per role_slug. Kept deliberately NARROW for
# roles with an existing confusable neighbor (e.g. mlops vs ml-cluster)
# to avoid manufacturing MORE of the confusion this whole diagnostic
# chain was trying to reduce, rather than broad/greedy matching that
# would maximize row count at the cost of label quality.
ROLE_TITLE_SYNONYMS = {
    "data-analyst": ["data analyst", "data analytics"],
    "product-manager": ["product manager", "product management"],
    "network-engineer": ["network engineer", "network administrator"],
    "devsecops": ["devsecops"],
    "game-developer": ["game developer", "unity developer", "unreal developer"],
    "mobile": ["android developer", "ios developer", "mobile developer",
               "flutter developer", "react native developer"],
    "engineering-manager": ["engineering manager"],
    "software-architect": ["software architect", "solutions architect"],
    "blockchain": ["blockchain developer", "solidity developer", "web3 developer"],
    "technical-writer": ["technical writer", "technical writing"],
    "devrel": ["developer relations", "developer advocate", "devrel"],
    "postgresql-dba": ["database administrator", "postgresql dba", "postgres dba"],
    "mlops": ["mlops"],  # deliberately narrow — see module docstring
    "ux-design": ["ux designer", "ui/ux designer", "product designer", "ux/ui designer"],
    "full-stack": ["full stack developer", "full-stack developer", "fullstack developer"],
    # Remaining 7 classes have real, adequate volume already — synonyms
    # provided for completeness (--all-roles), not recommended by default.
    "backend": ["backend developer", "backend engineer"],
    "frontend": ["frontend developer", "front-end developer"],
    "devops": ["devops engineer"],
    "data-engineer": ["data engineer"],
    "ml-cluster": ["machine learning engineer", "ai engineer", "data scientist"],
    "qa": ["qa engineer", "test engineer", "sdet", "quality assurance"],
    "cyber-security": ["cybersecurity", "security engineer", "security analyst"],
}


def _normalize_tag(t: str) -> str:
    return t.strip().lower().replace(" ", "-").replace(".", "")


def classify_title(title: str, target_roles: list) -> str:
    """Returns the FIRST matching role_slug for a posting title, or None.
    Checked in target_roles' given order — list narrower/more specific
    synonym sets first if you extend ROLE_TITLE_SYNONYMS, since a title
    could plausibly match more than one role's keyword list."""
    title_l = title.lower()
    for role in target_roles:
        for synonym in ROLE_TITLE_SYNONYMS.get(role, []):
            if synonym in title_l:
                return role
    return None


# ── Internshala: general job listing sweep ──────────────────────────────
# Reuses the SAME category slugs jobs.py's INTERNSHALA_JOB_CATS already
# uses successfully, plus a couple of broader ones, to maximize the
# variety of postings seen for keyword-classification — NOT betting on
# a specific thin-role URL slug existing.
SWEEP_CATEGORIES = [
    "computer-science", "software-development", "web-development",
    "data-science", "machine-learning", "python", "backend-developer",
    "frontend-developer", "product-management", "quality-assurance",
    "cyber-security", "database", "cloud-computing", "mobile-app-development",
]


def scrape_internshala_sweep(target_roles: list, max_pages_per_cat: int = 2) -> list:
    results = []
    seen_urls = set()
    sess = _session()

    for cat_slug in SWEEP_CATEGORIES:
        if circuit_breaker.is_open("https://internshala.com"):
            logger.warning("[Internshala sweep] circuit open — stopping")
            break
        for base_path in [
            f"https://internshala.com/fresher-jobs/{cat_slug}-jobs",
            f"https://internshala.com/jobs/{cat_slug}-jobs",
        ]:
            for page in range(1, max_pages_per_cat + 1):
                url = base_path if page == 1 else f"{base_path}/page-{page}/"
                try:
                    resp = sess.get(url, timeout=15)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "lxml")

                    h2_tags = [h2 for h2 in soup.find_all("h2")
                               if h2.find("a", href=re.compile(r"/job/detail/|/jobs/detail/"))]
                    if not h2_tags:
                        break

                    for h2 in h2_tags:
                        link_el = h2.find("a", href=re.compile(r"/job/detail/|/jobs/detail/"))
                        title = h2.get_text(strip=True)
                        if not link_el or not title:
                            continue
                        role = classify_title(title, target_roles)
                        if not role:
                            continue  # doesn't match any target role's synonyms — skip

                        href = link_el["href"]
                        full_url = href if href.startswith("http") else "https://internshala.com" + href
                        if full_url in seen_urls:
                            continue
                        seen_urls.add(full_url)

                        scope = _card_scope(h2)
                        skills = _skills_in_scope(scope, limit=8)

                        company_el = next((e for e in scope if "company" in " ".join(e.get("class", [])).lower()), None)
                        company = company_el.get_text(strip=True) if company_el else ""

                        results.append({
                            "role_slug": role, "title": title, "company": company,
                            "url": full_url, "skills": skills,
                        })

                    if not soup.select_one(f"a[href*='page-{page + 1}']"):
                        break
                    time.sleep(0.4)

                except Exception as e:
                    logger.error(f"[Internshala sweep] {url}: {e}")
                    break
        time.sleep(0.3)

    logger.info(f"[Internshala sweep] {len(results)} matching postings found")
    return results


# ── Unstop: general job feed ─────────────────────────────────────────────
UNSTOP_GENERAL_URLS = [
    "https://api.unstop.com/job",
    "https://api.unstop.com/jobs/software-development-jobs",
    "https://api.unstop.com/jobs/data-science-jobs",
]


def scrape_unstop_sweep(target_roles: list) -> list:
    results = []
    seen_urls = set()
    sess = _session()

    for page_url in UNSTOP_GENERAL_URLS:
        if circuit_breaker.is_open(page_url):
            logger.warning("[Unstop sweep] circuit open — stopping")
            break
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = soup.find_all("a", href=re.compile(r"(api\.unstop\.com|unstop\.com)/jobs?/[^/]+$"))
            for card in cards:
                h2 = card.find("h2")
                if not h2:
                    continue
                title = h2.get_text(strip=True)
                if not title or len(title) < 3:
                    continue
                role = classify_title(title, target_roles)
                if not role:
                    continue

                href = card["href"]
                full_url = href if href.startswith("http") else "https://unstop.com" + href
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                all_texts = [t.strip() for t in card.stripped_strings
                             if t.strip() and t.strip() != title and len(t.strip()) > 1]
                company = all_texts[0] if all_texts else ""

                results.append({
                    "role_slug": role, "title": title, "company": company,
                    "url": full_url, "skills": [],  # Unstop cards rarely expose a skill-tag list directly
                })
            time.sleep(0.4)

        except Exception as e:
            logger.error(f"[Unstop sweep] {page_url}: {e}")

    logger.info(f"[Unstop sweep] {len(results)} matching postings found")
    return results


def normalize_row(posting: dict) -> dict:
    skill_tags = [_normalize_tag(s) for s in posting["skills"] if s.strip()]
    preview = f"{posting['title']} at {posting['company']}".strip()
    return {
        "source": SOURCE_TAG,
        "original_category": posting["title"],
        "role_slug": posting["role_slug"],
        "original_text_preview": preview[:200],
        "skill_tags": skill_tags,
        "degree_level": "unknown",
        "degree_field": "unknown",
        "years_experience": "unknown",
        "extraction_error": "",
    }


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default=None, help="comma-separated role_slugs; defaults to the thin-class set")
    ap.add_argument("--all-roles", action="store_true", help="attempt all 22 classes — not recommended by default")
    args = ap.parse_args()

    if args.all_roles:
        target_roles = list(ROLE_TITLE_SYNONYMS.keys())
    elif args.roles:
        target_roles = [r.strip() for r in args.roles.split(",")]
    else:
        target_roles = DEFAULT_THIN_ROLES

    print(f"Target roles: {target_roles}\n")
    circuit_breaker.reset()

    internshala_results = scrape_internshala_sweep(target_roles)
    unstop_results = scrape_unstop_sweep(target_roles)
    all_postings = internshala_results + unstop_results

    if not all_postings:
        print("\nNo matching postings found for ANY target role. Before assuming the "
              "sites are unreachable, check: (1) is circuit_breaker reporting the host "
              "as open/tripped in the logs above, (2) do SWEEP_CATEGORIES/UNSTOP_GENERAL_URLS "
              "still match the live site's current URL structure — these were inherited "
              "from jobs.py's already-working patterns but sites change over time.")
        return

    rows = [normalize_row(p) for p in all_postings]

    print("\nMatches per role:")
    from collections import Counter
    counts = Counter(r["role_slug"] for r in rows)
    for role in target_roles:
        n = counts.get(role, 0)
        flag = "  <-- ZERO MATCHES, check synonyms/URL coverage for this role" if n == 0 else ""
        print(f"  {role:25s} {n:4d}{flag}")

    fieldnames = ["source", "original_category", "role_slug", "original_text_preview",
                  "skill_tags", "degree_level", "degree_field", "years_experience",
                  "extraction_error"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print("\nSample rows for manual review (same discipline as every prior data script —")
    print("never trust scraped/generated data blindly before merging):")
    for r in rows[:8]:
        print(f"  [{r['role_slug']}] {r['original_category']!r} -> {r['skill_tags']}")

    empty_skills = sum(1 for r in rows if not r["skill_tags"])
    if empty_skills:
        print(f"\nNOTE: {empty_skills}/{len(rows)} rows have NO skill tags at all "
              f"(common for Unstop cards, which don't always expose a tag list in "
              f"the card itself) — these rows carry only the role label with no "
              f"feature content and should be DROPPED before merging, not kept as "
              f"zero-signal rows.")


if __name__ == "__main__":
    main()