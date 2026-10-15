"""
pilot_extract_skills.py — Phase 2 pilot: extract normalized skill tags from a
small stratified sample of BOTH resume datasets via Gemini, for manual review
BEFORE running the full ~578-row batch.

Run this from inside your project (so it can reuse GEMINI_API_KEY from your
existing .env) after copying it next to your other scripts.

WHAT THIS DOES
  1. Loads UpdatedResumeDataSet.csv, fixes the encoding corruption present in
     ~75% of rows (verified: bytes were UTF-8, misread as latin-1), filters
     to only the categories that map to a ROLE_SLUG we're keeping, and drops
     everything else (Sales, HR, Advocate, etc. — no roadmap.sh equivalent).
  2. Loads candidate_job_role_dataset.csv, filters/drops the same way
     (drops HR, Marketing, Designer, C#/PHP Developer, and the two n=1 noise
     rows — Finance, Video Game Designer).
  3. Takes a small stratified sample (2 rows per source category) from each,
     NOT the full dataset — this is a pilot to sanity-check extraction
     quality before burning the full batch of Gemini calls.
  4. For Tier 1 (free-text resumes): calls Gemini to extract skill_tags,
     degree_level, degree_field, years_experience — using "unknown" for
     anything it can't confidently infer (never blank/null).
  5. For Tier 2 (already-structured skills column): light normalization only
     — no Gemini call needed, it's already a clean comma-separated list.
  6. Writes everything to pilot_output.csv for manual review — original
     text/label next to extracted output, so you can eyeball whether the
     extraction is trustworthy before scaling up.

WHAT YOU NEED TO CHANGE BEFORE RUNNING
  - GEMINI_MODEL_CHAIN below should match the fallback chain your onboarding
    module already uses (gemini-3.5-flash -> gemini-2.5-flash -> gemini-2.x).
  - If you already have a Gemini client wrapper in your onboarding blueprint,
    prefer calling THAT instead of the raw client setup below, so extraction
    behavior stays consistent across the app. This script sets up its own
    client only so it can run standalone for the pilot.
  - Install the SDK if not already present: pip install google-genai
"""
import os
import re
import json
import time
import csv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from careercompass.config import Config  # also loads .env
load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
TIER1_CSV = str(Config.TIER1_RAW_CSV)
TIER2_CSV = str(Config.TIER2_RAW_CSV)
OUTPUT_CSV = str(Config.INTERIM_DATA_DIR / "pilot_output.csv")
SAMPLES_PER_CATEGORY = 2  # keep pilot small on purpose

GEMINI_MODEL_CHAIN = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Tier 1: source Category -> ROLE_SLUG. Categories not listed here are dropped.
TIER1_MAP = {
    "DevOps Engineer": "devops",
    "Python Developer": "backend",
    "Java Developer": "backend",
    "Hadoop": "data-engineer",
    "ETL Developer": "data-engineer",
    "Database": "data-engineer",
    "Network Security Engineer": "cyber-security",
    "Blockchain": "blockchain",
    "Testing": "qa",
    "Automation Testing": "qa",
    "Web Designing": "frontend",
}

# Tier 2: source job_role -> ROLE_SLUG. job_roles not listed here are dropped.
# NOTE: "Mobile Developer" merges android+ios (dataset can't distinguish them).
# NOTE: "Data Scientist" + "AIML" merge into ml-cluster (skill sets are near-
#       identical in this dataset — see pilot findings from earlier analysis).
TIER2_MAP = {
    "Frontend Developer": "frontend",
    "Web Developer": "frontend",
    "Backend Developer": "backend",
    "DevOps Engineer": "devops",
    "Kubernetes Operations Engineer": "devops",
    "Full Stack Java Developer": "full-stack",
    "Full Stack Python Developer": "full-stack",
    "Game Developer": "game-developer",
    "Software Project Manager": "product-manager",
    "Mobile Developer": "mobile",              # merged android/ios
    "Blockchain Developer": "blockchain",
    "Cybersecurity Engineer": "cyber-security",
    "Data Analyst": "data-analyst",
    "Data Scientist": "ml-cluster",             # merged ai-eng/ai-ds/ml
    "AIML": "ml-cluster",
}

EXTRACTION_PROMPT = """You are extracting structured features from a resume for a machine learning pipeline. Return ONLY valid JSON, no markdown fences, no preamble.

Resume text:
---
{resume_text}
---

Return exactly this JSON shape:
{{
  "skill_tags": ["list", "of", "normalized", "lowercase", "skill", "slugs"],
  "degree_level": "bachelors" | "masters" | "phd" | "unknown",
  "degree_field": "short field name, e.g. computer-science" | "unknown",
  "years_experience": integer or "unknown"
}}

Rules:
- skill_tags should be normalized to lowercase, hyphenated slugs where multi-word
  (e.g. "Machine Learning" -> "machine-learning", "Node.js" -> "nodejs").
- Only include skills actually evidenced in the text — do not infer skills that
  aren't mentioned.
- If a field cannot be confidently determined from the text, use "unknown" —
  never leave it blank or null.
"""


def fix_mojibake(text: str) -> str:
    """
    Verified fix: ~75% of UpdatedResumeDataSet rows were UTF-8 bytes that got
    misread as latin-1 during the original dataset's creation. This round-trip
    (encode as latin-1, decode as utf-8) reverses that. Falls back to the
    original text if the round-trip isn't valid (should be rare/never here,
    but kept defensive rather than assuming).
    """
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def load_tier1_sample(path: str, n_per_category: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Category"].isin(TIER1_MAP.keys())].copy()
    df["Resume"] = df["Resume"].apply(fix_mojibake)
    df["role_slug"] = df["Category"].map(TIER1_MAP)
    df["source"] = "tier1_real"
    sampled = (
        df.groupby("Category", group_keys=False)[df.columns]
        .apply(lambda g: g.sample(min(n_per_category, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    return sampled


def normalize_degree_level(qualification: str) -> str:
    """
    FIX (pilot review): original version only checked "Bachelor"/"Master"
    substrings and missed "PhD"/"Doctorate" entirely — e.g. an AIML row with
    degree_field="PhD in Artificial Intelligence" was coming out as
    degree_level="unknown". Checks all three now.
    """
    q = str(qualification).lower()
    if "phd" in q or "doctorate" in q:
        return "phd"
    if "master" in q:
        return "masters"
    if "bachelor" in q:
        return "bachelors"
    return "unknown"


def load_tier2_sample(path: str, n_per_category: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["job_role"].isin(TIER2_MAP.keys())].copy()
    # known dirty value: "Entry," with trailing comma
    df["experience_level"] = df["experience_level"].str.rstrip(",").str.strip()
    df["role_slug"] = df["job_role"].map(TIER2_MAP)
    df["source"] = "tier2_synthetic"
    sampled = (
        df.groupby("job_role", group_keys=False)[df.columns]
        .apply(lambda g: g.sample(min(n_per_category, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    return sampled


def normalize_tier2_skills(skills_str: str) -> list:
    """Tier 2 is already a clean comma-separated list — just slugify."""
    return [s.strip().lower().replace(" ", "-").replace(".", "") for s in skills_str.split(",")]


def call_gemini_with_backoff(prompt: str, client, max_retries: int = 3, base_delay: float = 5.0) -> str:
    """
    FIX (2nd pilot run): the original version only printed/logged on 429
    errors — any other failure (invalid model name, deprecated model, auth
    issue) hit a silent `break` with zero visibility, which masked what was
    actually happening with the first models in the chain.

    FIX: also detects PERMANENT quota failures (the API's own "limit: 0"
    marker, meaning this key has zero free-tier quota for this model —
    no amount of waiting will ever fix that) and skips retries for that
    model immediately instead of wasting 3 full backoff cycles on a call
    that mathematically cannot succeed.
    """
    last_error = None
    for model_name in GEMINI_MODEL_CHAIN:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                print(f"    [{model_name}] attempt {attempt+1} failed: {err_str[:200]}")

                if "limit: 0" in err_str:
                    print(f"    -> {model_name} has ZERO free-tier quota on this key — "
                          f"skipping remaining retries, moving to next model")
                    break

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)", err_str)
                    delay = float(match.group(1)) + 1.0 if match else base_delay * (attempt + 1)
                    print(f"    rate limited on {model_name}, waiting {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    break  # non-rate-limit, non-quota error — no point retrying same model
    raise RuntimeError(f"All models/retries exhausted. Last error: {last_error}")


def extract_tier1_via_gemini(resume_text: str, client) -> dict:
    """
    Calls Gemini (with backoff/retry, see call_gemini_with_backoff). Returns
    a dict matching the EXTRACTION_PROMPT schema, or an error marker if all
    retries/models fail — errors are captured per-row so one bad call doesn't
    kill the whole batch.
    """
    prompt = EXTRACTION_PROMPT.format(resume_text=resume_text[:6000])  # guard token budget

    try:
        raw = call_gemini_with_backoff(prompt, client)
        # Gemini sometimes prepends conversational text before the JSON fence —
        # startswith("```") checks are fragile (known issue) — find the JSON
        # object directly instead.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in response: {raw[:200]}")
        return json.loads(raw[start : end + 1])
    except Exception as e:
        return {"error": str(e), "skill_tags": [], "degree_level": "unknown",
                "degree_field": "unknown", "years_experience": "unknown"}


def main():
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY not found in environment. Set it in .env before running."
        )

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("Loading and sampling Tier 1 (real resumes)...")
    tier1 = load_tier1_sample(TIER1_CSV, SAMPLES_PER_CATEGORY)
    print(f"  {len(tier1)} rows sampled across {tier1['Category'].nunique()} categories")

    print("Loading and sampling Tier 2 (synthetic, structured)...")
    tier2 = load_tier2_sample(TIER2_CSV, SAMPLES_PER_CATEGORY)
    print(f"  {len(tier2)} rows sampled across {tier2['job_role'].nunique()} categories")

    # FIX (2nd pilot run): checkpoint incrementally so a slow/interrupted run
    # (these can now take minutes per row under heavy rate limiting) doesn't
    # lose all progress, and so re-running the script skips rows that already
    # succeeded rather than re-spending quota on them.
    fieldnames = ["source", "original_category", "role_slug", "original_text_preview",
                  "skill_tags", "degree_level", "degree_field", "years_experience",
                  "extraction_error"]

    already_done = set()
    if Path(OUTPUT_CSV).exists():
        prior = pd.read_csv(OUTPUT_CSV)
        # only skip rows that succeeded — always retry previous errors
        ok_prior = prior[prior["extraction_error"].isna() | (prior["extraction_error"] == "")]
        already_done = set(zip(ok_prior["source"], ok_prior["original_category"], ok_prior["original_text_preview"]))
        print(f"Found existing {OUTPUT_CSV} with {len(ok_prior)} successful rows — will skip those.")

    write_header = not Path(OUTPUT_CSV).exists()
    out_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()

    print("\nExtracting Tier 1 via Gemini (this makes real API calls)...")
    for i, row in tier1.iterrows():
        key = ("tier1_real", row["Category"], row["Resume"][:200])
        if key in already_done:
            print(f"  [{i+1}/{len(tier1)}] {row['Category']} -> {row['role_slug']} (already done, skipping)")
            continue

        print(f"  [{i+1}/{len(tier1)}] {row['Category']} -> {row['role_slug']}")
        extracted = extract_tier1_via_gemini(row["Resume"], client)
        result = {
            "source": "tier1_real",
            "original_category": row["Category"],
            "role_slug": row["role_slug"],
            "original_text_preview": row["Resume"][:200],
            "skill_tags": extracted.get("skill_tags", []),
            "degree_level": extracted.get("degree_level", "unknown"),
            "degree_field": extracted.get("degree_field", "unknown"),
            "years_experience": extracted.get("years_experience", "unknown"),
            "extraction_error": extracted.get("error", ""),
        }
        writer.writerow(result)
        out_file.flush()  # write immediately, don't wait for buffer/program exit
        # FIX (pilot review): 0.5s was too aggressive and triggered a burst
        # rate limit mid-run. 2s is more conservative — adjust based on your
        # actual quota tier if you have one.
        time.sleep(2.0)

    print("\nNormalizing Tier 2 (no API calls needed, already structured)...")
    for i, row in tier2.iterrows():
        key = ("tier2_synthetic", row["job_role"], row["skills"])
        if key in already_done:
            continue
        result = {
            "source": "tier2_synthetic",
            "original_category": row["job_role"],
            "role_slug": row["role_slug"],
            "original_text_preview": row["skills"],
            "skill_tags": normalize_tier2_skills(row["skills"]),
            "degree_level": normalize_degree_level(row["qualification"]),
            "degree_field": row["qualification"],
            "years_experience": row["experience_level"],
            "extraction_error": "",
        }
        writer.writerow(result)
        out_file.flush()

    out_file.close()

    out_df = pd.read_csv(OUTPUT_CSV)
    print(f"\nDone. {OUTPUT_CSV} now has {len(out_df)} total rows — review manually before scaling up.")

    n_errors = (out_df["extraction_error"].notna() & (out_df["extraction_error"] != "")).sum()
    if n_errors:
        print(f"WARNING: {n_errors} rows still have extraction errors — check those, "
              f"then just rerun this script to retry only the failed ones.")


if __name__ == "__main__":
    main()