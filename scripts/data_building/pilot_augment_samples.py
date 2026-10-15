"""
pilot_augment_samples.py — generates a SMALL pilot batch of augmented synthetic
samples for the classes that pilot review found were too thin to trust
(candidate_job_role_dataset has only 3-7 unique skill combos per class,
repeated across 50 rows — see pilot review findings).

This is a pilot (small batch) for manual review, same discipline as
pilot_extract_skills.py — do NOT scale this to full volume until you've
eyeballed the output for genuine diversity (not just reworded duplicates).

WHAT THIS DOES
  1. For each thin class, pulls the existing skill patterns from
     candidate_job_role_dataset as *inspiration* (not a cap).
  2. Asks Gemini to generate N structured, realistic samples per class in one
     batched call (skill_tags, degree_level, degree_field, years_experience),
     explicitly instructed to vary beyond the seed patterns.
  3. Writes augmented_pilot.csv, tagged source="tier2_augmented" so it's
     always distinguishable from the original synthetic rows in later
     analysis/reporting — never silently merged as if equivalent.

FIXES INCLUDED FROM PILOT REVIEW
  - Tier 2 degree_level normalization missed "PhD" (only checked Bachelor/
    Master) — fixed here and should be back-ported into the original
    normalize step in pilot_extract_skills.py before any full run.
  - Rate limiting: pilot run hit two consecutive 429s mid-batch. The model
    fallback chain didn't help because all three models likely share one
    project-level quota — falling back just burns another attempt against
    the same limit. Fixed here with a longer base delay + backoff that
    respects the API's own suggested retry delay instead of immediately
    trying the next model.
"""
import os
import re
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from careercompass.config import Config  # also loads .env
load_dotenv()

TIER2_CSV = str(Config.TIER2_RAW_CSV)
OUTPUT_CSV = str(Config.INTERIM_DATA_DIR / "augmented_pilot.csv")
SAMPLES_PER_CLASS = 8  # pilot scale — review before scaling to full volume

GEMINI_MODEL_CHAIN = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Only classes pilot review flagged as too thin to trust as-is.
# devops/cyber-security/blockchain/backend already have a strong Tier 1 real
# backbone and don't need this.
THIN_CLASSES = {
    "full-stack": ["Full Stack Java Developer", "Full Stack Python Developer"],
    "game-developer": ["Game Developer"],
    "product-manager": ["Software Project Manager"],
    "mobile": ["Mobile Developer"],
    "ml-cluster": ["Data Scientist", "AIML"],
    "data-analyst": ["Data Analyst"],
    "frontend": ["Frontend Developer", "Web Developer"],
}

AUGMENTATION_PROMPT = """You are generating realistic, varied synthetic candidate profiles for a machine learning training set — for a role of "{role_display}".

Here are a few EXISTING example skill patterns for this role (for inspiration only — do NOT just reword these, genuinely vary beyond them):
{seed_examples}

Generate {n} DIFFERENT, realistic candidate profiles for this role. Requirements for genuine diversity:
- Vary skill subset size and composition — not everyone should have the same skills.
- Include some skills beyond the seed examples that are genuinely relevant to this role in the real world.
- Vary experience_level across entry/mid/senior realistically (not all the same level).
- Vary degree_level and degree_field realistically (not all identical).
- Each profile should be plausibly distinct from the others — avoid near-duplicate outputs.

Return ONLY a valid JSON array, no markdown fences, no preamble. Each element:
{{
  "skill_tags": ["lowercase", "hyphenated", "skill", "slugs"],
  "degree_level": "bachelors" | "masters" | "phd" | "unknown",
  "degree_field": "short field name",
  "years_experience": integer
}}
"""


def normalize_degree_level(qualification: str) -> str:
    """
    FIX from pilot review: original tier2 normalization only checked for
    "Bachelor"/"Master" substrings and missed "PhD"/"Doctorate" entirely
    (e.g. AIML rows showing degree_field="PhD in Artificial Intelligence"
    but degree_level="unknown"). This version checks all three.
    """
    q = str(qualification).lower()
    if "phd" in q or "doctorate" in q:
        return "phd"
    if "master" in q:
        return "masters"
    if "bachelor" in q:
        return "bachelors"
    return "unknown"


def get_seed_examples(df: pd.DataFrame, job_roles: list, max_examples: int = 5) -> str:
    """Pull existing unique skill patterns as inspiration, not a hard cap."""
    subset = df[df["job_role"].isin(job_roles)]
    unique_skills = subset["skills"].drop_duplicates().head(max_examples).tolist()
    return "\n".join(f"  - {s}" for s in unique_skills)


def call_gemini_with_backoff(prompt: str, client, max_retries: int = 3, base_delay: float = 5.0) -> str:
    """
    FIX from pilot review: previously cycled through the model fallback chain
    on any error, which doesn't help against a shared rate-limit quota — it
    just burns another attempt against the same limit. Now retries the SAME
    model with exponential backoff first (respecting rate limits), and only
    falls back to a different model after retries are exhausted (for actual
    model-level failures, not throughput throttling).
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
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # try to respect the API's own suggested retry delay
                    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)", err_str)
                    delay = float(match.group(1)) + 1.0 if match else base_delay * (attempt + 1)
                    print(f"    rate limited on {model_name}, waiting {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    break  # non-rate-limit error, no point retrying same model
    raise RuntimeError(f"All models/retries exhausted. Last error: {last_error}")


def parse_json_array(raw: str) -> list:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def main():
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY not found in environment. Set it in .env before running.")

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    df2 = pd.read_csv(TIER2_CSV)
    results = []

    for role_slug, source_roles in THIN_CLASSES.items():
        print(f"Generating {SAMPLES_PER_CLASS} pilot samples for '{role_slug}'...")
        seed_examples = get_seed_examples(df2, source_roles)
        prompt = AUGMENTATION_PROMPT.format(
            role_display=role_slug.replace("-", " ").title(),
            seed_examples=seed_examples,
            n=SAMPLES_PER_CLASS,
        )
        try:
            raw = call_gemini_with_backoff(prompt, client)
            samples = parse_json_array(raw)
            for s in samples:
                results.append({
                    "source": "tier2_augmented",
                    "role_slug": role_slug,
                    "skill_tags": s.get("skill_tags", []),
                    "degree_level": s.get("degree_level", "unknown"),
                    "degree_field": s.get("degree_field", "unknown"),
                    "years_experience": s.get("years_experience", "unknown"),
                })
            print(f"  got {len(samples)} samples")
        except Exception as e:
            print(f"  FAILED for {role_slug}: {e}")
        time.sleep(2.0)  # pacing between classes, not just within retries

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Wrote {len(out_df)} augmented rows to {OUTPUT_CSV}")
    print("\nCheck for genuine diversity before scaling — look at unique skill_tags")
    print("combinations per role_slug, not just row count:")
    dup_check = out_df.groupby("role_slug")["skill_tags"].apply(lambda s: s.astype(str).nunique())
    print(dup_check)


if __name__ == "__main__":
    main()