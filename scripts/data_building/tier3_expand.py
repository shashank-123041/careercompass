"""
tier3_expand.py — grows the 9 Tier 3 role classes (see train_role_classifier.py's
TIER3_ROLES) with fresh, web-grounded synthetic samples, following the SAME
discipline documented for the original tier3_generated.csv: "web-search-grounded
synthetic for zero-coverage roles" — grounded, not hallucinated.

WHY A NEW BATCH: role_classifier_meta.json shows these 9 classes still sitting
at 6-12 rows each (vs up to 660 for the largest class) — a comparable published
benchmark study (2,483 resumes / 24 categories) found EVERY model it tested,
regardless of architecture, hit its floor on exactly this kind of minority-
class scarcity. That's the actual lever on macro F1, not model choice.

GROUNDING SOURCE: SEED_GROUNDING below is NOT pulled from the existing (thin)
Tier 1/2 dataset — for these 9 roles that data barely exists, which is the
whole problem. Instead each entry is a paraphrased summary of current (2026)
job-description/skill-requirement research per role, gathered and condensed
by hand before writing this script — not the dataset's own sparse seed
examples, and not Gemini's unguided assumptions about what these roles need.

BUDGET: reuses gemini_utils.py's shared daily-call tracker (18/day, 13s
between calls, same account-confirmed limits pilot_extract_skills.py and
resume_parser.py already respect) rather than a second, independent limiter.
One call per role = 9 calls total for a full run — well inside budget, with
headroom to retry failures same-day if needed.

OUTPUT SCHEMA: matches final_dataset_v3.csv's columns exactly (source,
original_category, role_slug, original_text_preview, skill_tags,
degree_level, degree_field, years_experience, extraction_error) so this
can be concatenated directly onto the existing dataset for retraining —
tagged with a NEW source value ("tier3_websearch_v2") so it's always
distinguishable from the original tier3_generated.csv batch, never
silently merged as if equivalent.
"""
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from careercompass.config import Config
from careercompass.resume import gemini_utils as gu  # shared budget tracker + backoff — see module docstring

OUTPUT_CSV = str(Config.INTERIM_DATA_DIR / "tier3_expanded_v2.csv")
SAMPLES_PER_ROLE = 20  # ~2-3x the current 6-12 rows/class for these 9 classes
SOURCE_TAG = "tier3_websearch_v2"

# Paraphrased from current (2026) job-description/skill-requirement research
# per role — see module docstring. Not verbatim from any single source.
SEED_GROUNDING = {
    "devrel": (
        "Developer Relations / Developer Advocate. Sits between engineering, "
        "product, and the external developer community. Writes technical "
        "content (blog posts, tutorials, docs), builds and maintains SDK/API "
        "code samples, speaks at conferences and meetups, runs developer "
        "forums/Discord/community channels, gathers developer feedback and "
        "routes it back to product/engineering. Increasingly a hybrid "
        "engineering+communication role at AI/dev-tools companies."
    ),
    "devsecops": (
        "DevSecOps Engineer. Integrates security into CI/CD pipelines rather "
        "than treating it as a separate gate. Uses SAST/DAST tools (SonarQube, "
        "Snyk, Checkmarx), container security (Docker, Kubernetes hardening), "
        "infrastructure-as-code (Terraform, Ansible), secrets scanning, "
        "vulnerability assessment (Nessus), cloud security across AWS/Azure/GCP, "
        "and compliance frameworks (PCI-DSS, HIPAA, GDPR). Scripts in "
        "Python/Bash to automate security checks in the pipeline."
    ),
    "mlops": (
        "MLOps Engineer. Operationalizes ML models: deployment, monitoring, "
        "scaling, retraining pipelines. Tools: MLflow, Kubeflow, Airflow, "
        "Docker, Kubernetes, Terraform, cloud ML platforms (SageMaker, Vertex "
        "AI), data/model versioning (DVC), CI/CD for ML (GitHub Actions, "
        "Jenkins), model monitoring/observability. Builds on a DevOps "
        "foundation (CI/CD, containers, monitoring) plus ML-pipeline-specific "
        "tooling. Strong Python required."
    ),
    "network-engineer": (
        "Network Engineer. Designs, implements, and troubleshoots network "
        "infrastructure — routers, switches, WAN/LAN, VLANs, routing "
        "protocols (OSPF, BGP, EIGRP), NAT/ACLs. CCNA/CCNP certification is "
        "the standard credential path. Increasingly expected to know network "
        "automation/scripting (Python, Ansible), SD-WAN, cloud networking, "
        "and basic network security (firewalls, VPNs)."
    ),
    "postgresql-dba": (
        "PostgreSQL Database Administrator. Installs, configures, tunes, and "
        "maintains PostgreSQL database servers. Core duties: backup/recovery "
        "strategy, replication and high availability, query/performance "
        "tuning, access control and security, capacity planning, disaster "
        "recovery. Often also touches related tools: AWS RDS/Aurora "
        "PostgreSQL, monitoring (CloudWatch, Datadog), scripting (Bash/"
        "Python) for automation, and sometimes adjacent databases (MySQL, "
        "Snowflake) in mixed environments."
    ),
    "software-architect": (
        "Software Architect. Makes high-level structural decisions for a "
        "system: microservices vs monolith, database choice, cloud "
        "infrastructure, API design. Guides engineering teams on "
        "implementation, reviews designs, balances trade-offs (scalability, "
        "cost, maintainability). Typically 5+ years of hands-on software "
        "engineering experience first. Familiar with UML, cloud platforms "
        "(AWS/Azure certifications common), and system design patterns."
    ),
    "technical-writer": (
        "Technical Writer. Produces user manuals, API references, how-to "
        "guides, and knowledge-base articles that translate complex "
        "technical systems into clear documentation. Tools: Markdown, "
        "Confluence, GitBook, MadCap Flare, Git for version control, "
        "sometimes Figma/Snagit for visuals. Works closely with engineers "
        "and product managers to gather information; increasingly expected "
        "to understand APIs, SaaS concepts, and docs-as-code workflows."
    ),
    "ux-design": (
        "UX/UI Designer. Designs user flows, wireframes, and high-fidelity "
        "prototypes, primarily in Figma (industry standard), also Sketch/"
        "Adobe XD. Conducts user research and usability testing, builds and "
        "maintains design systems (components, auto-layout, variants). "
        "Collaborates closely with product managers and developers; some "
        "roles expect basic HTML/CSS familiarity to work near implementation."
    ),
    "engineering-manager": (
        "Engineering Manager. Splits time across delivery execution (sprint "
        "planning, unblocking the team, code review), people management "
        "(1:1s, mentoring, hiring, performance), and technical strategy "
        "(architecture decisions, design review) — without necessarily "
        "coding daily. Strong communication skills for translating technical "
        "concepts to non-technical stakeholders. Usually promoted from a "
        "senior/staff engineering background, not hired as a pure manager."
    ),
}

AUGMENTATION_PROMPT = """You are generating realistic, varied synthetic candidate profiles for a machine learning training set — for a role of "{role_display}".

Here is a grounded summary of what this role actually involves, based on current job-description and skill-requirement research (use this as your factual anchor — do not invent responsibilities or tools that contradict it):
{grounding}

Generate {n} DIFFERENT, realistic candidate profiles for this role. Requirements for genuine diversity:
- Vary skill subset size and composition — not everyone should have the same skills.
- Include specific tools/technologies genuinely relevant to this role (drawn from or consistent with the grounding above), not generic buzzwords.
- Vary experience_level across entry/mid/senior realistically (not all the same level).
- Vary degree_level and degree_field realistically (not all identical — some technical writers/UX designers may have non-CS degrees, for instance).
- Each profile should be plausibly distinct from the others — avoid near-duplicate outputs.

Return ONLY a valid JSON array, no markdown fences, no preamble. Each element:
{{
  "skill_tags": ["lowercase", "hyphenated", "skill", "slugs"],
  "degree_level": "bachelors" | "masters" | "phd" | "unknown",
  "degree_field": "short field name",
  "years_experience": integer
}}
"""


def parse_json_array(raw: str) -> list:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found in environment. Set it in .env before running.")

    from google import genai
    client = genai.Client(api_key=api_key)

    roles = list(SEED_GROUNDING.keys())
    remaining_budget = gu.budget_remaining()
    print(f"Daily Gemini budget remaining: {remaining_budget} calls. "
          f"This run needs up to {len(roles)} calls (1 per role).")
    if remaining_budget < len(roles):
        print(f"WARNING: budget may run out mid-run — will stop cleanly and "
              f"report which roles were NOT completed, rather than burn "
              f"remaining calls on guaranteed 429s.")

    results = []
    for role_slug in roles:
        if gu.budget_remaining() <= 0:
            print(f"Budget exhausted — stopping before '{role_slug}'. "
                  f"Rerun tomorrow to pick up remaining roles (already-written "
                  f"rows in {OUTPUT_CSV} are untouched).")
            break

        print(f"Generating {SAMPLES_PER_ROLE} web-grounded samples for '{role_slug}'...")
        prompt = AUGMENTATION_PROMPT.format(
            role_display=role_slug.replace("-", " ").title(),
            grounding=SEED_GROUNDING[role_slug],
            n=SAMPLES_PER_ROLE,
        )
        try:
            raw = gu.call_gemini_with_backoff(prompt, client)
            samples = parse_json_array(raw)
            for s in samples:
                results.append({
                    "source": SOURCE_TAG,
                    "original_category": role_slug.replace("-", " ").title(),
                    "role_slug": role_slug,
                    "original_text_preview": "",  # no source text — fully synthetic, grounded via prompt only
                    "skill_tags": s.get("skill_tags", []),
                    "degree_level": s.get("degree_level", "unknown"),
                    "degree_field": s.get("degree_field", "unknown"),
                    "years_experience": s.get("years_experience", "unknown"),
                    "extraction_error": "",
                })
            print(f"  got {len(samples)} samples "
                  f"(budget remaining: {gu.budget_remaining()})")
        except Exception as e:
            print(f"  FAILED for {role_slug}: {e}")
        time.sleep(2.0)  # pacing between roles, on top of call_gemini_with_backoff's own MIN_SECONDS_BETWEEN_CALLS

    if not results:
        print("\nNo rows generated — check GEMINI_API_KEY and budget before rerunning.")
        return

    write_header = not Path(OUTPUT_CSV).exists()
    fieldnames = ["source", "original_category", "role_slug", "original_text_preview",
                  "skill_tags", "degree_level", "degree_field", "years_experience",
                  "extraction_error"]
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Wrote {len(results)} rows to {OUTPUT_CSV}")
    print("Check for genuine diversity before merging into final_dataset_v3.csv — "
          "unique skill_tags combos per role_slug, not just row count:")

    import pandas as pd
    out_df = pd.DataFrame(results)
    dup_check = out_df.groupby("role_slug")["skill_tags"].apply(lambda s: s.astype(str).nunique())
    print(dup_check)

    roles_completed = set(out_df["role_slug"].unique())
    roles_missing = set(roles) - roles_completed
    if roles_missing:
        print(f"\nNOT completed this run (budget or errors): {sorted(roles_missing)}")
        print("Rerun the script tomorrow — it appends, so completed roles won't be re-requested"
              " unless you clear the output file first.")


if __name__ == "__main__":
    main()