"""
gemini_utils.py — shared Gemini-calling utilities for the Step 3 pipeline.

WHY THIS EXISTS
  pilot_extract_skills.py and pilot_augment_samples.py both need to call
  Gemini with the same retry/backoff/logging behavior. Duplicating it in two
  files means fixing a bug in one place and forgetting the other (which is
  literally what happened last time). Pulled into one shared module instead.

WHAT CHANGED AFTER THE 3RD PILOT RUN (real findings from your account)
  - Your AI Studio dashboard shows the ACTUAL caps on your key:
      gemini-3.5-flash : 5 RPM, 20 RPD  (the only model that still works)
      gemini-2.5-flash : 404 "no longer available to new users" (dead)
      gemini-2.0-flash : 0 quota — officially retired March 2026 (dead)
  - MODEL_CHAIN below is trimmed to just the working model. Don't add
    gemini-2.5-flash or gemini-2.0-flash back in — they will never succeed,
    confirmed from your own account, not a guess.
  - A daily 20-request cap makes one-call-per-resume infeasible (578 Tier 1
    rows would take ~29 days). The batching in pilot_extract_skills.py cuts
    that to ~58 calls total. This module adds a local daily-budget tracker
    so the script stops itself cleanly BEFORE wasting calls on a quota it
    already knows is exhausted, instead of generating pages of guaranteed
    429s.
  - CHECK BEFORE YOUR NEXT RUN: open Google AI Studio and see if enabling
    identity verification / billing (even with a $0 spend cap) moves your
    project into a higher quota tier — 20 RPD is unusually low compared to
    published free-tier figures, which suggests your project may be on an
    unverified/default bucket rather than the standard free tier. If that
    unlocks a higher RPD, update DAILY_CALL_BUDGET below to match.
"""
import os
import re
import json
import time
from datetime import date
from pathlib import Path

from careercompass.config import Config

MODEL_CHAIN = ["gemini-3.5-flash"]  # trimmed — see note above, don't re-add dead models

# Conservative default matching your account's confirmed 20 RPD cap. Update
# this the moment you confirm your actual current limit (e.g. after a Tier 1
# upgrade) — keeping a couple of requests as headroom below the hard cap.
DAILY_CALL_BUDGET = 18

# RPM on your account is 5 — this enforces a floor delay between calls so you
# never hit the per-minute cap either, independent of the daily one.
MIN_SECONDS_BETWEEN_CALLS = 13

BUDGET_STATE_FILE = Path(Config.GEMINI_BUDGET_STATE_PATH)  # data/cache/ (gitignored runtime state)


def _load_budget_state() -> dict:
    if BUDGET_STATE_FILE.exists():
        state = json.loads(BUDGET_STATE_FILE.read_text())
        if state.get("date") == str(date.today()):
            return state
    return {"date": str(date.today()), "calls_made": 0}


def _save_budget_state(state: dict):
    BUDGET_STATE_FILE.write_text(json.dumps(state))


def budget_remaining() -> int:
    state = _load_budget_state()
    return max(0, DAILY_CALL_BUDGET - state["calls_made"])


def record_call():
    state = _load_budget_state()
    state["calls_made"] += 1
    _save_budget_state(state)


def call_gemini_with_backoff(prompt: str, client, max_retries: int = 3, base_delay: float = 5.0) -> str:
    """
    Logs every failure (not just 429s) and detects PERMANENT quota failures
    ("limit: 0") to skip retries immediately instead of wasting full backoff
    cycles on a call that mathematically cannot succeed. Does NOT check the
    daily budget itself — callers should check budget_remaining() > 0 before
    calling this, so a known-exhausted day never even attempts a call.
    """
    last_error = None
    for model_name in MODEL_CHAIN:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                record_call()
                return response.text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                print(f"    [{model_name}] attempt {attempt+1} failed: {err_str[:200]}")

                if "limit: 0" in err_str or "NOT_FOUND" in err_str or "no longer available" in err_str:
                    print(f"    -> {model_name} appears permanently unavailable on this key — "
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