"""
ml/bootstrap_scheduler.py — runs opportunity_bootstrap.py's pseudo-label
generation periodically inside the Flask app, the SAME architectural
pattern as sync/opportunity_engine.py (state dict, get_state(),
start_scheduler(), trigger_manual(), everything wrapped so a failure
anywhere degrades to a logged non-fatal outcome, never a crashed thread).

WHY THIS EXISTS: opportunity_bootstrap.py was originally a manual CLI
script (`python -m ml.opportunity_bootstrap`). Per explicit requirement:
this needs to run automatically alongside the main app, AND — just as
important — it must be structurally incapable of blocking or crashing
the app when real interaction/signup data is thin or entirely absent,
since that's expected to be the normal state for a while, not an edge
case to handle defensively as an afterthought.

WHAT "SAFE WITH ZERO DATA" MEANS HERE, CONCRETELY:
  - Zero real users with resume_skills -> generate() already returns []
    (see opportunity_bootstrap.py) -> logged as a normal outcome, not
    a warning or error.
  - Missing/absent grounding dataset (final_dataset_v3.csv) for the
    synthetic path -> _build_role_skill_pools / _load_synthetic_profiles
    now catch FileNotFoundError and return empty rather than raising
    (see opportunity_bootstrap.py's SAFE ON MISSING FILE docstrings).
  - Empty opportunities table (no scraper data synced yet) ->
    opp_ops.match_by_skills() already returns [] for that case, which
    generate()/generate_synthetic() already handle as skipped_no_candidates.
  - ANY other unexpected exception -> caught at the top level in _run()
    below, logged with full traceback, status set to 'failed', and the
    scheduler reschedules itself for next time regardless. A bad run
    never prevents the NEXT run from being attempted.

This intentionally only regenerates the pseudo-label CSVs (data/
opportunity_pseudo_labels*.csv) — it does NOT retrain the CatBoost
model automatically. Model training is a deliberate, reviewed step
(see PROJECT_STATUS.md's "empirical validation over assertion"
principle) — auto-retraining on a schedule without a human looking at
the output first is a different, riskier decision this scheduler does
not make on its own.
"""
import logging
import threading
from datetime import datetime
from typing import Optional

from careercompass.config import Config
from careercompass.ml.opportunity_bootstrap import generate, generate_synthetic, write_csv, OUTPUT_PATH, SYNTHETIC_OUTPUT_PATH, COMBINED_OUTPUT_PATH

log = logging.getLogger(__name__)

# How often to regenerate — this is a CHEAP batch CSV write, not a
# live-serving path, so daily is a reasonable default. Adjust via
# Config the same way OPPORTUNITY_SYNC_INTERVAL_HOURS already is.
BOOTSTRAP_INTERVAL_HOURS = Config.BOOTSTRAP_INTERVAL_HOURS

_lock = threading.Lock()
_state: dict = {
    "running": False, "trigger": None, "started_at": None,
    "last_status": None, "last_completed": None, "error": None,
    "last_real_rows": None, "last_synthetic_rows": None, "last_combined_rows": None,
}
_run_lock = threading.Lock()
_timer: Optional[threading.Timer] = None


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


def start_scheduler():
    log.info("Opportunity bootstrap scheduler starting (interval: %dh)", BOOTSTRAP_INTERVAL_HOURS)
    _schedule(5, "startup")  # small offset from the other schedulers so they don't collide


def trigger_manual() -> bool:
    if _state["running"]:
        return False
    _schedule(0, "manual")
    return True


def stop_scheduler():
    global _timer
    if _timer:
        _timer.cancel()
        _timer = None


def _schedule(delay: float, trigger: str):
    t = threading.Timer(delay, _run, kwargs={"trigger": trigger})
    t.daemon = True
    t.start()


def _schedule_next():
    global _timer
    if _timer:
        _timer.cancel()
    secs = BOOTSTRAP_INTERVAL_HOURS * 3600
    _timer = threading.Timer(secs, _run, kwargs={"trigger": "scheduled"})
    _timer.daemon = True
    _timer.start()
    log.info("Next opportunity bootstrap run in %.1f h", BOOTSTRAP_INTERVAL_HOURS)


def _run(trigger: str = "scheduled"):
    if not _run_lock.acquire(blocking=False):
        log.warning("Opportunity bootstrap already running — skipping (%s)", trigger)
        return

    _set(running=True, trigger=trigger, started_at=_now(), error=None)

    try:
        real_rows = generate()
        synthetic_rows = generate_synthetic()
        combined = real_rows + synthetic_rows

        write_csv(real_rows, OUTPUT_PATH)
        write_csv(synthetic_rows, SYNTHETIC_OUTPUT_PATH)
        write_csv(combined, COMBINED_OUTPUT_PATH)

        if not combined:
            # EXPECTED, NORMAL outcome while real signups/grounding data
            # are thin — logged at INFO, not WARNING, so it doesn't read
            # as an alarm every single day this remains true.
            log.info("Opportunity bootstrap: 0 rows this run (real=0, synthetic=0). "
                     "This is expected while user signups and/or the grounding "
                     "dataset are unavailable — not an error condition.")
            status = "success_empty"
        else:
            log.info("Opportunity bootstrap: %d real + %d synthetic = %d combined rows",
                      len(real_rows), len(synthetic_rows), len(combined))
            status = "success"

        _set(last_status=status, last_real_rows=len(real_rows),
             last_synthetic_rows=len(synthetic_rows), last_combined_rows=len(combined))

    except Exception as e:
        # CATCH-ALL, DELIBERATE: this background job must never take the
        # app down. Anything unexpected (DB connection issue, malformed
        # data, whatever) is logged with a full traceback for debugging,
        # but the scheduler still reschedules itself below — a bad run
        # tonight doesn't prevent tomorrow's run from being attempted.
        log.exception("Opportunity bootstrap run failed: %s", e)
        _set(last_status="failed", error=str(e))

    finally:
        _set(running=False, last_completed=_now())
        _run_lock.release()
        _schedule_next()


def _now() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")