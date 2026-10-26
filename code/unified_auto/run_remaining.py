"""Complete dependent evaluation after the bounded validation-only adaptation."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "outputs/autolaparo_unified_20261021"
state = {"state": "running", "pid": os.getpid(), "stages": [],
         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "current_stage": "waiting_for_validation_selection"}


def save():
    tmp = RUN / "run_status.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(RUN / "run_status.json")


try:
    save()
    deadline = time.time() + 3600
    while not (RUN / "selection.json").exists():
        if time.time() > deadline:
            raise TimeoutError("Validation selection was not completed")
        time.sleep(3)
    for script, stage in (
        ("experiment.py", "test"),
        ("verify_benchmark.py", "verify_val"),
        ("verify_benchmark.py", "benchmark"),
        ("verify_benchmark.py", "verify_test"),
        ("summarize.py", None),
    ):
        state["current_stage"] = script + (":" + stage if stage else "")
        save()
        start = time.time()
        args = [sys.executable, str(HERE / script)] + ([stage] if stage else [])
        with (RUN / f"{script[:-3]}_{stage or 'run'}.log").open("w") as f:
            subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        state["stages"].append({"script": script, "stage": stage, "seconds": time.time() - start})
        save()
    state.update(state="complete", current_stage="complete")
    save()
except Exception:
    state.update(state="failed", error=traceback.format_exc())
    save()
    raise
