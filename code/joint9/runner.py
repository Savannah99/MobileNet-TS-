"""Persistent sequential queue, reentrant completed stages and duplicate-run lock."""
import os
import fcntl
import subprocess
import traceback
from base import *


def main():
    setup()
    with (RUN/"runner.lock").open("w") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("ALREADY_RUNNING",flush=True);return
        (RUN/"runner.pid").write_text(str(os.getpid())+"\n")
        commands=[("module_checks",["checks.py"]),("cholec_head",["prepare.py","head"])]
        commands += [(f"rich_val_{d}",["prepare.py","rich","--dataset",d,"--split","val"]) for d in DATASETS]
        commands += [("validation",["evaluate.py","val"])]
        commands += [(f"rich_test_{d}",["prepare.py","rich","--dataset",d,"--split","test"]) for d in DATASETS]
        commands += [("test",["evaluate.py","test"]),("statistics",["evaluate.py","statistics"])]
        commands += [(f"verify_{d}",["verify_bench.py","verify",d]) for d in DATASETS]
        commands += [(f"benchmark_{d}",["verify_bench.py","benchmark",d]) for d in DATASETS]
        commands += [("report",["report.py"])]
        qpath=RUN/"queue.json"
        q=read(qpath) if qpath.exists() else {"steps":[{"name":n,"state":"pending"} for n,_ in commands]}
        q.update(state="running",pid=os.getpid())
        try:
            for (name,args),item in zip(commands,q["steps"]):
                assert name==item["name"]
                if item["state"]=="complete":continue
                item.update(state="running",started_utc=now());q.update(current=name,updated_utc=now())
                save_json(qpath,q)
                with (RUN/f"{name}.log").open("a",buffering=1) as f:
                    f.write(f"\nSTART {now()} {args}\n")
                    subprocess.run([PYTHON,"-u",str(Path(__file__).parent/args[0]),*args[1:]],
                                   cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                item.update(state="complete",finished_utc=now());save_json(qpath,q)
            q.update(state="complete",current=None,completed_utc=now());save_json(qpath,q)
        except Exception:
            q.update(state="failed",error=traceback.format_exc(),updated_utc=now())
            for item in q["steps"]:
                if item["state"]=="running":item["state"]="failed"
            save_json(qpath,q);raise


if __name__=="__main__":main()
