"""Regenerate matched E2 inputs and replay the saved protocols in a new run."""
import argparse
import os
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory name under runs/")
    parser.add_argument("--stage", choices=["generate", "matched", "search"], default="search")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("Use only letters, digits, underscores, and hyphens for the run ID")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    import count_transport_matched as matched
    import count_transport_matched_evaluate as metrics
    import count_transport_search_run as search

    run = matched.HERE / "runs" / args.run_id
    run.mkdir()
    primary = run / "matched"
    primary.mkdir()
    shutil.copyfile(matched.DEFAULT_RUN / "PLAN.md", primary / "PLAN.md")
    matched.freeze(primary)
    matched.generate(primary)
    print(f"Generated matched inputs: {primary}", flush=True)
    if args.stage == "generate":
        return

    matched.fit_all(primary)
    metrics.evaluate_run(primary)
    if args.stage == "matched":
        return

    followup = run / "search"
    followup.mkdir()
    shutil.copyfile(search.DEFAULT_RUN / "PLAN.md", followup / "PLAN.md")
    search.PRIMARY_RUN = primary
    search.freeze(followup)
    search.fit_all(followup)
    search.evaluate(followup)


if __name__ == "__main__":
    main()
