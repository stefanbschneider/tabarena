"""Step 1: build the slim prediction store the ensemble scripts read.

For every method of the BeyondArena roster, pick its default and tuned config per ``core2k`` task
from the hosted results tables, then stream the method's ``processed.zip`` and keep only those
predictions (see ``slim_predictions.py``). Nothing is trained or predicted.

About 71 GB are streamed and about 10 GB are written. Methods already extracted are skipped, so the
script can be restarted.

    python -P examples/beyondarena/advanced/post_hoc_ensemble/run_extract_predictions.py --root ~/beyond_slim
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from tabarena.contexts.beyondarena.methods import beyond_method_metadata_lst

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from slim_predictions import core2k_tasks, extract_method, select_members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory of the slim store.")
    parser.add_argument("--methods", nargs="*", default=None, help="Method names to extract (default: the roster).")
    parser.add_argument("--workers", type=int, default=3, help="Artifacts streamed at the same time.")
    args = parser.parse_args()

    root: Path = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    tasks = core2k_tasks()
    methods = [m for m in beyond_method_metadata_lst if args.methods is None or m.method in args.methods]

    members = pd.concat([select_members(m, tasks) for m in methods], ignore_index=True)
    members_path = root / "members.parquet"
    if members_path.exists():
        previous = pd.read_parquet(members_path)
        members = pd.concat([previous[~previous["method"].isin(members["method"])], members], ignore_index=True)
    members.to_parquet(members_path)

    def run(method_metadata) -> dict:
        name = method_metadata.method
        if (root / "preds" / name / "_summary.json").exists():
            return {"method": name, "skipped": True}
        summary = extract_method(method_metadata, members[members["method"] == name], root)
        print(summary, flush=True)
        return summary

    # Smallest artifacts first, so that most methods are available early.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run, sorted(methods, key=lambda m: m.can_hpo)))


if __name__ == "__main__":
    main()
