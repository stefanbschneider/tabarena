"""Unfinished (stopped by the API quota, see README.md). Fit single-member TabPFN-3.5 ensembles through the hosted API, one seed per config.

Each config runs TabPFN-3.5 with ``n_estimators=1`` and its own ``random_state`` on split 0 of the
tiny and small BeyondArena datasets (up to 10k training rows, no text columns), with the same
inner folds and preprocessing as the stored run. Averaging the first ``k`` seeds then simulates a
``k``-member TabPFN-3.5 ensemble (see ``run_within_vs_across_tfm.py``). One more config runs the
API default (8 members, seed 0) on the first 10 datasets, as a check against the stored
TA-TabPFN-3.5 predictions.

    TABPFN_TOKEN=... python -P examples/beyondarena/advanced/post_hoc_ensemble/run_tabpfn_members.py \
        --expname ~/tabpfn_members --root ~/beyond_slim

``run`` fits (restartable: finished configs are cached under ``--expname``), ``collect`` writes the
predictions into the slim store as methods ``TA-TabPFN-3.5-API-n1-s<seed>`` and ``-n8-s0``.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from slim_predictions import _labels_agree

AG_NAME = "TA-TabPFN-3.5-API"
#: Datasets on which the 8-member API default also runs, as a check against the stored predictions.
N_CHECK_DATASETS = 10


def configs(n_seeds: int) -> list[dict]:
    """The check config first, then one single-member config per seed."""
    return [{"n_estimators": 8}, *({"n_estimators": 1, "random_state": seed} for seed in range(n_seeds))]


def config_label(config: dict) -> str:
    return f"n{config['n_estimators']}-s{config.get('random_state', 0)}"


def datasets() -> list[str]:
    from tabarena.contexts import BeyondArenaContext

    tasks = BeyondArenaContext(backend="native").task_metadata.drop_duplicates("dataset")
    keep = (tasks["n_samples_train_per_fold"] <= 10_000) & (tasks["num_text_cols"] == 0)
    return sorted(tasks.loc[keep, "dataset"])


def run_shard(expname: str, dataset_names: list[str], n_seeds: int) -> None:
    from tabpfn_api_model import config_generator

    from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
    from tabarena.benchmark.task.metadata.collection import TaskSubset
    from tabarena.contexts import BeyondArenaContext

    experiments = BeyondArenaExperimentBundle(models=[(config_generator(configs(n_seeds)), 0)]).build_experiments()
    check_datasets = set(datasets()[:N_CHECK_DATASETS])
    context = BeyondArenaContext(backend="native")
    for dataset in dataset_names:
        # The 8-member check config runs on a few datasets only; the API allows 60 fits per minute.
        dataset_experiments = experiments if dataset in check_datasets else experiments[1:]
        # One dataset at a time, so that a failing dataset does not stop the shard.
        try:
            context.build_and_run_jobs(
                dataset_experiments,
                expname=expname,
                task_subset=TaskSubset(subset=["core2k", "lite"], dataset_names=[dataset]),
                register=False,
                debug_mode=True,
            )
        except Exception as e:
            print(f"FAILED {dataset}: {type(e).__name__}: {e}", flush=True)


def collect(expname: Path, root: Path, n_seeds: int) -> pd.DataFrame:
    """Copy the predictions into the slim store, checking labels against the stored ones."""
    rows = []
    for i, config in enumerate(configs(n_seeds)):
        method = f"{AG_NAME}-{config_label(config)}"
        for path in sorted((expname / "data" / f"{AG_NAME}_c{i + 1}_BAG_L1").glob("*/0_0/results.pkl")):
            with gzip.open(path, "rb") as f:
                result = pickle.load(f)
            dataset, artifacts = path.parts[-3], result["simulation_artifacts"]
            name = f"{AG_NAME}_c{i + 1}_BAG_L1"
            with np.load(root / "labels" / dataset / "0.npz") as labels:
                if not (
                    _labels_agree(labels["y_val"], artifacts["y_val"])
                    and _labels_agree(labels["y_test"], artifacts["y_test"])
                ):
                    raise ValueError(f"{method}: labels of {dataset} differ from the stored ones")
            out = root / "preds" / method / dataset / "0.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out,
                models=np.array([method]),
                val=artifacts["pred_proba_dict_val"][name][None].astype(np.float32),
                test=artifacts["pred_proba_dict_test"][name][None].astype(np.float32),
            )
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "fold": 0,
                    "n_estimators": config["n_estimators"],
                    "seed": config.get("random_state", 0),
                    "metric": result["metric"],
                    "problem_type": result["problem_type"],
                    "metric_error": float(result["metric_error"]),
                    "metric_error_val": float(result["metric_error_val"]),
                    "time_train_s": float(result["time_train_s"]),
                    "time_infer_s": float(result["time_infer_s"]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_parquet(root / "tabpfn_api_members.parquet")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "collect"])
    parser.add_argument("--expname", type=Path, required=True, help="Directory of the raw results.")
    parser.add_argument("--root", type=Path, required=True, help="Slim store to add the predictions to.")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6, help="Processes calling the API at the same time.")
    parser.add_argument("--shard", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    expname, root = args.expname.expanduser(), args.root.expanduser()

    if args.command == "collect":
        table = collect(expname, root, args.seeds)
        print(table.groupby("method").size().to_string())
        return
    if args.shard is not None:
        run_shard(str(expname), args.shard.split(","), args.seeds)
        return
    names = datasets()
    print(f"{len(names)} datasets, {args.seeds + 1} configs each, {args.workers} workers")
    shards = [names[i :: args.workers] for i in range(args.workers)]
    expname.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, shard in enumerate(shards):
        cmd = [sys.executable, "-P", __file__, "run", "--expname", str(expname), "--root", str(root)]
        cmd += ["--seeds", str(args.seeds), "--shard", ",".join(shard)]
        log = open(expname / f"worker_{i}.log", "w")  # noqa: SIM115
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))  # noqa: S603
    for proc in procs:
        proc.wait()


if __name__ == "__main__":
    main()
