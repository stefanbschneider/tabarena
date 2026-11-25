"""TabPFN-3.5 at several ensemble sizes through the hosted API, on split 0 of the tiny and small datasets.

The hosted API charges a flat amount per predict call (10,000 tokens, whatever the table size or the
number of members) and a key gets 5M tokens a month. So the ensemble size is set directly with
``n_estimators``, and only the final model's test predictions are requested: one call per dataset
and size. The 8-member predictions are the stored TA-TabPFN-3.5 ones.

Datasets: split 0 of the BeyondArena datasets with at most 10k training rows and no text columns
(84 datasets). Commands, in order:

``capture``
    Runs the TabArena pipeline once per dataset with ``TabPFN35CaptureModel`` (no API calls) and
    stores the final model's preprocessed training and test tables: the same splits, preprocessing
    and categorical columns as the stored run.
``estimate``
    Prices every planned call with the API's free cost estimate and stops when the total exceeds
    ``--budget``.
``run``
    Fits and predicts each dataset at every size in ``SIZES`` (plus 8 members on the first
    ``N_CHECK_DATASETS`` datasets, a check against the stored predictions) and writes the test
    predictions into the slim store as ``TA-TabPFN-3.5-API-n<k>``. Restartable; stops at once when
    the monthly quota is reached.

    python -P examples/beyondarena/advanced/post_hoc_ensemble/run_tabpfn_members.py capture \
        --capture ~/tabpfn_capture --root ~/beyond_slim
    TABPFN_TOKEN=... python -P examples/beyondarena/advanced/post_hoc_ensemble/run_tabpfn_members.py run \
        --capture ~/tabpfn_capture --root ~/beyond_slim
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from slim_predictions import _labels_agree
from tabpfn_api_model import MODEL_PATH, call_with_retry

API = "TA-TabPFN-3.5-API"
#: Ensemble sizes requested from the API; 8 members (the default) are stored. 32 members would cost
#: about twice the flat rate per call and does not fit in one month's quota.
SIZES = [1, 2, 4, 16]
#: Datasets on which 8 members are also requested, as a check against the stored predictions.
N_CHECK_DATASETS = 10
#: Tokens per predict call, as the API's cost estimate reports them.
TOKENS_PER_CALL = 10_000


def datasets() -> list[str]:
    from tabarena.contexts import BeyondArenaContext

    tasks = BeyondArenaContext(backend="native").task_metadata.drop_duplicates("dataset")
    keep = (tasks["n_samples_train_per_fold"] <= 10_000) & (tasks["num_text_cols"] == 0)
    return sorted(tasks.loc[keep, "dataset"])


def planned_calls(names: list[str]) -> list[tuple[str, int]]:
    """``(dataset, n_estimators)`` pairs, dataset by dataset so that a short quota leaves whole datasets."""
    check = set(names[:N_CHECK_DATASETS])
    return [(d, k) for d in names for k in (sorted([*SIZES, 8]) if d in check else SIZES)]


# -- capture ---------------------------------------------------------------------------------------


def capture(capture_dir: Path, root: Path) -> None:
    from tabpfn_api_model import TabPFN35CaptureModel

    from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
    from tabarena.benchmark.task.metadata.collection import TaskSubset
    from tabarena.contexts import BeyondArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    generator = ConfigGenerator(search_space={}, model_cls=TabPFN35CaptureModel, manual_configs=[{}])
    experiments = BeyondArenaExperimentBundle(models=[(generator, 0)]).build_experiments()
    context = BeyondArenaContext(backend="native")
    for dataset in datasets():
        out = capture_dir / dataset
        if (out / "X_test.parquet").exists():
            continue
        for stale in out.glob("*"):
            stale.unlink()
        os.environ["TABPFN_CAPTURE_DIR"] = str(out)
        context.build_and_run_jobs(
            experiments,
            expname=None,  # a throwaway results directory: only the captured tables are kept
            task_subset=TaskSubset(subset=["core2k", "lite"], dataset_names=[dataset]),
            register=False,
            debug_mode=True,
        )
        finish_capture(out, root / "labels" / dataset / "0.npz")
        print(f"captured {dataset}", flush=True)


def finish_capture(out: Path, labels_path: Path) -> None:
    """Pick the test table among the refit model's predict calls and check it against the stored labels."""
    with np.load(labels_path) as labels:
        y_val, y_test = labels["y_val"], labels["y_test"]
    y_train = np.load(out / "y_train.npy", allow_pickle=False)
    # The stored validation labels are the out-of-fold labels of all training rows, in training order.
    if not _labels_agree(y_val, y_train):
        raise ValueError(f"{out.name}: the captured training labels differ from the stored ones")
    candidates = [p for p in sorted(out.glob("X_predict_*.parquet")) if len(pd.read_parquet(p)) == len(y_test)]
    if not candidates:
        raise ValueError(f"{out.name}: no predict call of the refit model has the test size {len(y_test)}")
    # The test prediction is the last call of that size (an earlier one would be a throughput probe).
    candidates[-1].rename(out / "X_test.parquet")
    for path in out.glob("X_predict_*.parquet"):
        path.unlink()


# -- API -------------------------------------------------------------------------------------------


def _client_estimator(problem_type: str, n_estimators: int, categorical: list[int] | None):
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    model_cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
    return model_cls(
        model_path=MODEL_PATH,
        n_estimators=n_estimators,
        random_state=0,
        categorical_features_indices=categorical,
        ignore_pretraining_limits=True,
    )


def estimate(capture_dir: Path, budget: int) -> int:
    from tabpfn_client.cost import estimate_cost

    total = 0
    for dataset, k in planned_calls(datasets()):
        X_train = pd.read_parquet(capture_dir / dataset / "X_train.parquet")
        X_test = pd.read_parquet(capture_dir / dataset / "X_test.parquet")
        cost = estimate_cost(
            np.zeros(X_train.shape), np.zeros(X_test.shape), model_version=MODEL_PATH.split("_")[0], n_estimators=k
        )
        total += int(cost.estimated_cost)
    print(f"{len(planned_calls(datasets()))} calls, {total:,} tokens estimated, budget {budget:,}")
    if total > budget:
        raise SystemExit("The planned calls exceed the budget.")
    return total


def predict_test(capture_dir: Path, dataset: str, n_estimators: int) -> np.ndarray:
    """Test predictions in the stored format: positive-class probability (binary), one column
    per class in label order (multiclass), or the prediction (regression).
    """
    out = capture_dir / dataset
    meta = json.loads((out / "meta.json").read_text())
    X_train, X_test = pd.read_parquet(out / "X_train.parquet"), pd.read_parquet(out / "X_test.parquet")
    y_train = np.load(out / "y_train.npy")
    model = _client_estimator(meta["problem_type"], n_estimators, meta["categorical_features_indices"])
    call_with_retry(model.fit, X_train, y_train)
    if meta["problem_type"] == "regression":
        return np.asarray(call_with_retry(model.predict, X_test), dtype=np.float32)
    proba = np.asarray(call_with_retry(model.predict_proba, X_test), dtype=np.float32)
    n_classes = int(y_train.max()) + 1
    full = np.zeros((len(X_test), max(n_classes, 2)), dtype=np.float32)
    full[:, np.asarray(model.classes_, dtype=int)] = proba
    return full[:, 1] if meta["problem_type"] == "binary" else full


def run(capture_dir: Path, root: Path) -> None:
    calls = planned_calls(datasets())
    done = 0
    for dataset, k in calls:
        path = root / "preds" / f"{API}-n{k}" / dataset / "0.npz"
        if path.exists():
            done += 1
            continue
        start = time.time()
        test = predict_test(capture_dir, dataset, k)
        with np.load(root / "labels" / dataset / "0.npz") as labels:
            if len(test) != len(labels["y_test"]):
                raise ValueError(f"{dataset}: {len(test)} test predictions for {len(labels['y_test'])} labels")
        path.parent.mkdir(parents=True, exist_ok=True)
        # No validation predictions in this design; `val` holds an empty placeholder.
        np.savez(path, models=np.array([f"{API}-n{k}"]), val=np.zeros((1, 0), np.float32), test=test[None])
        done += 1
        print(f"[{done}/{len(calls)}] {dataset} n_estimators={k} ({time.time() - start:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["capture", "estimate", "run"])
    parser.add_argument("--capture", type=Path, required=True, help="Directory of the captured tables.")
    parser.add_argument("--root", type=Path, required=True, help="Slim store (labels, predictions).")
    parser.add_argument("--budget", type=int, default=4_800_000, help="Token budget for `estimate`.")
    args = parser.parse_args()
    capture_dir, root = args.capture.expanduser(), args.root.expanduser()
    if args.command == "capture":
        capture(capture_dir, root)
    elif args.command == "estimate":
        estimate(capture_dir, args.budget)
    else:
        estimate(capture_dir, args.budget)
        run(capture_dir, root)


if __name__ == "__main__":
    main()
