"""Slim, local copy of the hosted BeyondArena predictions for cross-model ensembling.

The hosted ``processed.zip`` artifacts hold the validation and test predictions of every config of
every method on every split, about 71 GB zipped for the 26 roster methods. A cross-model ensemble
of one config per model needs a small part of that: per method and task, the default config and
the tuned config (the config with the lowest validation error on that task).

:func:`extract_method` streams one method's ``processed.zip`` over HTTP, keeps only those configs
on the requested tasks and never writes the zip to disk. :class:`SlimStore` reads the result.

Layout of the store::

    <root>/members.parquet                  one row per (method, dataset, fold, config)
    <root>/labels/<dataset>/<fold>.npz      y_val, y_test (written by the first method, checked by the others)
    <root>/preds/<method>/<dataset>/<fold>.npz   models, val, test (float32, one row per kept config)

With enough disk for the full artifacts, the same ensembles can be built without this module:
``BeyondArenaContext().load_repo(methods=...)`` returns an ``EvaluationRepositoryCollection`` whose
``evaluate_ensembles_per`` runs the same greedy ensemble selection.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

PUBLIC_BASE_URL = "https://data.tabarena.ai"
ROLE_DEFAULT = "default"
ROLE_TUNED = "tuned"


def core2k_tasks() -> pd.DataFrame:
    """The ``(dataset, fold)`` tasks of BeyondArena's default ``core2k`` protocol."""
    from tabarena.contexts.beyondarena.context import CORE2K_TASKS_CSV

    return pd.read_csv(CORE2K_TASKS_CSV).rename(columns={"split": "fold"})[["dataset", "fold"]]


def select_members(method_metadata: MethodMetadata, tasks: pd.DataFrame) -> pd.DataFrame:
    """Pick the default and the tuned config of one method on every task it has results for.

    The tuned config is the one with the lowest validation error on the task, the rule behind the
    leaderboard's ``(tuned)`` rows. Only validation scores enter the choice. ``time_train_all_s`` is
    the fit time summed over every config the method ran on the task: the cost of having tuned it.
    """
    results = method_metadata.load_model_results()
    results = results.merge(tasks, on=["dataset", "fold"], how="inner")
    name = method_metadata.method
    keys = ["dataset", "fold"]

    per_task = results.groupby(keys).agg(time_train_all_s=("time_train_s", "sum"), n_configs=("method", "nunique"))
    tuned = results.loc[results.groupby(keys)["metric_error_val"].idxmin()].assign(role=ROLE_TUNED)
    default = results[results["method"] == method_metadata.config_default].assign(role=ROLE_DEFAULT)
    members = pd.concat([default, tuned], ignore_index=True)
    members = members.rename(columns={"method": "config"}).assign(method=name)
    members = members.merge(per_task, left_on=keys, right_index=True)
    columns = [
        "method",
        *keys,
        "role",
        "config",
        "metric",
        "problem_type",
        "metric_error",
        "metric_error_val",
        "time_train_s",
        "time_infer_s",
        "time_train_all_s",
        "n_configs",
    ]
    return members[columns]


def _slice_predictions(path: Path, shape: list[int], dtype: str, rows: list[int]) -> np.ndarray:
    preds = np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=tuple(shape))
    return np.ascontiguousarray(preds[rows]).astype(np.float32, copy=False)


def _labels_agree(stored: np.ndarray, new: np.ndarray) -> bool:
    if stored.shape != new.shape:
        return False
    if stored.dtype.kind == "f" or new.dtype.kind == "f":
        # The older artifacts stored regression targets as CSV text: equal up to float round-off.
        return bool(np.allclose(stored, new, rtol=1e-9, atol=0.0, equal_nan=True))
    return bool(np.array_equal(stored, new))


def _write_labels(path: Path, y_val: np.ndarray, y_test: np.ndarray) -> bool:
    """Write the task's labels, or compare them with the ones another method wrote.

    Returns whether the labels agree, the check that all methods predicted the same rows.
    """
    if path.exists():
        with np.load(path) as stored:
            return _labels_agree(stored["y_val"], np.asarray(y_val)) and _labels_agree(
                stored["y_test"], np.asarray(y_test)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, y_val=np.asarray(y_val), y_test=np.asarray(y_test))
    tmp.replace(path)
    return True


def extract_method(
    method_metadata: MethodMetadata,
    members: pd.DataFrame,
    root: str | Path,
    *,
    base_url: str = PUBLIC_BASE_URL,
) -> dict:
    """Stream one method's ``processed.zip`` and keep the predictions of ``members``.

    Both artifact layouts are read: one ``tasks.dat`` per dataset, and the older per-task
    ``metadata.json`` with zipped label CSVs. Files of the wanted tasks are spooled to a temporary
    directory until the task is complete, then sliced to the wanted configs and removed.
    """
    import requests
    from stream_unzip import stream_unzip

    from tabarena.simulation.task_data import TASKS_FILENAME, decode_dataset_tasks, read_task_dir

    root = Path(root)
    name = method_metadata.method
    out_dir = root / "preds" / name
    wanted: dict[tuple[str, int], list[str]] = {
        key: list(dict.fromkeys(group["config"])) for key, group in members.groupby(["dataset", "fold"])
    }
    wanted_datasets = {dataset for dataset, _ in wanted}
    pending = set(wanted)
    label_mismatches: list[tuple[str, int]] = []
    dataset_tasks: dict[str, dict] = {}
    spool = Path(tempfile.mkdtemp(prefix=f"slim-{name}-", dir=root))

    def finish(dataset: str, fold: int) -> None:
        task_dir = spool / dataset / str(fold)
        if dataset in dataset_tasks:
            task = dataset_tasks[dataset][fold]
        else:
            task = read_task_dir(task_dir)
        models = list(task.metadata["models"])
        configs = [c for c in wanted[(dataset, fold)] if c in models]
        rows = [models.index(c) for c in configs]
        dtype = task.metadata["dtype"]
        val = _slice_predictions(task_dir / "pred-val.dat", task.metadata["pred_val_shape"], dtype, rows)
        test = _slice_predictions(task_dir / "pred-test.dat", task.metadata["pred_test_shape"], dtype, rows)
        if not _write_labels(root / "labels" / dataset / f"{fold}.npz", task.labels_val, task.labels_test):
            label_mismatches.append((dataset, fold))
        path = out_dir / dataset / f"{fold}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, models=np.array(configs), val=val, test=test)
        tmp.replace(path)
        shutil.rmtree(task_dir)
        pending.discard((dataset, fold))

    def is_complete(dataset: str, fold: int) -> bool:
        task_dir = spool / dataset / str(fold)
        have = {p.name for p in task_dir.iterdir()} if task_dir.exists() else set()
        if not {"pred-val.dat", "pred-test.dat"} <= have:
            return False
        if dataset in dataset_tasks:
            return True
        return {"metadata.json", "label-val.csv.zip", "label-test.csv.zip"} <= have or {
            "metadata.json",
            "labels.dat",
        } <= have

    key = (method_metadata.method_downloader().key_prefix / "processed.zip").as_posix()
    try:
        with requests.get(f"{base_url}/{key}", stream=True, timeout=120) as response:
            response.raise_for_status()
            for raw_name, _size, chunks in stream_unzip(response.iter_content(chunk_size=1 << 20)):
                parts = raw_name.decode().split("/")
                is_tasks_file = len(parts) == 3 and parts[2] == TASKS_FILENAME
                is_task_file = len(parts) == 4 and parts[2].isdigit()
                dataset = parts[1] if len(parts) >= 3 and parts[0] == "model_predictions" else None
                if is_tasks_file and dataset in wanted_datasets:
                    _, dataset_tasks[dataset] = decode_dataset_tasks(b"".join(chunks))
                    for fold in [f for d, f in pending if d == dataset]:
                        if is_complete(dataset, fold):
                            finish(dataset, fold)
                elif is_task_file and (dataset, int(parts[2])) in pending:
                    fold = int(parts[2])
                    path = spool / dataset / parts[2] / parts[3]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as f:
                        for chunk in chunks:
                            f.write(chunk)
                    if is_complete(dataset, fold):
                        finish(dataset, fold)
                else:
                    for _ in chunks:
                        pass
    finally:
        shutil.rmtree(spool, ignore_errors=True)

    summary = {
        "method": name,
        "n_tasks": len(wanted) - len(pending),
        "n_missing": len(pending),
        "missing": sorted(pending)[:20],
        "label_mismatches": label_mismatches,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@dataclass
class TaskPredictions:
    """Predictions of several configs on one task: ``val`` and ``test`` are ``(n_configs, n_rows[, n_classes])``."""

    configs: list[str]
    val: np.ndarray
    test: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class SlimStore:
    """Reader for a store written by :func:`extract_method`."""

    def __init__(self, root: str | Path, *, read_members: bool = True):
        """Open the store. ``read_members=False`` skips the scan of the prediction files, for
        workers that are handed their rows of :attr:`members`.
        """
        self.root = Path(root)
        self.members = self._read_members() if read_members else None

    def _read_members(self) -> pd.DataFrame:
        """The member table, restricted to the predictions that are on disk."""
        members = pd.read_parquet(self.root / "members.parquet")
        present = {
            (method_dir.name, dataset_dir.name, int(path.stem))
            for method_dir in (self.root / "preds").iterdir()
            for dataset_dir in method_dir.iterdir()
            if dataset_dir.is_dir()
            for path in dataset_dir.glob("*.npz")
            if path.stem.isdigit()
        }
        index = pd.MultiIndex.from_frame(members[["method", "dataset", "fold"]])
        return members[index.isin(present)].reset_index(drop=True)

    def labels(self, dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.root / "labels" / dataset / f"{fold}.npz") as f:
            return f["y_val"], f["y_test"]

    def load(self, dataset: str, fold: int, members: pd.DataFrame) -> TaskPredictions:
        """Stack the predictions of ``members`` (rows of :attr:`members` for this task), in row order."""
        val, test, configs = [], [], []
        for method, config in zip(members["method"], members["config"], strict=True):
            with np.load(self.root / "preds" / method / dataset / f"{fold}.npz") as f:
                row = list(f["models"]).index(config)
                val.append(f["val"][row])
                test.append(f["test"][row])
            configs.append(config)
        y_val, y_test = self.labels(dataset, fold)
        return TaskPredictions(configs=configs, val=np.stack(val), test=np.stack(test), y_val=y_val, y_test=y_test)
