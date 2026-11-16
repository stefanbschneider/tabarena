"""Step 2: build the BeyondArena ensemble from stored predictions and rank it on the leaderboard.

Per ``core2k`` task, the pool holds one config per roster model: its tuned config (lowest validation
error on that task; a foundation model has a single config). Greedy ensemble selection (Caruana et
al., 40 steps) fits the weights on the validation predictions and the weighted average is scored on
the test predictions. Nothing is trained or predicted. See ``README.md`` for the assumptions.

    python -P examples/beyondarena/advanced/post_hoc_ensemble/run_build_ensemble.py \
        --root ~/beyond_slim --output examples/beyondarena/advanced/post_hoc_ensemble/results
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.contexts.beyondarena.methods import beyond_method_metadata_lst

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from ensembling import GREEDY, SINGLE_BEST, EnsembleSpec, evaluate
from slim_predictions import ROLE_DEFAULT, ROLE_TUNED, SlimStore

ENSEMBLE = "BeyondArena-Ensemble"
#: Foundation models: the roster methods without a search space.
TFM_METHODS = frozenset(m.method for m in beyond_method_metadata_lst if not m.can_hpo)
ALL_METHODS = frozenset(m.method for m in beyond_method_metadata_lst)
#: Models whose fold models stop early on the fold that also yields their validation predictions, which
#: makes those predictions optimistic (TabArena paper, Appendix A.6).
EARLY_STOPPED_METHODS = frozenset({"CatBoost", "LightGBM", "XGBoost", "TA-RealMLP", "TA-TabM"})
#: The best single roster entrant on the core2k leaderboard; it has results on every core2k task.
BEST_ROSTER_MODEL = "TA-Kumo-Tabular"
#: name -> (role of the pooled configs, pooled methods, ensembler)
VARIANTS: dict[str, tuple[str, frozenset[str], EnsembleSpec]] = {
    ENSEMBLE: (ROLE_TUNED, ALL_METHODS, GREEDY),
    f"{ENSEMBLE} (defaults only)": (ROLE_DEFAULT, ALL_METHODS, GREEDY),
    f"{ENSEMBLE} (no early-stopped models)": (ROLE_TUNED, ALL_METHODS - EARLY_STOPPED_METHODS, GREEDY),
    f"{ENSEMBLE} (TFMs only)": (ROLE_TUNED, TFM_METHODS, GREEDY),
    f"{ENSEMBLE} (no TFMs)": (ROLE_TUNED, ALL_METHODS - TFM_METHODS, GREEDY),
    "Best-Single-Model (val-selected)": (ROLE_TUNED, ALL_METHODS, SINGLE_BEST),
}


def run_task(root: Path, dataset: str, fold: int, members: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    """Fit every variant on one task. Returns result rows, long-format weight rows and single-config rescores."""
    store = SlimStore(root, read_members=False)
    unique = members.drop_duplicates(["method", "config"]).reset_index(drop=True)
    task = store.load(dataset, fold, unique)
    row_of = {(m, c): i for i, (m, c) in enumerate(zip(unique["method"], unique["config"], strict=True))}
    metric, problem_type = members["metric"].iloc[0], members["problem_type"].iloc[0]

    # Every pooled config rescored on its own, for the consistency check and the single-model baselines.
    rescored = []
    for (method, config), row in row_of.items():
        out = evaluate(task, [row], SINGLE_BEST, metric, problem_type)
        rescored.append(
            {
                "method": method,
                "dataset": dataset,
                "fold": fold,
                "config": config,
                "rescored_error": out["metric_error"],
                "rescored_error_val": out["metric_error_val"],
            }
        )

    results, weights = [], []
    for name, (role, methods, spec) in VARIANTS.items():
        pool = members[(members["role"] == role) & members["method"].isin(methods)]
        if pool.empty:
            continue
        rows = [row_of[(m, c)] for m, c in zip(pool["method"], pool["config"], strict=True)]
        out = evaluate(task, rows, spec, metric, problem_type)
        used = out["weights"] > 0
        # Training cost: every pooled model has to be fit, a tuned one with all of its configs.
        time_train_col = "time_train_all_s" if role == ROLE_TUNED else "time_train_s"
        results.append(
            {
                "dataset": dataset,
                "fold": fold,
                "method": name,
                "metric_error": out["metric_error"],
                "metric_error_val": out["metric_error_val"],
                "time_train_s": float(pool[time_train_col].sum()),
                "time_infer_s": float(pool["time_infer_s"].to_numpy()[used].sum()),
                "metric": metric,
                "problem_type": problem_type,
                "n_members": len(pool),
                "n_members_used": int(used.sum()),
            }
        )
        weights += [
            {"ensemble": name, "dataset": dataset, "fold": fold, "method": m, "weight": w}
            for m, w in zip(pool["method"], out["weights"], strict=True)
        ]
    return results, weights, rescored


def check_stored_predictions(rescored: pd.DataFrame) -> pd.DataFrame:
    """Compare the errors rescored from the stored predictions with the hosted ones, per method.

    Agreement shows that predictions, labels and metrics line up, on validation (what the weights are
    fitted on) and on test. The stored predictions are not always the exact arrays the hosted error
    was computed from, so a small share of tasks deviates (see the README); the check fails when a
    method deviates by more than 1% on more than 5% of its tasks.
    """
    for split in ("", "_val"):
        hosted = rescored[f"metric_error{split}"]
        rescored[f"rel_diff{split}"] = (rescored[f"rescored_error{split}"] - hosted) / hosted.abs().clip(lower=1e-12)
    aggregations = {"n": ("rel_diff", "size")}
    for name, column in (("val", "rel_diff_val"), ("test", "rel_diff")):
        aggregations |= {
            f"{name}_share_above_0p1pct": (column, lambda x: (x.abs() > 1e-3).mean()),
            f"{name}_share_above_1pct": (column, lambda x: (x.abs() > 1e-2).mean()),
            f"{name}_mean_rel_diff": (column, "mean"),
            f"{name}_max_abs_rel_diff": (column, lambda x: x.abs().max()),
        }
    return rescored.groupby("method").agg(**aggregations)


def fit_ensembles(root: Path, output: Path, workers: int) -> pd.DataFrame:
    """Fit every variant on every task and write the per-split results, weights and checks."""
    members = SlimStore(root).members
    groups = [(root, d, f, g.reset_index(drop=True)) for (d, f), g in members.groupby(["dataset", "fold"])]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(run_task, *zip(*groups, strict=True), chunksize=4))
    results = pd.DataFrame([row for rows, _, _ in outputs for row in rows])
    weights = pd.DataFrame([row for _, rows, _ in outputs for row in rows])
    rescored = pd.DataFrame([row for _, _, rows in outputs for row in rows])
    rescored = members.merge(rescored, on=["method", "dataset", "fold", "config"])
    check = check_stored_predictions(rescored)
    check.to_csv(output / "stored_prediction_check.csv")
    print(check.round(5).to_string())
    if check[["val_share_above_1pct", "test_share_above_1pct"]].to_numpy().max() > 0.05:
        raise AssertionError("Stored predictions do not reproduce the hosted errors.")
    rescored.to_parquet(output / "members_rescored.parquet")
    results.to_csv(output / "ensemble_results_per_split.csv", index=False)
    weights.to_parquet(output / "ensemble_weights.parquet")
    coverage = members[members["role"] == ROLE_TUNED].groupby("method").size().rename("n_tasks")
    coverage.to_csv(output / "member_coverage.csv")
    print(results.groupby("method")[["n_members", "n_members_used"]].mean().round(1))

    # How optimistic each model's validation error is: (test - validation) / test, median over datasets.
    rescored["optimism"] = (rescored["rescored_error"] - rescored["rescored_error_val"]) / rescored[
        "rescored_error"
    ].clip(lower=1e-12)
    optimism = rescored.groupby(["role", "method", "dataset"])["optimism"].mean()
    optimism.groupby(["role", "method"]).median().unstack("role").to_csv(output / "validation_optimism.csv")
    return results


def paired_comparison(results: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Each variant against the best roster model, paired per dataset (errors averaged over its splits)."""
    from run_tfm_ensembling_study import holm, paired_summary, relative_improvement

    rescored = pd.read_parquet(output / "members_rescored.parquet")
    base = rescored[(rescored["method"] == BEST_ROSTER_MODEL) & (rescored["role"] == ROLE_TUNED)]
    base = base.groupby("dataset")["metric_error"].mean()
    rng = np.random.default_rng(0)
    rows = []
    for name, group in results.groupby("method"):
        error = group.groupby("dataset")["metric_error"].mean().reindex(base.index)
        rows.append(
            {"ensemble": name, "baseline": BEST_ROSTER_MODEL, **paired_summary(relative_improvement(base, error), rng)}
        )
    table = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
    table["wilcoxon_p_holm"] = holm(table["wilcoxon_p"])
    table.to_csv(output / "ensemble_vs_best_roster_model_paired.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Slim store written by run_extract_predictions.py.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the result tables.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse", action="store_true", help="Reuse the per-split results in --output.")
    args = parser.parse_args()
    root, output = args.root.expanduser(), args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    if args.reuse:
        # Round-trip parsing: a last-digit change would turn exact ties between variants into wins and losses.
        results = pd.read_csv(output / "ensemble_results_per_split.csv", float_precision="round_trip")
    else:
        results = fit_ensembles(root, output, args.workers)
    print(paired_comparison(results, output).round(4).to_string())

    from tabarena.contexts import BeyondArenaContext

    context = BeyondArenaContext(backend="native")
    columns = [
        "dataset",
        "fold",
        "method",
        "metric_error",
        "metric_error_val",
        "time_train_s",
        "time_infer_s",
        "metric",
        "problem_type",
    ]
    new_results = results[columns].assign(method_type="baseline")
    subsets = {
        "core2k": ["core2k"],
        "core2k_tiny": ["core2k", "tiny"],
        "core2k_small": ["core2k", "small"],
        "core2k_medium": ["core2k", "medium"],
        "core2k_large": ["core2k", "large"],
        "core2k_iid": ["core2k", "iid"],
        "core2k_temporal": ["core2k", "temporal"],
        "core2k_grouped": ["core2k", "grouped"],
    }
    # The main ensemble alone next to the roster, then one leaderboard per subset with every variant.
    main_only = new_results[new_results["method"] == ENSEMBLE]
    leaderboard = context.compare(output_dir=None, new_results=main_only, subset=["core2k"], plot=False)
    leaderboard.to_csv(output / "leaderboard_core2k.csv", index=False)
    for name, subset in subsets.items():
        leaderboard = context.compare(output_dir=None, new_results=new_results, subset=subset, plot=False)
        leaderboard.to_csv(output / f"leaderboard_{name}_variants.csv", index=False)
        print(f"\n{name}\n{leaderboard.iloc[:8, :6].to_string()}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
