"""Step 3: does ensembling across different tabular foundation models (TFMs) beat staying within one?

Every stored TFM prediction already contains that model's built-in ensembling (its estimators over
feature permutations, preprocessing recipes or contexts). So the stored predictions can answer:

1. Does a post-hoc ensemble across TFMs improve on a single TFM with its built-in ensembling?
2. Are two TFMs of the same family (checkpoints of one lineage) worth less together than two TFMs
   of different families, once the strength of the added model is accounted for?
3. Is it better to scale up within a family, or to average two small models of different families?

All ensembles here are fitted on validation predictions or not fitted at all (uniform averages), and
datasets are the unit of every test. See ``README.md`` for the design and its limits.

    python -P examples/beyondarena/advanced/post_hoc_ensemble/run_tfm_ensembling_study.py \
        --root ~/beyond_slim --output examples/beyondarena/advanced/post_hoc_ensemble/results/tfm_study
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.contexts.beyondarena.methods import beyond_method_metadata_lst

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from ensembling import GREEDY, SINGLE_BEST, evaluate, uniform
from slim_predictions import ROLE_TUNED, SlimStore

#: Family = one lineage of checkpoints from one group. Models without a sibling are their own family.
FAMILIES: dict[str, list[str]] = {
    "TabPFN": ["TA-TabPFN-2.6", "TA-RealTabPFN-v2.5", "TA-TabPFN-3", "TA-TabPFN-3.5", "TA-TabPFN-3.5-Fast"],
    "Kumo": ["TA-Kumo-Tabular", "TA-Kumo-Tabular-Medium", "TA-Kumo-Tabular-Small"],
    "Seldon": ["TA-Seldon", "TA-Seldon-Mini"],
    "TabDPT": ["TA-TabDPT", "TA-TabDPT-1.3"],
}
#: (small sibling, large sibling) pairs for the scale-up comparison.
SCALE_PAIRS = [
    ("TA-TabPFN-3.5-Fast", "TA-TabPFN-3.5"),
    ("TA-Kumo-Tabular-Small", "TA-Kumo-Tabular"),
    ("TA-Seldon-Mini", "TA-Seldon"),
]
SMALL_MODELS = [small for small, _ in SCALE_PAIRS]
TFM_METHODS = [m.method for m in beyond_method_metadata_lst if not m.can_hpo]
FAMILY_OF = {m: next((f for f, members in FAMILIES.items() if m in members), m) for m in TFM_METHODS}


def _residuals(pred: np.ndarray, y: np.ndarray, problem_type: str) -> np.ndarray:
    """Per-row prediction errors, flattened: ``pred - y``, with ``y`` one-hot for multiclass."""
    if problem_type == "multiclass":
        target = np.zeros_like(pred)
        target[np.arange(len(y)), y.astype(int)] = 1.0
        return (pred - target).ravel()
    return pred - y.astype(pred.dtype)


def run_task(root: Path, dataset: str, fold: int, members: pd.DataFrame, pools: dict[str, list[str]]) -> dict:
    """Score singles, all uniform pairs and the pooled ensembles on one task."""
    store = SlimStore(root, read_members=False)
    members = members.set_index("method").loc[TFM_METHODS].reset_index()
    task = store.load(dataset, fold, members)
    metric, problem_type = members["metric"].iloc[0], members["problem_type"].iloc[0]
    index = {m: i for i, m in enumerate(TFM_METHODS)}
    key = {"dataset": dataset, "fold": fold}

    singles = []
    for method, i in index.items():
        out = evaluate(task, [i], SINGLE_BEST, metric, problem_type)
        singles.append({**key, "method": method, "error": out["metric_error"], "error_val": out["metric_error_val"]})

    residuals = np.stack([_residuals(task.test[i], task.y_test, problem_type) for i in index.values()])
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(residuals)

    pairs = []
    for a, b in itertools.combinations(TFM_METHODS, 2):
        for spec in (uniform(2), GREEDY):
            out = evaluate(task, [index[a], index[b]], spec, metric, problem_type)
            pairs.append(
                {
                    **key,
                    "a": a,
                    "b": b,
                    "ensembler": spec.name,
                    "error": out["metric_error"],
                    "residual_corr": correlation[index[a], index[b]],
                }
            )

    ensembles = []
    for name, methods in pools.items():
        rows = [index[m] for m in methods]
        for spec in (GREEDY, SINGLE_BEST, uniform(len(rows))):
            out = evaluate(task, rows, spec, metric, problem_type)
            ensembles.append(
                {
                    **key,
                    "pool": name,
                    "ensembler": spec.name,
                    "error": out["metric_error"],
                    "error_val": out["metric_error_val"],
                    "time_infer_s": float(members["time_infer_s"].to_numpy()[rows][out["weights"] > 0].sum()),
                }
            )
    return {"singles": singles, "pairs": pairs, "ensembles": ensembles}


def per_dataset(frame: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    """Average ``value`` over the splits of each dataset: datasets are the unit of analysis."""
    return frame.groupby([*by, "dataset"], as_index=False)[value].mean()


def paired_summary(improvement: pd.Series, rng: np.random.Generator, n_boot: int = 10_000) -> dict:
    """Summary of a per-dataset relative improvement: mean with a bootstrap CI over datasets,
    median, share of datasets improved and the two-sided Wilcoxon signed-rank p-value.
    """
    from scipy.stats import wilcoxon

    x = improvement.to_numpy()
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "n_datasets": len(x),
        "mean_pct": 100 * x.mean(),
        "ci_low_pct": 100 * np.quantile(boot, 0.025),
        "ci_high_pct": 100 * np.quantile(boot, 0.975),
        "median_pct": 100 * np.median(x),
        "share_improved": float((x > 0).mean()),
        "wilcoxon_p": float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0,
    }


def relative_improvement(base: pd.Series, new: pd.Series) -> pd.Series:
    """``(base - new) / base`` per dataset, clipped to [-1, 1] so near-zero errors cannot dominate a mean."""
    return ((base - new) / base.clip(lower=1e-12)).clip(-1, 1)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-adjusted p-values."""
    order = p_values.sort_values().index
    m = len(order)
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Slim store written by run_extract_predictions.py.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the result tables.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.expanduser(), args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    members = SlimStore(root).members
    members = members[(members["role"] == ROLE_TUNED) & members["method"].isin(TFM_METHODS)]
    # Tasks on which every TFM has predictions, so every arm sees the same tasks and nothing is imputed.
    n_methods = members.groupby(["dataset", "fold"])["method"].nunique()
    common = n_methods[n_methods == len(TFM_METHODS)].index
    members = members.set_index(["dataset", "fold"]).loc[common].reset_index()
    print(f"{len(common)} tasks on {members['dataset'].nunique()} datasets have predictions of all {len(TFM_METHODS)}")

    # Rank the TFMs by their mean rank of validation error. Pool choices below use this order only.
    val_rank = members.assign(rank=members.groupby(["dataset", "fold"])["metric_error_val"].rank())
    val_rank = per_dataset(val_rank, "rank", ["method"]).groupby("method")["rank"].mean().sort_values()
    flagship = {family: next(m for m in val_rank.index if FAMILY_OF[m] == family) for family in set(FAMILY_OF.values())}
    flagships = [m for m in val_rank.index if flagship[FAMILY_OF[m]] == m]
    pools = {"all TFMs": TFM_METHODS, "one per family": flagships}
    for family, methods in FAMILIES.items():
        pools[f"within {family}"] = methods
        # The same number of models, drawn from different families: the best-validating flagships.
        pools[f"cross, size of {family}"] = flagships[: len(methods)]
    pd.Series(pools).apply(", ".join).rename("methods").to_csv(output / "pools.csv", index_label="pool")

    groups = [(root, d, f, g.reset_index(drop=True), pools) for (d, f), g in members.groupby(["dataset", "fold"])]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        outputs = list(executor.map(run_task, *zip(*groups, strict=True), chunksize=4))
    singles = pd.DataFrame([r for o in outputs for r in o["singles"]])
    pairs = pd.DataFrame([r for o in outputs for r in o["pairs"]])
    ensembles = pd.DataFrame([r for o in outputs for r in o["ensembles"]])
    singles.to_parquet(output / "singles_per_split.parquet")
    pairs.to_parquet(output / "pairs_per_split.parquet")
    ensembles.to_parquet(output / "ensembles_per_split.parquet")

    single = per_dataset(singles, "error", ["method"]).pivot(index="dataset", columns="method", values="error")
    single_val = per_dataset(singles, "error_val", ["method"]).pivot(
        index="dataset", columns="method", values="error_val"
    )
    ens = per_dataset(ensembles, "error", ["pool", "ensembler"]).pivot(
        index="dataset", columns=["pool", "ensembler"], values="error"
    )
    best_tfm = val_rank.index[0]

    # -- Question 1: cross-TFM ensembles against single TFMs -------------------------------------------
    rows = []
    arms = {
        "greedy, all TFMs": ens[("all TFMs", "greedy")],
        "uniform, all TFMs": ens[("all TFMs", "uniform")],
        "greedy, one per family": ens[("one per family", "greedy")],
    }
    baselines = {
        f"best TFM overall ({best_tfm})": single[best_tfm],
        "best TFM per task (val-selected)": ens[("all TFMs", "single_best")],
    }
    for (arm, new), (baseline, base) in itertools.product(arms.items(), baselines.items()):
        rows.append({"ensemble": arm, "baseline": baseline, **paired_summary(relative_improvement(base, new), rng)})
    q1 = pd.DataFrame(rows)
    q1["wilcoxon_p_holm"] = holm(q1["wilcoxon_p"])
    q1.to_csv(output / "q1_cross_tfm_vs_single.csv", index=False)

    # -- Question 2a: pooled within-family ensembles against size-matched cross-family ensembles -------
    rows = []
    for family in FAMILIES:
        within, cross = ens[(f"within {family}", "greedy")], ens[(f"cross, size of {family}", "greedy")]
        # Gain of each ensemble over the best single model of its own pool, picked per task on validation.
        # The difference of the two gains is the comparison: it does not reward a pool for stronger members.
        gain_within = relative_improvement(ens[(f"within {family}", "single_best")], within)
        gain_cross = relative_improvement(ens[(f"cross, size of {family}", "single_best")], cross)
        rows.append(
            {
                "family": family,
                "n_models": len(FAMILIES[family]),
                "cross_pool": ", ".join(pools[f"cross, size of {family}"]),
                "within_gain_pct": 100 * gain_within.mean(),
                "cross_gain_pct": 100 * gain_cross.mean(),
                **{f"gain_difference_{k}": v for k, v in paired_summary(gain_cross - gain_within, rng).items()},
                "cross_vs_within_error_mean_pct": 100 * relative_improvement(within, cross).mean(),
            }
        )
    q2a = pd.DataFrame(rows)
    q2a["gain_difference_wilcoxon_p_holm"] = holm(q2a["gain_difference_wilcoxon_p"])
    q2a.to_csv(output / "q2a_within_vs_cross_family_pools.csv", index=False)

    # -- Question 2b: all pairs of TFMs. Anchor = the member with the lower validation error ----------
    keys = ["a", "b", "ensembler"]
    pair = per_dataset(pairs, "error", keys).merge(per_dataset(pairs, "residual_corr", keys), on=[*keys, "dataset"])
    lookup, lookup_val = single.stack().rename("e"), single_val.stack().rename("e")
    for side in ("a", "b"):
        idx = pd.MultiIndex.from_frame(pair[["dataset", side]])
        pair[f"error_{side}"] = lookup.reindex(idx).to_numpy()
        pair[f"val_{side}"] = lookup_val.reindex(idx).to_numpy()
    a_is_anchor = pair["val_a"] <= pair["val_b"]
    pair["error_anchor"] = np.where(a_is_anchor, pair["error_a"], pair["error_b"])
    val_anchor = np.where(a_is_anchor, pair["val_a"], pair["val_b"])
    val_partner = np.where(a_is_anchor, pair["val_b"], pair["val_a"])
    # How much weaker the added model is, on validation: 0 = as strong as the anchor.
    pair["partner_gap"] = ((val_partner - val_anchor) / np.clip(val_anchor, 1e-12, None)).clip(0, 1)
    pair["gain"] = relative_improvement(pair["error_anchor"], pair["error"])
    pair["same_family"] = pair["a"].map(FAMILY_OF) == pair["b"].map(FAMILY_OF)
    pair.to_parquet(output / "pairs_per_dataset.parquet")

    pair_columns = ["gain", "partner_gap", "residual_corr"]
    by_pair = pair.groupby([*keys, "same_family"], as_index=False)[pair_columns].mean()
    by_pair.to_csv(output / "q2b_pairs.csv", index=False)

    def pair_statistics(frame: pd.DataFrame) -> dict:
        """Raw and gap-adjusted difference between same-family and cross-family pairs."""
        p = frame.groupby(["a", "b", "same_family"], as_index=False)[pair_columns].mean()
        same = p["same_family"].to_numpy(dtype=bool)
        design = np.column_stack([np.ones(len(p)), p["partner_gap"], same.astype(float)])
        coef = np.linalg.lstsq(design, p["gain"].to_numpy(), rcond=None)[0]
        return {
            "gain_same_family_pct": 100 * p.loc[same, "gain"].mean(),
            "gain_cross_family_pct": 100 * p.loc[~same, "gain"].mean(),
            "raw_difference_pct": 100 * (p.loc[~same, "gain"].mean() - p.loc[same, "gain"].mean()),
            "cross_family_effect_adjusted_pct": -100 * coef[2],
            "partner_gap_slope_pct": 100 * coef[1],
            "residual_corr_same_family": p.loc[same, "residual_corr"].mean(),
            "residual_corr_cross_family": p.loc[~same, "residual_corr"].mean(),
        }

    datasets = pair["dataset"].unique()
    q2b_tables = []
    for ensembler, frame in pair.groupby("ensembler"):
        point = pair_statistics(frame)
        by_dataset = dict(iter(frame.groupby("dataset")))
        boot = pd.DataFrame(
            [
                pair_statistics(pd.concat([by_dataset[d] for d in rng.choice(datasets, size=len(datasets))]))
                for _ in range(1000)
            ]
        )
        table = pd.DataFrame({"estimate": point, "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)})
        table["p_bootstrap"] = [2 * min((boot[c] > 0).mean(), (boot[c] < 0).mean()) for c in table.index]
        q2b_tables.append(table.rename_axis("statistic").reset_index().assign(ensembler=ensembler))
    q2b = pd.concat(q2b_tables, ignore_index=True)
    q2b.to_csv(output / "q2b_same_vs_cross_family_pairs.csv", index=False)

    # -- Question 3: scale up within a family, or average two small models of different families? -----
    time_infer = per_dataset(members, "time_infer_s", ["method"]).pivot(
        index="dataset", columns="method", values="time_infer_s"
    )
    uniform_pairs = pair[pair["ensembler"] == "uniform"]
    pair_error = {
        frozenset(k): g.set_index("dataset")["error"].reindex(single.index)
        for k, g in uniform_pairs.groupby(["a", "b"])
    }
    large_of = dict(SCALE_PAIRS)
    rows = []
    for small_a, small_b in itertools.combinations(SMALL_MODELS, 2):
        average = pair_error[frozenset((small_a, small_b))]
        for large in dict.fromkeys((large_of[small_a], large_of[small_b])):
            cost_ratio = ((time_infer[small_a] + time_infer[small_b]) / time_infer[large]).median()
            rows.append(
                {
                    "small_pair": f"{small_a} + {small_b}",
                    "large": large,
                    "median_infer_time_ratio": cost_ratio,
                    **paired_summary(relative_improvement(single[large], average), rng),
                }
            )
    q3 = pd.DataFrame(rows)
    q3["wilcoxon_p_holm"] = holm(q3["wilcoxon_p"])
    q3.to_csv(output / "q3_scale_up_vs_cross_family_pair.csv", index=False)

    val_rank.rename("mean_val_rank").to_csv(output / "tfm_validation_rank.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, table in (("Q1", q1), ("Q2a", q2a), ("Q2b", q2b), ("Q3", q3)):
        print(f"\n{name}\n{table.round(4).to_string()}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
