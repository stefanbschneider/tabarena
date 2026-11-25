"""Unfinished (stopped by the API quota, see README.md). Step 5: at equal compute, more TabPFN-3.5 members or a second foundation model?

Reads the single-member TabPFN-3.5 runs written by ``run_tabpfn_members.py`` (16 seeds on split 0
of the tiny and small datasets) and the stored predictions of the other TFMs. Averaging the first
``k`` seeds simulates TabPFN-3.5 with ``k`` members. Every comparison is paired per dataset and uses
no test information:

1. Saturation: the error of ``k`` members for k in 1, 2, 4, 8, 16, against 8 members (the default).
2. Equal budget: TabPFN-3.5 with 16 members against TabPFN-3.5 with 8 members plus one other TFM at
   its stored default size, averaged with equal weight (and, as a second arm, with greedy weights
   fitted on validation). Both arms run two TabPFN-3.5-sized ensembles' worth of forward passes when
   the partner costs as much as 8 TabPFN-3.5 members; the partner's measured predict time is
   reported next to it, so the budget can be read per partner.

    python -P examples/beyondarena/advanced/post_hoc_ensemble/run_within_vs_across_tfm.py \
        --root ~/beyond_slim --output examples/beyondarena/advanced/post_hoc_ensemble/results/within_vs_across
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # `python -P` leaves the script directory off the path
from ensembling import GREEDY, evaluate, get_metric, uniform
from run_tfm_ensembling_study import TFM_METHODS, holm, paired_summary, relative_improvement
from slim_predictions import ROLE_TUNED, SlimStore, TaskPredictions

API = "TA-TabPFN-3.5-API"
SIZES = [1, 2, 4, 8, 16]
#: Partners for the equal-budget arm: the TFMs of other families.
PARTNERS = [m for m in TFM_METHODS if not m.startswith("TA-TabPFN") and m != "TA-RealTabPFN-v2.5"]


def _members(n_seeds: int) -> list[str]:
    return [f"{API}-n1-s{seed}" for seed in range(n_seeds)]


def _score(metric: str, problem_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    from tabarena.simulation.ensemble_selection_config_scorer import TaskEvaluator

    scorer = get_metric(metric, problem_type)
    y, pred = TaskEvaluator._maybe_preprocess_bulk(scorer, y, pred)
    return float(scorer.error(y, pred))


def run_task(root: Path, dataset: str, stored: pd.DataFrame, n_seeds: int) -> list[dict]:
    """Errors of every arm on split 0 of one dataset."""
    store = SlimStore(root, read_members=False)
    metric, problem_type = stored["metric"].iloc[0], stored["problem_type"].iloc[0]
    seeds = pd.DataFrame({"method": _members(n_seeds), "config": _members(n_seeds)})
    members = store.load(dataset, 0, seeds)
    partners = store.load(dataset, 0, stored.reset_index(drop=True))
    rows = []

    for k in SIZES:
        if k <= n_seeds:
            error = _score(metric, problem_type, members.y_test, members.test[:k].mean(axis=0))
            rows.append({"dataset": dataset, "arm": "TabPFN-3.5", "members": k, "partner": None, "error": error})
    tabpfn8 = TaskPredictions(
        configs=["tabpfn8"],
        val=members.val[:8].mean(axis=0)[None],
        test=members.test[:8].mean(axis=0)[None],
        y_val=members.y_val,
        y_test=members.y_test,
    )
    for i, partner in enumerate(stored["method"]):
        pair = TaskPredictions(
            configs=["tabpfn8", partner],
            val=np.concatenate([tabpfn8.val, partners.val[i : i + 1]]),
            test=np.concatenate([tabpfn8.test, partners.test[i : i + 1]]),
            y_val=members.y_val,
            y_test=members.y_test,
        )
        for spec in (uniform(2), GREEDY):
            out = evaluate(pair, [0, 1], spec, metric, problem_type)
            arm = f"TabPFN-3.5 x8 + partner ({spec.name})"
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "members": 8,
                    "partner": partner,
                    "error": out["metric_error"],
                    "partner_weight": float(out["weights"][1]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    args = parser.parse_args()
    root, output = args.root.expanduser(), args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    api = pd.read_parquet(root / "tabpfn_api_members.parquet")
    singles = api[api["n_estimators"] == 1]
    complete = singles.groupby("dataset")["seed"].nunique()
    datasets = sorted(complete.index[complete >= args.seeds])
    print(f"{len(datasets)} datasets with all {args.seeds} single-member runs")

    stored = SlimStore(root).members
    stored = stored[(stored["role"] == ROLE_TUNED) & (stored["fold"] == 0) & stored["method"].isin(PARTNERS)]
    stored = stored[stored["dataset"].isin(datasets)]

    rows = []
    for dataset in datasets:
        rows += run_task(root, dataset, stored[stored["dataset"] == dataset], args.seeds)
    table = pd.DataFrame(rows)
    table.to_parquet(output / "arms_per_dataset.parquet")

    # Reproduction check: the API's 8-member run against the average of 8 single-member runs.
    check = api[api["n_estimators"] == 8].set_index("dataset")["metric_error"]
    avg8 = table[(table["arm"] == "TabPFN-3.5") & (table["members"] == 8)].set_index("dataset")["error"]
    joined = pd.concat([check.rename("api_n8"), avg8.rename("avg_of_8_singles")], axis=1, join="inner")
    joined.to_csv(output / "check_n8_vs_avg_of_singles.csv")

    # 1. Saturation, against 8 members.
    curve = table[table["arm"] == "TabPFN-3.5"].pivot(index="dataset", columns="members", values="error")
    saturation = pd.DataFrame(
        [{"members": k, **paired_summary(relative_improvement(curve[8], curve[k]), rng)} for k in curve.columns]
    )
    saturation.to_csv(output / "saturation_vs_8_members.csv", index=False)

    # 2. Equal budget: 16 members against 8 members plus a partner.
    rows = []
    for (arm, partner), group in table[table["partner"].notna()].groupby(["arm", "partner"]):
        pair_error = group.set_index("dataset")["error"].reindex(curve.index)
        keep = pair_error.notna()
        rows.append(
            {
                "arm": arm,
                "partner": partner,
                "partner_weight": group["partner_weight"].mean(),
                **paired_summary(relative_improvement(curve.loc[keep, 16], pair_error[keep]), rng),
                "vs_8_members_mean_pct": 100 * relative_improvement(curve.loc[keep, 8], pair_error[keep]).mean(),
            }
        )
    budget = pd.DataFrame(rows).sort_values(["arm", "mean_pct"], ascending=[True, False])
    budget["wilcoxon_p_holm"] = holm(budget["wilcoxon_p"])
    # Partner cost: its median predict time relative to the stored 8-member TabPFN-3.5.
    time_infer = SlimStore(root).members
    time_infer = time_infer[(time_infer["fold"] == 0) & time_infer["dataset"].isin(datasets)]
    time_infer = time_infer.pivot_table(index="dataset", columns="method", values="time_infer_s")
    ratio = {p: float((time_infer[p] / time_infer["TA-TabPFN-3.5"]).median()) for p in PARTNERS}
    budget["partner_predict_time_vs_tabpfn8"] = budget["partner"].map(ratio)
    budget.to_csv(output / "equal_budget_16_members_vs_8_plus_partner.csv", index=False)

    pd.set_option("display.width", 250)
    print(joined.describe().round(4).to_string())
    print(saturation.round(3).to_string())
    print(budget.round(3).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
