"""Step 5: at equal compute, more TabPFN-3.5 members or a second foundation model?

Reads the test predictions of TabPFN-3.5 at 1, 2, 4 and 16 members written by
``run_tabpfn_members.py`` (split 0 of the tiny and small datasets), the stored 8-member TabPFN-3.5
predictions and the stored predictions of the other TFMs. All ensembles are equal-weight averages,
so no choice uses test information, and every comparison is paired per dataset:

1. Saturation: the error at ``k`` members against 8 members (TabPFN-3.5's default).
2. Equal budget: 16 members against 8 members plus one TFM of another family. The partner's predict time relative to 8 TabPFN-3.5 members is reported next to
   each row, so the budget can be read per partner.

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
from run_tfm_ensembling_study import TFM_METHODS, holm, paired_summary, relative_improvement
from slim_predictions import ROLE_TUNED, SlimStore

API = "TA-TabPFN-3.5-API"
STORED_TABPFN = "TA-TabPFN-3.5"
SIZES = [1, 2, 4, 8, 16]
#: Partners: the TFMs outside the TabPFN family.
PARTNERS = [m for m in TFM_METHODS if "TabPFN" not in m]


def _error(metric: str, problem_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    from autogluon.core.metrics import get_metric

    return float(get_metric(metric, problem_type).error(y, pred))


def _test(root: Path, method: str, dataset: str) -> np.ndarray | None:
    path = root / "preds" / method / dataset / "0.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return f["test"][0]


def run_dataset(root: Path, dataset: str, info: pd.Series) -> list[dict]:
    with np.load(root / "labels" / dataset / "0.npz") as labels:
        y = labels["y_test"]
    tabpfn = {k: _test(root, f"{API}-n{k}", dataset) for k in SIZES if k != 8}
    tabpfn[8] = _test(root, STORED_TABPFN, dataset)
    rows = []

    def add(arm: str, members: int, pred: np.ndarray, partner: str | None = None) -> None:
        error = _error(info["metric"], info["problem_type"], y, pred)
        rows.append({"dataset": dataset, "arm": arm, "members": members, "partner": partner, "error": error})

    for k, pred in tabpfn.items():
        if pred is not None:
            add("TabPFN-3.5", k, pred)
    for partner in PARTNERS:
        other = _test(root, partner, dataset)
        if other is None:
            continue
        add("TabPFN-3.5 + partner", 8, (tabpfn[8] + other) / 2, partner)
    # Check: the API's own 8-member predictions against the stored ones, where requested.
    api8 = _test(root, f"{API}-n8", dataset)
    if api8 is not None:
        rows.append(
            {
                "dataset": dataset,
                "arm": "check: API 8 members",
                "members": 8,
                "partner": None,
                "error": _error(info["metric"], info["problem_type"], y, api8),
                "max_abs_diff_to_stored": float(np.abs(api8 - tabpfn[8]).max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.expanduser(), args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    members = SlimStore(root).members
    members = members[(members["role"] == ROLE_TUNED) & (members["fold"] == 0)]
    have = [p.name for p in (root / "preds" / f"{API}-n16").iterdir() if p.is_dir()]
    datasets = sorted(d for d in have if all(_test(root, f"{API}-n{k}", d) is not None for k in SIZES if k != 8))
    print(f"{len(datasets)} datasets with every ensemble size")
    info = members.drop_duplicates("dataset").set_index("dataset")[["metric", "problem_type"]]

    table = pd.DataFrame([row for d in datasets for row in run_dataset(root, d, info.loc[d])])
    table.to_parquet(output / "arms_per_dataset.parquet")
    if "max_abs_diff_to_stored" in table:
        check = table[table["arm"] == "check: API 8 members"][["dataset", "max_abs_diff_to_stored"]]
        check.to_csv(output / "check_api_8_members_vs_stored.csv", index=False)

    # 1. Saturation, against 8 members.
    curve = table[table["arm"] == "TabPFN-3.5"].pivot(index="dataset", columns="members", values="error")
    saturation = pd.DataFrame(
        [{"members": k, **paired_summary(relative_improvement(curve[8], curve[k]), rng)} for k in curve.columns]
    )
    saturation.to_csv(output / "saturation_vs_8_members.csv", index=False)

    # 2. Equal budget: 2k members of TabPFN-3.5 against k members plus a partner.
    time_infer = members.pivot_table(index="dataset", columns="method", values="time_infer_s").reindex(datasets)
    rows = []
    pairs = table[table["arm"] == "TabPFN-3.5 + partner"]
    for (k, partner), group in pairs.groupby(["members", "partner"]):
        pair_error = group.set_index("dataset")["error"].reindex(curve.index)
        keep = pair_error.notna()
        rows.append(
            {
                "tabpfn_members": k,
                "partner": partner,
                "against": f"TabPFN-3.5 x{2 * k}",
                **paired_summary(relative_improvement(curve.loc[keep, 2 * k], pair_error[keep]), rng),
                "vs_tabpfn_alone_mean_pct": 100 * relative_improvement(curve.loc[keep, k], pair_error[keep]).mean(),
                "partner_predict_time_vs_tabpfn8": float((time_infer[partner] / time_infer[STORED_TABPFN]).median()),
            }
        )
    budget = pd.DataFrame(rows).sort_values(["tabpfn_members", "mean_pct"], ascending=[True, False])
    budget["wilcoxon_p_holm"] = holm(budget["wilcoxon_p"])
    budget.to_csv(output / "equal_budget_more_members_vs_partner.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(saturation.round(3).to_string())
    print(budget.round(3).to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
