"""Step 4: figures for the walkthrough, from the tables the other steps wrote.

python -P examples/beyondarena/advanced/post_hoc_ensemble/run_make_figures.py \
        --results examples/beyondarena/advanced/post_hoc_ensemble/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabarena.contexts.beyondarena.methods import beyond_method_metadata_lst

ENSEMBLE = "BeyondArena-Ensemble"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TFM_METHODS = {m.method for m in beyond_method_metadata_lst if not m.can_hpo}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "ytick.left": False,
        "xtick.bottom": False,
    }
)


def figure_leaderboard(results: Path, figures: Path, top: int = 24) -> None:
    leaderboard = pd.read_csv(results / "leaderboard_core2k_variants.csv").head(top).iloc[::-1]
    is_ensemble = leaderboard["method"].str.startswith(ENSEMBLE)
    is_selection = leaderboard["method"].str.startswith("Best-Single")
    colors = np.where(is_ensemble, ORANGE, np.where(is_selection, AQUA, BLUE))
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * top + 1))
    ax.barh(
        leaderboard["method"],
        leaderboard["elo"],
        xerr=[leaderboard["elo-"], leaderboard["elo+"]],
        color=colors,
        height=0.68,
        error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0},
    )
    for y, (elo, plus) in enumerate(zip(leaderboard["elo"], leaderboard["elo+"], strict=True)):
        ax.text(elo + plus + 6, y, f"{elo:.0f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(leaderboard["elo"].min() - 80, (leaderboard["elo"] + leaderboard["elo+"]).max() + 50)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, AQUA, BLUE)]
    labels = ["cross-model ensemble", "single model picked per task on validation", "roster entrant"]
    ax.legend(handles, labels, frameon=False, loc="lower right")
    ax.set_xlabel("Elo on core2k (bars: 95% bootstrap interval)")
    ax.set_title(f"core2k leaderboard with cross-model ensembles, top {top}", loc="center")
    fig.tight_layout()
    fig.savefig(figures / "leaderboard_core2k.png", dpi=200)
    plt.close(fig)


def _dataset_mean_weights(weights: pd.DataFrame, ensemble: str) -> pd.DataFrame:
    """Dataset x method matrix of weights, averaged over splits; a model absent from a task has weight 0."""
    w = weights[weights["ensemble"] == ensemble]
    per_task = w.pivot_table(index=["dataset", "fold"], columns="method", values="weight", fill_value=0.0)
    return per_task.groupby("dataset").mean()


def figure_weights(results: Path, figures: Path) -> None:
    weights = pd.read_parquet(results / "ensemble_weights.parquet")
    matrix = _dataset_mean_weights(weights, ENSEMBLE)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(matrix), size=(2000, len(matrix)))
    boot = matrix.to_numpy()[idx].mean(axis=1)
    summary = pd.DataFrame(
        {"mean": matrix.mean(), "low": np.quantile(boot, 0.025, axis=0), "high": np.quantile(boot, 0.975, axis=0)}
    ).sort_values("mean")
    summary.to_csv(results / "ensemble_mean_weights.csv", index_label="method")

    is_tfm = summary.index.isin(TFM_METHODS)
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(summary) + 1))
    ax.barh(
        summary.index,
        summary["mean"],
        xerr=[summary["mean"] - summary["low"], summary["high"] - summary["mean"]],
        color=np.where(is_tfm, BLUE, AQUA),
        height=0.68,
        error_kw={"ecolor": MUTED, "elinewidth": 1},
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA)]
    ax.legend(handles, ["foundation model", "tuned tree / neural / linear model"], frameon=False, loc="lower right")
    ax.set_xlabel("Mean weight in the ensemble across datasets (bars: 95% bootstrap interval over datasets)")
    ax.set_title("Which models the BeyondArena ensemble relies on")
    fig.tight_layout()
    fig.savefig(figures / "ensemble_weights.png", dpi=200)
    plt.close(fig)


def figure_subsets(results: Path, figures: Path) -> None:
    names = ["core2k", "tiny", "small", "medium", "large", "iid", "temporal", "grouped"]
    series = {
        ENSEMBLE: ("all models", ORANGE),
        f"{ENSEMBLE} (no early-stopped models)": ("no early-stopped models", AQUA),
        f"{ENSEMBLE} (TFMs only)": ("TFMs only", VIOLET),
    }
    rows = []
    for name in names:
        file = "leaderboard_core2k_variants.csv" if name == "core2k" else f"leaderboard_core2k_{name}_variants.csv"
        leaderboard = pd.read_csv(results / file).set_index("method")
        roster = leaderboard[~leaderboard.index.str.startswith((ENSEMBLE, "Best-Single"))]
        row = {"subset": name, "best_roster_entrant": roster.index[0], "best_roster_elo": roster["elo"].iloc[0]}
        row |= {f"elo_{label}": leaderboard.loc[method, "elo"] for method, (label, _) in series.items()}
        row["elo_best_single_val_selected"] = leaderboard.loc["Best-Single-Model (val-selected)", "elo"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(results / "ensemble_vs_best_by_subset.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(table))[::-1]
    elo_columns = ["best_roster_elo", *[f"elo_{label}" for label, _ in series.values()]]
    for yi, lo, hi in zip(y, table[elo_columns].min(axis=1), table[elo_columns].max(axis=1), strict=True):
        ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(table["best_roster_elo"], y, s=46, color=BLUE, label="best roster entrant", zorder=2, marker="D")
    for label, color in series.values():
        ax.scatter(table[f"elo_{label}"], y, s=46, color=color, label=f"ensemble, {label}", zorder=3)
    ax.set_yticks(y, table["subset"])
    ax.grid(axis="x")
    ax.set_xlabel("Elo within the subset (each subset has its own scale)")
    ax.set_title("Cross-model ensembles against the best roster entrant, by task subset")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "ensemble_by_subset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_tfm_pairs(results: Path, figures: Path) -> None:
    pairs = pd.read_csv(results / "tfm_study" / "q2b_pairs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    titles = {"uniform": "Equal-weight average of two TFMs", "greedy": "Weights fitted on validation"}
    for ax, (ensembler, title) in zip(axes, titles.items(), strict=True):
        p = pairs[pairs["ensembler"] == ensembler]
        for same, color, label in ((False, BLUE, "different families"), (True, ORANGE, "same family")):
            q = p[p["same_family"] == same]
            ax.scatter(
                100 * q["partner_gap"],
                100 * q["gain"],
                s=30,
                color=color,
                label=label,
                alpha=0.85,
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=3 if same else 2,
            )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.grid(axis="y")
        ax.set_title(title)
        ax.set_xlabel("How much weaker the added TFM is on validation (%)")
    axes[0].set_ylabel("Error reduction over the stronger TFM alone (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figures / "tfm_pairs.png", dpi=200)
    plt.close(fig)


def figure_tfm_pools(results: Path, figures: Path) -> None:
    q1 = pd.read_csv(results / "tfm_study" / "q1_cross_tfm_vs_single.csv")
    q1 = q1[q1["baseline"].str.startswith("best TFM per task")].iloc[::-1]
    q2a = pd.read_csv(results / "tfm_study" / "q2a_within_vs_cross_family_pools.csv").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))

    ax = axes[0]
    ax.errorbar(
        q1["mean_pct"],
        q1["ensemble"],
        xerr=[q1["mean_pct"] - q1["ci_low_pct"], q1["ci_high_pct"] - q1["mean_pct"]],
        fmt="o",
        color=BLUE,
        ecolor=MUTED,
        elinewidth=1,
    )
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.margins(y=0.25)
    ax.set_xlabel("Mean error reduction over the best single TFM,\npicked per task on validation (%)")
    ax.set_title("Across all TFMs")

    ax = axes[1]
    y = np.arange(len(q2a))
    ax.barh(y + 0.18, q2a["cross_gain_pct"], height=0.32, color=BLUE, label="size-matched pool from different families")
    ax.barh(y - 0.18, q2a["within_gain_pct"], height=0.32, color=ORANGE, label="pool within the family")
    ax.set_yticks(y, [f"{f} ({n} models)" for f, n in zip(q2a["family"], q2a["n_models"], strict=True)])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Mean error reduction over the pool's own best model,\npicked per task on validation (%)")
    ax.set_title("Same number of models, same or different family")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1)
    fig.tight_layout()
    fig.savefig(figures / "tfm_pools.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.expanduser()
    figures = results.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_leaderboard(results, figures)
    figure_weights(results, figures)
    figure_subsets(results, figures)
    if (results / "tfm_study").exists():
        figure_tfm_pairs(results, figures)
        figure_tfm_pools(results, figures)


if __name__ == "__main__":
    main()
