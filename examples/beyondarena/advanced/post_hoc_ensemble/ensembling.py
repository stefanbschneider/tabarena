"""Fit and score post-hoc ensembles on one task of a :class:`~slim_predictions.SlimStore`.

The ensemblers, metrics and the fit-on-validation, score-on-test evaluator are TabArena's own
(``tabarena.simulation``), the ones behind the leaderboard's ``(tuned + ensemble)`` rows. This
module only feeds them predictions from the slim store instead of an ``EvaluationRepository``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from tabarena.simulation.ensemble.basic_ensemblers import FixedWeightsEnsembler, SingleBestEnsembler
from tabarena.simulation.ensemble.greedy_ensembler import GreedyEnsembler
from tabarena.simulation.ensemble_selection_config_scorer import TaskEvaluator, get_fast_rmse, get_fast_roc_auc

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from slim_predictions import TaskPredictions

    from tabarena.simulation.ensemble.abstract_ensembler import AbstractEnsembler

#: Greedy ensemble selection steps, the value TabArena uses (paper, Appendix C.4).
N_ITERATIONS = 40


def get_metric(metric_name: str, problem_type: str) -> Scorer:
    """The task's metric, in the fast implementation the simulator uses where one exists."""
    from autogluon.core.metrics import get_metric as ag_get_metric

    from tabarena.metrics import _fast_log_loss

    if metric_name == "log_loss":
        return _fast_log_loss.fast_log_loss
    if metric_name == "roc_auc":
        return get_fast_roc_auc()
    if metric_name in ("rmse", "root_mean_squared_error"):
        return get_fast_rmse()
    return ag_get_metric(metric=metric_name, problem_type=problem_type)


@dataclass
class EnsembleSpec:
    """One ensemble to fit: a name, the ensembler and its arguments."""

    name: str
    ensembler_cls: type[AbstractEnsembler] = GreedyEnsembler
    ensembler_kwargs: dict = field(default_factory=lambda: {"ensemble_size": N_ITERATIONS})


GREEDY = EnsembleSpec("greedy")
SINGLE_BEST = EnsembleSpec("single_best", SingleBestEnsembler, {})


def uniform(n_members: int) -> EnsembleSpec:
    """Equal-weight average: nothing is fitted, so validation data plays no role."""
    return EnsembleSpec("uniform", FixedWeightsEnsembler, {"weights": np.full(n_members, 1.0 / n_members)})


def evaluate(task: TaskPredictions, rows: list[int], spec: EnsembleSpec, metric_name: str, problem_type: str) -> dict:
    """Fit ``spec`` on the validation predictions of the members ``rows`` and score it on test.

    Returns ``metric_error`` (test), ``metric_error_val`` and ``weights`` (one per entry of ``rows``).
    Labels of the test split are used for scoring only.
    """
    metric = get_metric(metric_name, problem_type)
    evaluator = TaskEvaluator(
        ensembler_cls=spec.ensembler_cls,
        ensembler_kwargs=spec.ensembler_kwargs,
        eval_metric=metric,
        fit_eval_metric=metric,
        problem_type=problem_type,
    )
    pred_val = task.val[rows]
    results, _ = evaluator.run(
        pred_train=pred_val,
        y_train=task.y_val,
        pred_test=task.test[rows],
        y_test=task.y_test,
        return_metric_error_val=True,
        pred_val=pred_val,
        y_val=task.y_val,
    )
    return {
        "metric_error": float(results["metric_error"]),
        "metric_error_val": float(results["metric_error_val"]),
        "weights": np.asarray(results["ensemble_weights"], dtype=float),
    }
