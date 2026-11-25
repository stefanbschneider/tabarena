"""TabPFN-3.5 through the hosted Prior Labs API, as a TabArena model.

Used by ``run_tabpfn_members.py`` to fit TabPFN-3.5 with a chosen ensemble size and seed on the
same BeyondArena splits, inner folds and preprocessing as the stored ``TA-TabPFN-3.5`` run. The
preprocessing and the fold settings mirror :class:`tabarena.models.tabpfn_3_5.model.TabPFN35Model`
(categorical columns detected after AutoGluon's model-agnostic preprocessing, a fixed random state,
sequential folds and a refit on all training rows); only the forward passes run on Prior Labs'
servers instead of a local GPU.

The client reads its key from ``TABPFN_TOKEN``. In a session whose network proxy injects the key,
set it to a placeholder (``TABPFN_TOKEN=proxy-injected``).
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.core.models import AbstractModel

from tabarena.utils.config_utils import ConfigGenerator

if TYPE_CHECKING:
    import pandas as pd

#: The API alias of the checkpoint the stored TA-TabPFN-3.5 run used.
MODEL_PATH = "v3.5_default"
#: Attempts per API call. The API allows 60 fit requests per minute per key and answers more with
#: HTTP 429 and the wait in seconds; transient server errors are retried the same way.
MAX_ATTEMPTS = 30


def call_with_retry(fn, *args, **kwargs) -> Any:
    """Call ``fn``, waiting and retrying on rate limits and transient server errors."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            message = str(e)
            if "usage limit" in message.lower():
                raise  # the monthly quota: waiting does not help
            transient = "429" in message or re.search(r"HTTP 5\d\d", message) or "inaccessible" in message
            if not transient or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = re.search(r"Retry in (\d+)s", message)
            time.sleep((int(wait.group(1)) if wait else 10) + 1 + attempt % 5)
    raise AssertionError("unreachable")


class TabPFN35APIModel(AbstractModel):
    """TabPFN-3.5 fitted and predicted by the hosted API (``tabpfn_client``).

    Hyperparameters are passed to the client's estimator: ``n_estimators`` (the API default for
    v3.5 is 8) and ``random_state`` (default 0, as in the local wrapper).
    """

    ag_key = "TA-TABPFN-3.5-API"
    ag_name = "TA-TabPFN-3.5-API"

    fixed_random_state: ClassVar[int] = 0
    #: The warm-up's dummy fit would spend API requests and warms nothing in this process.
    warmup_dummy_fit: ClassVar[bool] = False
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}

    _categorical_indices: list[int] | None = None

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
            self._categorical_indices = [X.columns.get_loc(col) for col in categorical_cols] or None
        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor

        X = self.preprocess(X, y=y, is_train=True)
        model_cls = TabPFNRegressor if self.problem_type == "regression" else TabPFNClassifier
        hps = {
            "model_path": MODEL_PATH,
            "random_state": self.fixed_random_state,
            "categorical_features_indices": self._categorical_indices,
            "ignore_pretraining_limits": True,
            **self._get_model_params(),
        }
        self.model = call_with_retry(model_cls(**hps).fit, X, y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            return call_with_retry(self.model.predict, X)
        y_pred_proba = call_with_retry(self.model.predict_proba, X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        return 1, 0  # the work runs on the API

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return int(200e6)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        args = super()._get_default_ag_args_ensemble(**kwargs)
        args.update(cls._default_ag_args_ensemble_extra)
        return args


def config_generator(configs: list[dict]) -> ConfigGenerator:
    """One config per entry of ``configs`` (e.g. ``{"n_estimators": 1, "random_state": 3}``)."""
    return ConfigGenerator(search_space={}, model_cls=TabPFN35APIModel, manual_configs=configs)


class TabPFN35CaptureModel(TabPFN35APIModel):
    """Records the preprocessed tables of the final (refit) model instead of calling the API.

    Run through the TabArena pipeline, it writes the training table, its labels and every table the
    refit model is asked to predict, after exactly the preprocessing ``TabPFN35APIModel`` applies,
    to the directory in the ``TABPFN_CAPTURE_DIR`` environment variable. The fold models, whose
    validation predictions the captured design does not use, predict the training prior.
    """

    ag_key = "TA-TABPFN-3.5-CAPTURE"
    ag_name = "TA-TabPFN-3.5-Capture"

    _is_refit: bool = False
    _prior = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None, **kwargs) -> None:
        import json
        import os
        from pathlib import Path

        import numpy as np

        X = self.preprocess(X, y=y, is_train=True)
        self._is_refit = X_val is None
        if self.problem_type == "regression":
            self._prior = float(np.mean(y))
        else:
            counts = np.bincount(np.asarray(y, dtype=int), minlength=self.num_classes or int(np.max(y)) + 1)
            self._prior = counts / counts.sum()
        if not self._is_refit:
            return
        out = Path(os.environ["TABPFN_CAPTURE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        X.to_parquet(out / "X_train.parquet")
        np.save(out / "y_train.npy", np.asarray(y))
        meta = {"problem_type": self.problem_type, "categorical_features_indices": self._categorical_indices}
        (out / "meta.json").write_text(json.dumps(meta))

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        import os
        from pathlib import Path

        import numpy as np

        X = self.preprocess(X, **kwargs)
        if self._is_refit:
            out = Path(os.environ["TABPFN_CAPTURE_DIR"])
            index = len(list(out.glob("X_predict_*.parquet")))
            X.to_parquet(out / f"X_predict_{index}.parquet")
        if self.problem_type == "regression":
            return np.full(len(X), self._prior)
        proba = np.tile(self._prior, (len(X), 1))
        return self._convert_proba_to_unified_form(proba)
