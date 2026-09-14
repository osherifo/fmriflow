"""Himalaya-based ridge regression models with cross-validated regularization."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fmriflow.core.array_utils import make_delayed
from fmriflow.core.types import ModelResult, PreparedData
from fmriflow.modules._decorators import model


# ─── Shared helpers ──────────────────────────────────────────

_VALID_BACKENDS = ("numpy", "cupy", "torch", "torch_cuda")


def _set_backend(backend):
    """Set the himalaya compute backend.

    Parameters
    ----------
    backend : str or None
        One of 'numpy', 'cupy', 'torch', 'torch_cuda'.
        If None, does nothing (himalaya uses its default).
    """
    if backend is None:
        return
    from himalaya.backend import set_backend
    set_backend(backend)


def _to_numpy(x):
    """Backend-agnostic conversion to a numpy array.

    Handles every himalaya backend: cupy arrays (``.get()``), torch tensors
    incl. ``torch_cuda`` (``.detach().cpu().numpy()``), and numpy passthrough.
    A prior ``hasattr(x, 'get')`` check only covered cupy, so torch_cuda
    tensors leaked into numpy ops and raised
    "can't convert cuda:0 device type tensor to numpy".
    """
    if x is None:
        return x
    if hasattr(x, 'get'):          # cupy ndarray
        return x.get()
    if hasattr(x, 'detach'):       # torch tensor (cpu or cuda)
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_alphas(spec):
    """Resolve alpha specification.

    'logspace(-2,5,20)' -> np.logspace(-2, 5, 20)
    [100, 200, 300]     -> np.array([100, 200, 300])
    """
    if isinstance(spec, str) and spec.startswith('logspace'):
        args = spec[len('logspace('):-1].split(',')
        parsed = [float(a.strip()) for a in args]
        if len(parsed) >= 3:
            parsed[2] = int(parsed[2])
        return np.logspace(*parsed)
    return np.array(spec, dtype=float)


def _compute_group_labels(feature_dims, delays, delays_applied):
    """Compute per-column group labels for BandedRidgeCV.

    Parameters
    ----------
    feature_dims : list[int]
        Number of dimensions per feature group (before delay).
    delays : list[int]
        Delay values used.
    delays_applied : bool
        Whether delays have already been applied to the data.

    Returns
    -------
    labels : list[int]
        Group label for each column of the (delayed) feature matrix.
    """
    if delays_applied:
        n_delays = len(delays)
        labels = []
        for group_idx, dim in enumerate(feature_dims):
            labels.extend([group_idx] * (dim * n_delays))
        return labels
    else:
        labels = []
        for group_idx, dim in enumerate(feature_dims):
            labels.extend([group_idx] * dim)
        return labels


def _compute_group_slices(feature_dims, delays_applied, delays=None):
    """Compute per-group column slices for ColumnKernelizer.

    Parameters
    ----------
    feature_dims : list[int]
        Number of dimensions per feature group (before delay).
    delays_applied : bool
        Whether delays have already been applied.
    delays : list[int], optional
        Delay values (needed if delays_applied=True).

    Returns
    -------
    slices : list[slice]
        One slice per feature group.
    """
    slices = []
    offset = 0
    for dim in feature_dims:
        if delays_applied:
            width = dim * len(delays)
        else:
            width = dim
        slices.append(slice(offset, offset + width))
        offset += width
    return slices


def _score_predictions(Y_pred, Y_true, metric='r2'):
    """Score predictions per target (voxel).

    Parameters
    ----------
    Y_pred : ndarray, shape (n_samples, n_targets)
    Y_true : ndarray, shape (n_samples, n_targets)
    metric : str
        'r2' or 'pearson_r'.

    Returns
    -------
    scores : ndarray, shape (n_targets,)
    """
    if metric == 'pearson_r':
        # Correlation per column
        Y_pred_c = Y_pred - Y_pred.mean(axis=0)
        Y_true_c = Y_true - Y_true.mean(axis=0)
        num = (Y_pred_c * Y_true_c).sum(axis=0)
        denom = np.sqrt((Y_pred_c ** 2).sum(axis=0) * (Y_true_c ** 2).sum(axis=0))
        return num / np.maximum(denom, 1e-12)
    elif metric == 'r2':
        ss_res = ((Y_true - Y_pred) ** 2).sum(axis=0)
        ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
        return 1 - ss_res / np.maximum(ss_tot, 1e-12)
    else:
        raise ValueError(f"Unknown metric '{metric}', expected 'r2' or 'pearson_r'")


class _Delayer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that applies temporal delays.

    Wraps ``core.array_utils.make_delayed`` so it can be used in sklearn
    pipelines (e.g. inside ``ColumnKernelizer``).
    """

    def __init__(self, delays=None):
        self.delays = delays if delays is not None else [0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return make_delayed(X, self.delays)

    def get_feature_names_out(self, input_features=None):
        n_in = len(input_features) if input_features is not None else 1
        names = []
        for d in self.delays:
            for i in range(n_in):
                feat = input_features[i] if input_features is not None else f"x{i}"
                names.append(f"{feat}_delay{d}")
        return np.array(names)


# ─── HimalayaRidgeModel ─────────────────────────────────────


@model("himalaya_ridge")
class HimalayaRidgeModel:
    """Cross-validated ridge regression via himalaya.

    Uses pre-delayed data from the default preprocessor.
    Wraps ``himalaya.ridge.RidgeCV``.
    """

    name = "himalaya_ridge"
    PARAM_SCHEMA = {
        "alphas": {"type": "string", "default": "logspace(-2,5,20)", "description": "Regularization values (logspace expression or list)"},
        "cv": {"type": "int", "default": 5, "min": 2, "description": "Cross-validation folds"},
        "score_metric": {"type": "string", "default": "r2", "enum": ["r2", "pearson_r"], "description": "Scoring metric"},
        "backend": {"type": "string", "enum": ["numpy", "cupy", "torch", "torch_cuda"], "description": "Compute backend"},
        "solver_params": {"type": "dict", "default": {},
                          "description": "RidgeCV solver parameters, e.g. {n_targets_batch: 10000, n_alphas_batch: 5} to cap memory"},
    }

    def fit(self, data: PreparedData, config: dict) -> ModelResult:
        from himalaya.ridge import RidgeCV

        model_cfg = config.get('model', {}).get('params', {})

        alphas = _resolve_alphas(model_cfg.get('alphas', 'logspace(-2,5,20)'))
        cv = model_cfg.get('cv', 5)
        score_metric = model_cfg.get('score_metric', 'r2')
        backend = model_cfg.get('backend', None)

        _set_backend(backend)

        ridge = RidgeCV(alphas=alphas, cv=cv, solver_params=dict(model_cfg.get('solver_params') or {}) or None)
        ridge.fit(data.X_train, data.Y_train)

        Y_pred = _to_numpy(ridge.predict(data.X_test))
        scores = _score_predictions(Y_pred, data.Y_test, metric=score_metric)

        return ModelResult(
            weights=_to_numpy(ridge.coef_),
            scores=scores,
            alphas=_to_numpy(ridge.best_alphas_),
            feature_names=data.feature_names,
            feature_dims=data.feature_dims,
            delays=data.delays,
        )

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        model_cfg = config.get('model', {}).get('params', {})
        cv = model_cfg.get('cv', 5)
        if not isinstance(cv, int) or cv < 2:
            errors.append(f"cv must be an integer >= 2, got {cv}")
        return errors


# ─── BandedRidgeModel ───────────────────────────────────────


@model("banded_ridge")
class BandedRidgeModel:
    """Banded ridge regression with per-feature-group regularization.

    Uses pre-delayed data. Computes group labels from ``feature_dims``
    and ``delays`` so each feature group x delay combination gets its
    own regularization strength.

    Wraps ``himalaya.ridge.BandedRidgeCV``.
    """

    name = "banded_ridge"
    PARAM_SCHEMA = {
        "alphas": {"type": "string", "default": "logspace(-2,5,20)", "description": "Regularization values (logspace expression or list)"},
        "cv": {"type": "int", "default": 5, "min": 2, "description": "Cross-validation folds"},
        "score_metric": {"type": "string", "default": "r2", "enum": ["r2", "pearson_r"], "description": "Scoring metric"},
        "backend": {"type": "string", "enum": ["numpy", "cupy", "torch", "torch_cuda"], "description": "Compute backend"},
        "solver_params": {"type": "dict", "default": {}, "description": "Solver-specific parameters"},
    }

    def fit(self, data: PreparedData, config: dict) -> ModelResult:
        from himalaya.ridge import BandedRidgeCV

        model_cfg = config.get('model', {}).get('params', {})

        alphas = _resolve_alphas(model_cfg.get('alphas', 'logspace(-2,5,20)'))
        cv = model_cfg.get('cv', 5)
        score_metric = model_cfg.get('score_metric', 'r2')
        backend = model_cfg.get('backend', None)
        solver_params = dict(model_cfg.get('solver_params', {}))

        _set_backend(backend)

        # Determine whether delays have already been applied to X_train.
        # Prefer explicit metadata; otherwise, infer/validate from shapes.
        metadata = getattr(data, "metadata", None) or {}
        if isinstance(metadata, dict) and "delays_applied" in metadata:
            delays_applied = bool(metadata["delays_applied"])
        else:
            n_features = data.X_train.shape[1]
            base_width = int(sum(data.feature_dims))
            n_delays = len(data.delays) if data.delays is not None else 0

            if n_delays == 0:
                # No delays configured; treat data as undelayed.
                delays_applied = False
            else:
                delayed_width = base_width * n_delays
                if n_features == delayed_width:
                    delays_applied = True
                elif n_features == base_width:
                    delays_applied = False
                else:
                    raise ValueError(
                        "Cannot infer whether delays have been applied: "
                        f"n_features={n_features}, base_width={base_width}, "
                        f"n_delays={n_delays}."
                    )
        groups = _compute_group_labels(
            data.feature_dims, data.delays, delays_applied,
        )

        # Inject alphas into solver_params if not already set
        if 'alphas' not in solver_params:
            solver_params['alphas'] = alphas

        ridge = BandedRidgeCV(
            groups=groups, solver_params=solver_params,
            cv=cv,
        )
        ridge.fit(data.X_train, data.Y_train)

        Y_pred = _to_numpy(ridge.predict(data.X_test))
        scores = _score_predictions(Y_pred, data.Y_test, metric=score_metric)

        metadata = {
            'deltas': _to_numpy(ridge.deltas_),
            'groups': groups,
        }

        return ModelResult(
            weights=_to_numpy(ridge.coef_),
            scores=scores,
            alphas=_to_numpy(ridge.best_alphas_),
            feature_names=data.feature_names,
            feature_dims=data.feature_dims,
            delays=data.delays,
            metadata=metadata,
        )

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        model_cfg = config.get('model', {}).get('params', {})
        cv = model_cfg.get('cv', 5)
        if not isinstance(cv, int) or cv < 2:
            errors.append(f"cv must be an integer >= 2, got {cv}")
        return errors


# ─── MultipleKernelRidgeModel ───────────────────────────────


@model("multiple_kernel_ridge")
class MultipleKernelRidgeModel:
    """Multiple kernel ridge regression via himalaya.

    Requires un-delayed data (``preparation.apply_delays: false``).
    Builds a pipeline with ``ColumnKernelizer`` that applies delays and
    kernelization per feature group, then fits ``MultipleKernelRidgeCV``
    with precomputed kernels.
    """

    name = "multiple_kernel_ridge"
    PARAM_SCHEMA = {
        "alphas": {"type": "string", "default": "logspace(-2,5,20)", "description": "Regularization values (logspace expression or list)"},
        "cv": {"type": "int", "default": 5, "min": 2, "description": "Cross-validation folds"},
        "solver": {"type": "string", "default": "random_search", "description": "Solver strategy"},
        "n_iter": {"type": "int", "default": 200, "min": 1, "description": "Solver iterations"},
        "score_metric": {"type": "string", "default": "r2", "enum": ["r2", "pearson_r"], "description": "Scoring metric"},
        "backend": {"type": "string", "enum": ["numpy", "cupy", "torch", "torch_cuda"], "description": "Compute backend"},
        "solver_params": {"type": "dict", "default": {}, "description": "Solver-specific parameters"},
        "delays": {"type": "list[int]", "description": "Delay values to apply per feature group"},
    }

    def fit(self, data: PreparedData, config: dict) -> ModelResult:
        from himalaya.kernel_ridge import (
            ColumnKernelizer, MultipleKernelRidgeCV,
        )

        model_cfg = config.get('model', {}).get('params', {})

        alphas = _resolve_alphas(model_cfg.get('alphas', 'logspace(-2,5,20)'))
        cv = model_cfg.get('cv', 5)
        solver = model_cfg.get('solver', 'random_search')
        n_iter = model_cfg.get('n_iter', 200)
        score_metric = model_cfg.get('score_metric', 'r2')
        backend = model_cfg.get('backend', None)
        solver_params = dict(model_cfg.get('solver_params', {}))

        _set_backend(backend)

        delays = data.delays if data.delays else model_cfg.get('delays', [0, 1, 2, 3])
        delays_applied = data.metadata.get('delays_applied', False)
        group_slices = _compute_group_slices(
            data.feature_dims, delays_applied, delays,
        )

        # Build per-group transformers: Delayer only
        # ColumnKernelizer handles kernelization internally — do NOT
        # include Kernelizer here or kernels will be double-computed.
        transformers = []
        for i, sl in enumerate(group_slices):
            name = data.feature_names[i] if i < len(data.feature_names) else f"group_{i}"
            if delays_applied:
                pipe = "passthrough"
            else:
                pipe = _Delayer(delays=delays)
            transformers.append((name, pipe, sl))

        column_kernelizer = ColumnKernelizer(transformers=transformers)

        # Build solver params
        solver_params.setdefault('n_iter', n_iter)
        solver_params.setdefault('alphas', alphas)

        mkr = MultipleKernelRidgeCV(
            kernels='precomputed',
            solver=solver,
            solver_params=solver_params,
            cv=cv,
        )

        # Transform and fit
        K_train = column_kernelizer.fit_transform(data.X_train)
        mkr.fit(K_train, data.Y_train)

        K_test = column_kernelizer.transform(data.X_test)
        # Backend-agnostic: works for numpy/cupy/torch/torch_cuda. The old
        # cupy-only `.get()` left torch_cuda tensors on the GPU and crashed
        # _score_predictions with "can't convert cuda:0 tensor to numpy".
        Y_pred = _to_numpy(mkr.predict(K_test))
        scores = _score_predictions(Y_pred, data.Y_test, metric=score_metric)

        # Also compute the *other* metric so reporters can render both
        # without us having to re-predict. Storing both is cheap (one
        # vector each of length n_voxels) and avoids having to keep the
        # fitted model object around.
        scores_pearson_r = _score_predictions(
            Y_pred, data.Y_test, metric='pearson_r')
        scores_r2 = _score_predictions(Y_pred, data.Y_test, metric='r2')

        metadata = {
            'deltas': _to_numpy(mkr.deltas_),
            'is_dual': True,
            'scores_pearson_r': scores_pearson_r,
            'scores_r2': scores_r2,
        }

        return ModelResult(
            weights=_to_numpy(mkr.dual_coef_),
            scores=scores,
            alphas=_to_numpy(mkr.best_alphas_),
            feature_names=data.feature_names,
            feature_dims=data.feature_dims,
            delays=data.delays,
            metadata=metadata,
        )

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        model_cfg = config.get('model', {}).get('params', {})
        cv = model_cfg.get('cv', 5)
        if not isinstance(cv, int) or cv < 2:
            errors.append(f"cv must be an integer >= 2, got {cv}")

        prep_cfg = config.get('preparation', {})
        prep_type = prep_cfg.get('type', 'default')
        if prep_type == 'default':
            apply_delays = prep_cfg.get('apply_delays', True)
            if apply_delays:
                errors.append(
                    "multiple_kernel_ridge requires preparation.apply_delays: false"
                )
        elif prep_type == 'pipeline':
            # Pipeline preparer: ensure no delay step is included
            # (delays are applied per-group inside ColumnKernelizer)
            step_names = [s.get('name') for s in prep_cfg.get('steps', [])]
            if 'delay' in step_names:
                errors.append(
                    "multiple_kernel_ridge applies delays internally; "
                    "remove the 'delay' step from preparation"
                )
        return errors
