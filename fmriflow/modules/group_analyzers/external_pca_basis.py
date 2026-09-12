"""Load a precomputed semantic PCA basis from an HDF file.

Sibling to :mod:`fmriflow.modules.group_analyzers.stacked_weights_pca`,
but instead of building the basis from the current cohort's weights it
reuses one that was estimated elsewhere (e.g. a co-occurrence-semantics
PCA distributed alongside a feature space as its canonical semantic
subspace).

The basis is typically stored under ``c`` in the source HDF as
``(fdim, fdim)``, one component per row, ordered by descending variance.
The analyzer takes the first ``n_components`` rows and expands them to
delayed-feature space by tiling across delays so the downstream
:class:`~fmriflow.modules.analyzers.project_to_subspace.ProjectToSubspaceAnalyzer`
projection ``basis.T @ block`` yields the same result as projecting the
per-delay-averaged weights with the raw basis.

This implements the standard "project model weights onto a pre-existing
semantic subspace" pattern — useful when you want every subject's
weights interpreted in a shared coordinate system independent of the
current cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fmriflow.core.group_types import GroupResult
from fmriflow.core.types import ModelResult, SemanticSubspace
from fmriflow.modules._decorators import group_analyzer
from fmriflow.modules.group_analyzers._helpers import my_cfg

logger = logging.getLogger(__name__)


@group_analyzer("external_pca_basis")
class ExternalPCABasisAnalyzer:
    """Plant a precomputed semantic-PC basis as a subject second-pass binding.

    Triggers the orchestrator's second-pass mechanism via
    ``produces_subject_artifact = True`` — after the analyzer runs, every
    subject's ``analyze + report`` stages re-run with the loaded basis
    bound into context under ``external.<binding_name>``.
    """

    name = "external_pca_basis"
    produces_subject_artifact = True
    PARAM_SCHEMA = {
        "path": {
            "type": "path",
            "required": True,
            "description": "HDF5 file holding the basis as the named dataset.",
        },
        "dataset": {
            "type": "str",
            "default": "c",
            "description": (
                "Name of the (fdim, fdim) dataset inside the HDF holding "
                "the PCs as columns, ordered by descending variance. "
                "Many precomputed semantic-PC files use the dataset key 'c'."
            ),
        },
        "singular_values_dataset": {
            "type": "str",
            "default": "l",
            "description": (
                "Optional name of a (fdim,) dataset of singular values; "
                "if absent the analyzer falls back to ones."
            ),
        },
        "feature": {
            "type": "str",
            "required": True,
            "description": (
                "Feature whose weight block this basis lives in (e.g. "
                "'english1000'). Must match the feature name in each "
                "subject's ModelResult so the second-pass projection "
                "can slice the right rows."
            ),
        },
        "n_components": {
            "type": "int",
            "default": 50,
            "description": "Number of leading PCs to keep.",
        },
        "output_key": {
            "type": "str",
            "description": (
                "Group artifact key. Defaults to "
                "'group.<feature>_pca_basis'."
            ),
        },
        "binding_name": {
            "type": "str",
            "description": (
                "Bare name (no 'external.' prefix) under which the "
                "basis is bound into each subject's context during the "
                "second pass. Defaults to '<feature>_pca_basis'."
            ),
        },
    }

    def analyze(self, group: GroupResult, config: dict) -> None:
        import h5py

        cfg = my_cfg(config, self.name)
        path = cfg.get("path")
        if not path:
            raise ValueError("external_pca_basis: 'path' is required")
        feature = cfg.get("feature")
        if not feature:
            raise ValueError("external_pca_basis: 'feature' is required")
        n_components = int(cfg.get("n_components", 50))
        dataset = cfg.get("dataset", "c")
        sv_dataset = cfg.get("singular_values_dataset", "l")
        output_key = cfg.get("output_key", f"group.{feature}_pca_basis")

        hdf_path = Path(path)
        if not hdf_path.is_file():
            raise FileNotFoundError(
                f"external_pca_basis: HDF not found at {hdf_path}")
        with h5py.File(hdf_path, "r") as h:
            if dataset not in h:
                raise KeyError(
                    f"external_pca_basis: dataset '{dataset}' not in "
                    f"{hdf_path} (available: {list(h.keys())})"
                )
            raw_basis = np.asarray(h[dataset], dtype=np.float64)
            singular = (
                np.asarray(h[sv_dataset], dtype=np.float64)
                if sv_dataset in h else None
            )

        if raw_basis.ndim != 2 or raw_basis.shape[0] != raw_basis.shape[1]:
            raise ValueError(
                f"external_pca_basis: expected square (fdim, fdim) basis "
                f"at '{dataset}', got shape {raw_basis.shape}"
            )

        # Pull n_delays + fdim from a sample subject's ModelResult so we
        # can expand the raw basis into delayed-feature space (the space
        # the model weights actually live in).
        n_delays, fdim = self._subject_delay_dims(group, feature)
        if fdim != raw_basis.shape[0]:
            raise ValueError(
                f"external_pca_basis: basis fdim={raw_basis.shape[0]} but "
                f"subjects' feature '{feature}' has fdim={fdim}"
            )

        k = min(n_components, raw_basis.shape[1])
        # Rows of the stored basis are the components (row k = PC k), so keep the
        # first k rows and transpose: the downstream ``basis.T @ block`` then equals
        # ``c[:k] @ mean_d(block_d)``, the same projection as ``basis @ weights``.
        raw_basis = raw_basis[:k, :].T                            # (fdim, K)
        # Tiling + scaling so ``basis.T @ block`` matches projecting the
        # per-delay-averaged weight block with the raw basis:
        #     mean_d(basis^T @ block_d) = basis^T @ mean_d(block_d)
        # which equals ``tile(basis/n_delays, n_delays).T @ block_stack``.
        basis = np.tile(raw_basis / float(n_delays), (n_delays, 1))  # (n_delays*fdim, K)
        sv = (singular[:k] if singular is not None
              else np.ones(k, dtype=np.float64))

        subspace = SemanticSubspace(
            basis=basis,
            singular_values=sv,
            feature=feature,
            n_delays=n_delays,
            feature_dim=fdim,
            metadata={
                "source": str(hdf_path),
                "dataset": dataset,
                "raw_basis_shape": list(raw_basis.shape),
                "delay_expansion": "tile-and-scale",
            },
        )
        group.put(output_key, subspace)
        binding_name = cfg.get("binding_name", f"{feature}_pca_basis")
        group.put(f"{output_key}.binding_name", binding_name)

    def subject_bindings(self, group: GroupResult) -> dict[str, object]:
        """Same convention as stacked_weights_pca — bind every
        :class:`SemanticSubspace` artifact under its recorded
        binding_name. Lets one config compose external + computed bases
        side-by-side without colliding."""
        bindings: dict[str, object] = {}
        for key, value in group.artifacts.items():
            if not isinstance(value, SemanticSubspace):
                continue
            binding_name_key = f"{key}.binding_name"
            name = group.artifacts.get(
                binding_name_key, f"{value.feature}_pca_basis")
            bindings[name] = value
        return bindings

    def validate_config(self, config: dict) -> list[str]:
        cfg = my_cfg(config, self.name)
        errors: list[str] = []
        if not cfg.get("path"):
            errors.append("external_pca_basis: 'path' is required")
        elif not Path(cfg["path"]).is_file():
            errors.append(f"external_pca_basis: file not found: {cfg['path']}")
        if not cfg.get("feature"):
            errors.append("external_pca_basis: 'feature' is required")
        n = cfg.get("n_components", 50)
        if not isinstance(n, int) or n <= 0:
            errors.append("external_pca_basis: 'n_components' must be positive int")
        return errors

    @staticmethod
    def _subject_delay_dims(group: GroupResult, feature: str) -> tuple[int, int]:
        """Return (n_delays, fdim) inferred from the first subject that has a
        ModelResult with this feature."""
        for sr in group.subjects:
            if sr.context is None or not sr.context.has("result"):
                continue
            result = sr.context.get("result", ModelResult)
            if feature not in result.feature_names:
                continue
            i = result.feature_names.index(feature)
            return len(result.delays), int(result.feature_dims[i])
        raise ValueError(
            f"external_pca_basis: no subject has a ModelResult containing "
            f"feature '{feature}' — can't determine n_delays / fdim"
        )
