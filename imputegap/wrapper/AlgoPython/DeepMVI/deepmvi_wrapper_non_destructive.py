# Non-destructive wrapper: never runs the runner, never writes A.npy, never overwrites imputed.npy
from __future__ import annotations
import os, numpy as np
from typing import Optional, Dict

class DeepMVI_Imputer_ND:
    def __init__(self, params: Optional[Dict] = None):
        params = params or {}
        self.dataset = params.get("dataset", "electricity")
        self.project_root = params.get("project_root", os.path.abspath(os.getcwd()))
        self.output_path = os.path.join(self.project_root, "output_deepmvi", self.dataset, "imputed.npy")
        self._imputed = None

    def fit(self, X_obs: np.ndarray, mask: Optional[np.ndarray] = None):
        # intentionally do NOT save mask or run the runner
        if os.path.exists(self.output_path):
            self._imputed = np.load(self.output_path)
            print("Loaded existing imputed:", self.output_path)
        else:
            print("No existing imputed.npy found; will fallback to mean on transform()")
        return self

    def transform(self, X_missing: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
        if self._imputed is not None:
            if self._imputed.shape == X_missing.shape:
                return self._imputed
            # try safe slice/pad
            min_d = min(self._imputed.shape[1], X_missing.shape[1])
            return self._imputed[:, :min_d]
        # fallback to mean impute
        X = X_missing.copy().astype(float)
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return X
