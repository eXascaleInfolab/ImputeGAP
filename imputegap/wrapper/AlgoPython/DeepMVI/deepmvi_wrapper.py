# (BEGIN wrapper file)
from __future__ import annotations
import os
import numpy as np
import subprocess
from typing import Optional, Dict

PAPER_LOCAL_PATH = "/mnt/data/Research Paper DS.pdf"

class DeepMVI_Imputer:
    def __init__(self, params: Optional[Dict] = None):
        params = params or {}
        self.dataset = params.get("dataset", "electricity")
        self.project_root = params.get("project_root", os.path.abspath(os.getcwd()))
        self.run_script = params.get("run_script", os.path.join("scripts", "run_deepmvi_for_dataset.py"))
        self.output_path = os.path.join(self.project_root, "output_deepmvi", self.dataset, "imputed.npy")
        self._imputed: Optional[np.ndarray] = None
        self.params = params

    def _run_deepmvi_runner(self) -> None:
        script_path = os.path.join(self.project_root, self.run_script)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"DeepMVI runner script not found: {script_path}")
        cmd = ["python3", script_path, "--dataset", self.dataset]
        print("Running DeepMVI runner:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=self.project_root)

    def fit(self, X_obs: np.ndarray, mask: Optional[np.ndarray] = None) -> "DeepMVI_Imputer":
        if mask is not None:
            mask_path = os.path.join(self.project_root, "data", self.dataset, "A.npy")
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            np.save(mask_path, mask.astype(int))
            print(f"Saved availability mask to {mask_path}")

        if os.path.exists(self.output_path):
            try:
                self._imputed = np.load(self.output_path)
                print(f"Loaded existing imputed array from {self.output_path}")
                return self
            except Exception as e:
                print("Failed to load existing imputed file:", e)

        self._run_deepmvi_runner()

        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Runner completed but {self.output_path} not found")
        self._imputed = np.load(self.output_path)
        print(f"Loaded imputed array from {self.output_path}")
        return self

    def transform(self, X_missing: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self._imputed is None:
            if os.path.exists(self.output_path):
                self._imputed = np.load(self.output_path)
                print(f"Loaded existing imputed array from {self.output_path}")
            else:
                print("No imputed result found; using mean imputation fallback")
                return self._mean_impute(X_missing)

        if self._imputed.shape != X_missing.shape:
            if self._imputed.shape[0] == X_missing.shape[0] and self._imputed.shape[1] != X_missing.shape[1]:
                min_d = min(self._imputed.shape[1], X_missing.shape[1])
                print("Shape mismatch: slicing imputed to match requested feature dimension")
                return self._imputed[:, :min_d]
            raise ValueError(f"Imputed array shape {self._imputed.shape} does not match requested {X_missing.shape}")

        return self._imputed

    def _mean_impute(self, X_obs: np.ndarray) -> np.ndarray:
        X = X_obs.copy().astype(float)
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        return X

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="electricity")
    parser.add_argument("--project-root", default=os.path.abspath(os.getcwd()))
    args = parser.parse_args()

    x_path = os.path.join(args.project_root, "data", args.dataset, "X.npy")
    if not os.path.exists(x_path):
        print("Data X.npy not found at:", x_path)
        raise SystemExit(1)
    X = np.load(x_path)
    a_path = os.path.join(args.project_root, "data", args.dataset, "A.npy")
    A = np.load(a_path) if os.path.exists(a_path) else None

    imputer = DeepMVI_Imputer({"dataset": args.dataset, "project_root": args.project_root})
    imputer.fit(X, A)
    X_imp = imputer.transform(X, A)
    print("Imputed shape:", X_imp.shape)
    out = os.path.join(args.project_root, "output_deepmvi", args.dataset, "imputed_from_wrapper.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, X_imp)
    print("Saved wrapper-produced imputed to", out)
# (END wrapper file)
