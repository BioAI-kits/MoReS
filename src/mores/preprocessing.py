"""Training-partition-only imputation, scaling, and nested LASSO-Cox selection."""
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from .data import MODALITIES


def splits(event, folds, seed):
    event = np.asarray(event, int)
    counts = np.bincount(event, minlength=2)
    if counts.min() < folds:
        raise ValueError(f"Need >= {folds} events and censored patients for stratified folds; got {counts.tolist()}")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(event)), event))


def truncated_y(data, tau):
    return Surv.from_arrays(data.event.astype(bool) & (data.time <= tau), np.minimum(data.time, tau))


class NumericTransform:
    def fit(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median)
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        return self

    def transform(self, x):
        return (np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale


class ClinicalTransform:
    """Fixed sex/stage categories; missing stage never receives ordinal stage 0."""
    @staticmethod
    def raw(frame):
        age = pd.to_numeric(frame.age, errors="coerce").to_numpy(float)
        sex = frame.sex.fillna("missing").astype(str).str.lower().to_numpy()
        stage = []
        for value in frame.stage:
            s = str(value).strip().upper()
            match = re.fullmatch(r"(?:STAGE\s*)?(IV|III|II|I)[ABC]?", s)
            stage.append(match.group(1) if match else "missing")
        categorical = np.column_stack([sex == v for v in ("female", "male", "missing")] +
                                      [np.array(stage) == v for v in ("I", "II", "III", "IV", "missing")])
        # Unknown sex is mapped to the explicit missing category.
        categorical[:, 2] = ~np.isin(sex, ["female", "male"])
        return age[:, None], categorical.astype(float)

    def fit(self, frame):
        age, _ = self.raw(frame)
        self.numeric = NumericTransform().fit(age)
        return self

    def transform(self, frame):
        age, categorical = self.raw(frame)
        return np.column_stack([self.numeric.transform(age), np.isfinite(age), categorical]).astype("float32")


def candidate_columns(x, cap):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.nanvar(x, axis=0)
    candidates = np.flatnonzero(np.isfinite(variance) & (variance > 1e-12))
    if cap is not None and len(candidates) > cap:
        candidates = candidates[np.argsort(-variance[candidates], kind="stable")[:cap]]
    return candidates


def fit_cox(x, y, alpha):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="all coefficients are zero.*")
        warnings.simplefilter("error", ConvergenceWarning)
        return CoxnetSurvivalAnalysis(l1_ratio=1., alphas=[alpha], max_iter=100000).fit(x, y)


def select_features(x, y, cfg, seed):
    """Tune LASSO penalties with fold-local filtering, imputation, scaling.

    Empty or short nonzero selections are retained without artificial padding.
    """
    folds = splits(y["event"], cfg.selection_folds, seed)
    scores = []
    for alpha in cfg.cox_alphas:
        fold_scores = []
        for tr, va in folds:
            ix = candidate_columns(x[tr], cfg.max_candidates)
            if not len(ix):
                fold_scores.append(.5)
                continue
            scaler = NumericTransform().fit(x[tr][:, ix])
            try:
                model = fit_cox(scaler.transform(x[tr][:, ix]), y[tr], alpha)
                fold_scores.append(model.score(scaler.transform(x[va][:, ix]), y[va]))
            except (ArithmeticError, ConvergenceWarning):
                # Reject a numerically invalid candidate; never silently use
                # its unfinished coefficients or switch to another estimator.
                fold_scores.append(-np.inf)
        scores.append(np.mean(fold_scores))
    candidates = candidate_columns(x, cfg.max_candidates)
    if not len(candidates):
        return np.array([], int), None
    values = np.asarray(x[:, candidates], float)
    transform = NumericTransform().fit(values)
    model = None
    for index in np.argsort(-np.asarray(scores), kind="stable"):
        if not np.isfinite(scores[index]):
            continue
        alpha = cfg.cox_alphas[index]
        try:
            model = fit_cox(transform.transform(values), y, alpha)
            break
        except (ArithmeticError, ConvergenceWarning):
            continue
    if model is None:
        raise ValueError("All LASSO-Cox candidates failed numerically; increase cox_alphas or training size")
    coef = model.coef_[:, 0]
    nonzero = np.flatnonzero(np.abs(coef) > 1e-10)
    order = nonzero[np.argsort(-np.abs(coef[nonzero]), kind="stable")[:cfg.top_features]]
    return candidates[order], alpha


class Preprocessor:
    def fit(self, data, cfg, seed):
        self.fit_ids = data.ids.tolist()
        self.clinical = ClinicalTransform().fit(data.clinical)
        self.selected, self.scalers, self.alphas = {}, {}, {}
        y = truncated_y(data, cfg.tau)
        for j, m in enumerate(MODALITIES):
            usable = data.available[:, j].astype(bool)
            if not usable.any():
                self.selected[m], self.scalers[m], self.alphas[m] = [], None, None
                continue
            ix, alpha = select_features(data.x[m][usable], y[usable], cfg, seed + j)
            self.selected[m] = [data.features[m][k] for k in ix]
            self.scalers[m] = NumericTransform().fit(data.x[m][usable][:, ix])
            self.alphas[m] = alpha
        return self

    def transform(self, data):
        result, coverage = {}, []
        for j, m in enumerate(MODALITIES):
            positions = {name: i for i, name in enumerate(data.features[m])}
            selected = self.selected[m]
            a = np.full((len(data), len(selected)), np.nan)
            for k, name in enumerate(selected):
                if name in positions:
                    a[:, k] = data.x[m][:, positions[name]]
            a[~data.available[:, j].astype(bool)] = np.nan
            mask = np.isfinite(a)
            # Coverage of model-selected features. Empty selection is explicit 0.
            cov = mask.mean(axis=1) if mask.shape[1] else np.zeros(len(data))
            scaled = self.scalers[m].transform(a) if self.scalers[m] is not None else a
            v = np.column_stack([scaled, mask, cov]).astype("float32")
            v[~data.available[:, j].astype(bool)] = 0
            result[m] = v
            coverage.append(cov)
        result["clinical"] = self.clinical.transform(data.clinical)
        result["clinical"][~data.available[:, 4].astype(bool)] = 0
        return result, np.column_stack(coverage), data.available.astype(bool).copy()

    def project(self, data):
        """Keep only fitted features before raw-scale augmentation."""
        from .data import Cohort
        x = {}
        for m in MODALITIES:
            lookup = {f: k for k, f in enumerate(data.features[m])}
            selected = self.selected[m]
            a = np.full((len(data), len(selected)), np.nan, dtype="float32")
            for k, f in enumerate(selected):
                if f in lookup:
                    a[:, k] = data.x[m][:, lookup[f]]
            x[m] = a
        return Cohort(data.ids, data.clinical, x, self.selected, data.available.copy(), data.time, data.event)
