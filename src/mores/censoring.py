"""Fold-local censoring models, left limits, IPCW, and gain pseudo-outcomes."""
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from .preprocessing import ClinicalTransform
from .survival import quadrature


class CensoringModel:
    def fit(self, data, cfg):
        self.cfg = cfg
        self.fit_ids = data.ids.tolist()
        self.time = data.time.copy()  # Original, not administratively truncated.
        self.clinical = ClinicalTransform().fit(data.clinical)
        self.model = None
        censored = ~data.event.astype(bool)
        self.constant = not censored.any()
        if self.constant:
            self.knots, self.km = np.array([0.]), np.array([1.])
        elif cfg.censoring == "km":
            self.knots, self.km = kaplan_meier_estimator(data.event.astype(bool), data.time, reverse=True)
        else:
            x = self.features(data)
            self.variable = x.std(axis=0) > 1e-10
            # sksurv Cox with ridge, Breslow ties; no automatic KM fallback.
            if not self.variable.any():
                self.knots, self.km = kaplan_meier_estimator(data.event.astype(bool), data.time, reverse=True)
                self.constant_covariates = True
            else:
                self.model = CoxPHSurvivalAnalysis(alpha=cfg.censor_alpha, ties="breslow", n_iter=200)
                self.model.fit(x[:, self.variable], Surv.from_arrays(censored, data.time))
                self.knots = self.model.unique_times_
        return self

    def features(self, data):
        return np.column_stack([self.clinical.transform(data.clinical), data.available])

    def survival(self, data, times, left=False):
        times = np.asarray(times, float)
        idx = np.searchsorted(self.knots, times, side="left" if left else "right") - 1
        if self.model is None:
            values = np.where(idx >= 0, self.km[np.maximum(idx, 0)], 1.)
            return np.broadcast_to(values, (len(data), len(times))).copy()
        functions = self.model.predict_survival_function(self.features(data)[:, self.variable])
        return np.array([np.where(idx >= 0, f.y[np.maximum(idx, 0)], 1.) for f in functions])

    def survival_at_observations(self, data):
        """Evaluate G_i(Y_i-) without constructing an n-by-n matrix."""
        idx = np.searchsorted(self.knots, data.time, side="left") - 1
        if self.model is None:
            return np.where(idx >= 0, self.km[np.maximum(idx, 0)], 1.)
        functions = self.model.predict_survival_function(self.features(data)[:, self.variable])
        return np.array([f.y[k] if k >= 0 else 1. for f, k in zip(functions, idx)])

    def weights(self, data, grid, strict=True):
        g = self.survival(data, grid)
        gy = self.survival_at_observations(data)
        alive = data.time[:, None] > grid
        dead = (data.time[:, None] <= grid) & data.event[:, None].astype(bool)
        # Positivity is checked BEFORE clipping. Events outside the window do
        # not require an IPCW denominator.
        support = ((g >= self.cfg.support_g).all(axis=0) &
                   ((self.time[:, None] >= grid).sum(axis=0) >= self.cfg.min_at_risk) &
                   (~dead | (gy[:, None] >= self.cfg.support_g)).all(axis=0))
        if strict and not support.all():
            raise ValueError(f"Insufficient censoring support at {grid[np.flatnonzero(~support)[0]]:.3f} months; shorten the prespecified horizon or inspect support")
        a = alive / np.maximum(g, self.cfg.g_floor)
        w = a + dead / np.maximum(gy[:, None], self.cfg.g_floor)
        used = alive.sum() + dead.sum()
        clipped = ((g < self.cfg.g_floor) & alive).sum() + ((gy[:, None] < self.cfg.g_floor) & dead).sum()
        return a, w, support, float(clipped/max(used, 1))


def gain_pseudo_outcome(prediction, original_data, censoring):
    cfg = censoring.cfg
    grid, weights = quadrature(cfg.tau, cfg.cuts,
                               np.r_[original_data.time, censoring.knots], cfg.integration_order)
    a, w, _, clipped = censoring.weights(original_data, grid)
    s0, s1 = prediction.curves(grid)
    b = ((s1-s0) * (a-w*s0)) @ weights
    return b, clipped
