"""Training, validation-only fusion tuning, and pooled repeated outer CV."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import logging
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from .censoring import CensoringModel
from .gain import GainEnsemble, cross_fitted_records, summary
from .metrics import ibs, brier, dynamic_auc, concordance, calibration, uno_concordance
from .preprocessing import splits
from .survival import quadrature, robust_weights
from .training import fit_upstream

log = logging.getLogger(__name__)


@dataclass
class MoReS:
    upstream: object
    gain: GainEnsemble
    cfg: object
    kappa: float
    eta: float
    provenance: dict

    def predict(self, data, times=None):
        times = np.asarray(self.cfg.cuts if times is None else times, float)
        if not np.isfinite(times).all() or (times < 0).any() or (times > self.cfg.tau).any():
            raise ValueError("Prediction times must lie inside the trained horizon")
        prediction = self.upstream.predict(data)
        z, q = summary(self.upstream, data, prediction)
        mean, scale = self.gain.predict(z, q)
        alpha, ell = robust_weights(q, mean, scale, self.kappa, self.eta, prediction.eligible)
        s0, s1 = prediction.curves(times)
        no_shrink = np.where((q > 0) & prediction.eligible, np.clip(mean/np.maximum(q, 1e-15), 0, 1), 0.)
        return {"times": times, "reference": s0, "joint": s1,
                "mores": s0+alpha[:, None]*(s1-s0),
                "no_shrink": s0+no_shrink[:, None]*(s1-s0),
                "alpha": alpha, "alpha_no_shrink": no_shrink, "q": q,
                "gain_mean": mean, "gain_scale": scale, "lower_gain": ell,
                "eligible": prediction.eligible}

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)  # Only load locally trusted bundles.
        if not isinstance(obj, cls):
            raise TypeError("Not a MoReS model bundle")
        return obj


def fit_components(data, cfg, seed):
    records = cross_fitted_records(data, cfg, seed)
    gain = GainEnsemble().fit(records, cfg, seed+301)
    upstream = fit_upstream(data, cfg, seed+701)
    provenance = {"fit_ids": data.ids.tolist(), "gain_cross_folds": records.provenance,
                  "ridge_folds": gain.tuning_provenance, "reference_folds": upstream.stack_provenance,
                  "selected_features": upstream.preprocessor.selected,
                  "cox_alphas": upstream.preprocessor.alphas,
                  "ridge_alpha": gain.alpha, "network_epochs": upstream.networks.epochs,
                  "network_configuration": asdict(upstream.networks.cfg)}
    return upstream, gain, provenance


def fit(data, cfg, seed=None):
    data.validate(True)
    cfg.validate()
    seed = cfg.seed if seed is None else seed
    candidates = [(k, e) for k in cfg.kappa_grid for e in cfg.eta_grid]
    scores = [[] for _ in candidates]
    validation_provenance = []
    for f, (tr, va) in enumerate(splits(data.event, cfg.inner_folds, seed)):
        log.info("Fusion tuning fold %d/%d (%d training patients)", f+1, cfg.inner_folds, len(tr))
        train, valid = data.subset(tr), data.subset(va)
        upstream, gain, provenance = fit_components(train, cfg, seed+10007*(f+1))
        prediction = upstream.predict(valid)
        z, q = summary(upstream, valid, prediction)
        mean, scale = gain.predict(z, q)
        grid, integration = quadrature(cfg.tau, cfg.cuts, np.r_[upstream.censoring.knots, valid.time], cfg.integration_order)
        a, w, _, clipped = upstream.censoring.weights(valid, grid)
        s0, s1 = prediction.curves(grid)
        for j, (kappa, eta) in enumerate(candidates):
            alpha, _ = robust_weights(q, mean, scale, kappa, eta, prediction.eligible)
            scores[j].append(ibs(s0+alpha[:, None]*(s1-s0), a, w, integration))
        validation_provenance.append({"fit_ids": train.ids.tolist(), "prediction_ids": valid.ids.tolist(),
                                      "components": provenance, "clipped_weight_fraction": clipped})
    best = int(np.argmin(np.mean(scores, axis=1)))
    kappa, eta = candidates[best]
    log.info("Refitting all %d training patients; kappa=%g eta=%g", len(data), kappa, eta)
    upstream, gain, provenance = fit_components(data, cfg, seed+900001)
    provenance["fusion_validation"] = validation_provenance
    provenance["fusion_search"] = [{"kappa": k, "eta": e, "mean_validation_ibs": float(np.mean(s))}
                                   for (k, e), s in zip(candidates, scores)]
    return MoReS(upstream, gain, cfg, kappa, eta, provenance)


def runtime_info():
    import scipy, sklearn, sksurv
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "sksurv": sksurv.__version__}


def cross_validate(data, cfg, output):
    """Pool outer OOF predictions before metrics, then summarize repetitions."""
    data.validate(True); cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg.save(output / "config.json")
    (output / "runtime.json").write_text(json.dumps(runtime_info(), indent=2))
    rows, calibration_rows = [], []
    # Shared integration grid for every model and fold; outcomes define only
    # evaluation integration discontinuities, not fitted models or gain inputs.
    grid, integration = quadrature(cfg.tau, cfg.cuts, data.time, cfg.integration_order)
    points = np.array([t for t in [6., 12., 24., 36., 60.] if t <= cfg.tau])
    for repeat in range(cfg.repeats):
        oof = {m: np.zeros((len(data), len(grid))) for m in ("reference", "joint", "no_shrink", "mores")}
        point_curves = {m: np.zeros((len(data), len(points))) for m in oof}
        diagnostics = {k: np.zeros(len(data)) for k in ["alpha", "alpha_no_shrink", "q", "gain_mean", "gain_scale", "lower_gain", "eligible"]}
        a = np.zeros((len(data), len(grid))); w = np.zeros_like(a)
        ap = np.zeros((len(data), len(points))); wp = np.zeros_like(ap)
        g_events = np.zeros((len(data), len(data)))
        marginal_g = np.zeros(len(data))
        support = np.ones(len(grid), bool); point_support = np.ones(len(points), bool)
        fold_id = np.zeros(len(data), int)
        clips = []
        for f, (tr, va) in enumerate(splits(data.event, cfg.outer_folds, cfg.seed+repeat)):
            log.info("Outer repeat %d/%d fold %d/%d", repeat+1, cfg.repeats, f+1, cfg.outer_folds)
            train, valid = data.subset(tr), data.subset(va)
            model = fit(train, cfg, cfg.seed+repeat*1000003+f*100003)
            pred, pred_points = model.predict(valid, grid), model.predict(valid, points)
            for name in oof:
                oof[name][va] = pred[name]
                point_curves[name][va] = pred_points[name]
            for name in diagnostics:
                diagnostics[name][va] = pred[name]
            censor = model.upstream.censoring
            a[va], w[va], supported, clipped = censor.weights(valid, grid, strict=False)
            ap[va], wp[va], supported_points, _ = censor.weights(valid, points, strict=False)
            g_events[va] = censor.survival(valid, data.time, left=True)
            km = CensoringModel().fit(train, replace(cfg, censoring="km"))
            marginal_g[va] = km.survival_at_observations(valid)
            support &= supported
            point_support &= supported_points
            fold_id[va] = f
            clips.append(clipped)
            (output / f"repeat-{repeat:02d}-fold-{f:02d}.json").write_text(json.dumps(model.provenance, indent=2))
        # Unsupported horizons remain NA. No adaptive clipping to manufacture
        # evaluability; common support is reported for all compared models.
        for name in oof:
            auc = dynamic_auc(point_curves[name], data.time, data.event, points, ap, wp)
            bs = brier(point_curves[name], ap, wp)
            row = {"repeat": repeat, "model": name,
                   "ibs": ibs(oof[name], a, w, integration) if support.all() else np.nan,
                   "ipcw_cindex": concordance(1-point_curves[name][:, -1], data.time, data.event,
                                               g_events, cfg.tau, cfg.g_floor, cfg.support_g) if support.all() else np.nan,
                   "uno_cindex": uno_concordance(1-point_curves[name][:, -1], data.time, data.event,
                                                marginal_g, cfg.tau, cfg.g_floor, cfg.support_g) if support.all() else np.nan,
                   "metric_status": "ok" if support.all() else "insufficient_censoring_support",
                   "clipped_fraction_fold_mean": float(np.mean(clips))}
            for j, t in enumerate(points):
                row[f"auc_{t:g}m"] = auc[j] if point_support[j] else np.nan
                row[f"brier_{t:g}m"] = bs[j] if point_support[j] else np.nan
                if point_support[j]:
                    for group, values in enumerate(calibration(point_curves[name][:, j], ap[:, j])):
                        calibration_rows.append({"repeat": repeat, "model": name, "month": t, "group": group, **values})
            rows.append(row)
        np.savez_compressed(output / f"oof-{repeat:02d}.npz", patient_ids=data.ids.astype(str), fold=fold_id,
                            times=grid, time_weights=integration, support=support, time=data.time, event=data.event,
                            **oof, **diagnostics)
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(calibration_rows).to_csv(output / "calibration.csv", index=False)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.select_dtypes(include="number").columns if c != "repeat"]
    frame.groupby("model")[numeric].agg(["mean", "std", "count"]).to_csv(output / "summary.csv")
    return frame
