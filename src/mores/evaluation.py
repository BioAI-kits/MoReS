"""Frozen external evaluation; perturbations leave censoring covariates intact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .augmentation import perturb
from .survival import quadrature
from .metrics import ibs, brier, dynamic_auc


def evaluate_frozen(model, data, output, horizon=36., perturbation=None, severity=.15):
    if not 0 < horizon <= model.cfg.tau:
        raise ValueError("Evaluation horizon must lie inside the trained horizon")
    overlap = set(data.ids) & set(model.provenance["fit_ids"])
    if overlap:
        raise ValueError("Frozen evaluation patients overlap training; use outer CV for internal evaluation")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    censor = model.upstream.censoring
    grid, integration = quadrature(horizon, model.cfg.cuts, np.r_[data.time, censor.knots], model.cfg.integration_order)
    a, w, support, clipped = censor.weights(data, grid, strict=False)
    ap, wp, point_support, _ = censor.weights(data, np.array([horizon]), strict=False)
    inputs = data if perturbation is None else perturb(data, severity, np.random.default_rng(model.cfg.seed),
                                                       kind=perturbation, rna_scale=model.cfg.rna_scale)
    prediction = model.predict(inputs, grid)
    at_horizon = model.predict(inputs, [horizon])
    rows = []
    for name in ["reference", "joint", "no_shrink", "mores"]:
        rows.append({"model": name, "n": len(data), "events": int(data.event.sum()), "horizon": horizon,
            "model_scope": "full fusion eligible" if prediction["eligible"].any() else "reference only",
            "n_joint_eligible": int(prediction["eligible"].sum()),
            "ibs": ibs(prediction[name], a, w, integration) if support.all() else np.nan,
            "auc": dynamic_auc(at_horizon[name], data.time, data.event, [horizon], ap, wp)[0] if point_support[0] else np.nan,
            "brier": brier(at_horizon[name], ap, wp)[0] if point_support[0] else np.nan,
            "clipped_fraction": clipped,
            "status": "ok" if support.all() else "insufficient_censoring_support"})
    pd.DataFrame(rows).to_csv(root/"metrics.csv", index=False)
    np.savez_compressed(root/"predictions.npz", patient_ids=data.ids.astype(str), support=support, **prediction)
    (root/"evaluation.json").write_text(json.dumps({"horizon": horizon, "perturbation": perturbation,
        "severity": severity, "censoring_source": "frozen training model", "target_refit": False,
        "outcome_unchanged": True}, indent=2))
    return rows
