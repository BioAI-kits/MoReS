"""Outcome-independent perturbations on raw prepared inputs."""
import numpy as np
from .data import Cohort, MODALITIES

KINDS = ("rna_mask", "methylation_noise", "cnv_rescale", "mutation_missing", "modality_drop", "platform_shift")


def perturb(data, severity, rng, kind=None, rna_scale="log2p1"):
    if not 0 <= severity < 1:
        raise ValueError("severity must be in [0,1)")
    kind = rng.choice(KINDS[:-1]) if kind is None else kind
    if kind not in KINDS:
        raise ValueError(f"Unknown perturbation {kind}")
    x = {m: a.copy() for m, a in data.x.items()}
    available = data.available.copy()
    if kind in {"rna_mask", "platform_shift"}:
        a = x["rna"]
        if kind == "rna_mask":
            a[rng.random(a.shape) < severity] = np.nan
        else:
            if rna_scale == "log2p1":
                # Gene-wise scale shift on expression scale, then back to log.
                scale = np.exp(rng.normal(0, severity, a.shape[1]))
                a[:] = np.log2(np.maximum(0., np.exp2(a)-1.)*scale + 1.)
            else:
                a *= np.exp(rng.normal(0, severity, a.shape[1]))
    elif kind == "methylation_noise":
        x["methylation"] = np.clip(x["methylation"] + rng.normal(0, severity/2, x["methylation"].shape), 0, 1)
    elif kind == "cnv_rescale":
        x["cnv"] *= np.exp(rng.normal(0, severity, (len(data), 1)))
    elif kind == "mutation_missing":
        a = x["mutation"]
        a[rng.random(a.shape) < severity] = np.nan
    elif kind == "modality_drop":
        for i in np.flatnonzero(rng.random(len(data)) < severity):
            candidates = np.flatnonzero(available[i, :4])
            if len(candidates) and available[i].sum() > 1:
                j = rng.choice(candidates)
                available[i, j] = False
                x[MODALITIES[j]][i] = np.nan
    # Empty observations are missing modalities; retain at least one expert.
    for j, m in enumerate(MODALITIES):
        if x[m].shape[1]:
            empty = ~np.isfinite(x[m]).any(axis=1)
            available[empty, j] = False
    none = ~available.any(axis=1)
    for m in MODALITIES:
        x[m][none] = data.x[m][none]
    available[none] = data.available[none]
    return Cohort(data.ids.copy(), data.clinical.copy(), x, data.features, available, data.time, data.event)
