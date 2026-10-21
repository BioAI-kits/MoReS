"""IPCW metrics with common time support and explicit non-estimability."""
import numpy as np


def brier(curves, a, w):
    return (a - 2*a*curves + w*curves**2).mean(axis=0)


def ibs(curves, a, w, integration):
    return float(brier(curves, a, w) @ integration)


def dynamic_auc(curves, time, event, grid, a, w):
    result = []
    for j, t in enumerate(grid):
        cases = (time <= t) & event.astype(bool)
        controls = time > t
        wc, wo = w[cases, j], a[controls, j]
        denominator = wc.sum()*wo.sum()
        if denominator <= 0:
            result.append(np.nan)
            continue
        risk_case, risk_control = 1-curves[cases, j], 1-curves[controls, j]
        diff = risk_case[:, None]-risk_control[None, :]
        credit = (diff > 1e-8) + .5*(np.abs(diff) <= 1e-8)
        result.append(float(np.sum(credit*wc[:, None]*wo[None, :])/denominator))
    return np.asarray(result)


def concordance(risk, time, event, g_at_events, tau, g_floor, support_g):
    """Conditional IPCW concordance; marginal G reduces to Uno's 1/G^2.

    g_at_events[j,i] = G_j(time_i-). Comparable pairs use G_i*G_j,
    not G_i^2 when censoring is conditional on different covariates.
    """
    numerator, denominator = 0., 0.
    for i in np.flatnonzero(event.astype(bool) & (time <= tau)):
        comparable = time > time[i]
        if not comparable.any():
            continue
        if g_at_events[i, i] < support_g or (g_at_events[comparable, i] < support_g).any():
            return np.nan
        weight = 1/(max(g_at_events[i, i], g_floor)*np.maximum(g_at_events[comparable, i], g_floor))
        diff = risk[i]-risk[comparable]
        numerator += np.sum(weight*((diff > 1e-8)+.5*(np.abs(diff) <= 1e-8)))
        denominator += weight.sum()
    return float(numerator/denominator) if denominator else np.nan


def calibration(curve, a, bins=5):
    records = []
    for index in np.array_split(np.argsort(curve, kind="stable"), min(bins, len(curve))):
        records.append({"n": len(index), "predicted": float(curve[index].mean()),
                        "observed_ipcw": float(a[index].mean())})
    return records


def uno_concordance(risk, time, event, marginal_g_at_event, tau, g_floor, support_g):
    """Uno concordance with marginal reverse-KM IPCW and event/censor ties.

    Each OOF case uses its own training-fold estimate of G. Same-time censored
    controls are comparable; two deaths at the same time are not ordered.
    """
    numerator, denominator = 0., 0.
    event = np.asarray(event, bool)
    for i in np.flatnonzero(event & (time <= tau)):
        comparable = (time > time[i]) | ((time == time[i]) & ~event)
        if not comparable.any():
            continue
        if marginal_g_at_event[i] < support_g:
            return np.nan
        weight = 1/max(marginal_g_at_event[i], g_floor)**2
        diff = risk[i]-risk[comparable]
        numerator += weight*np.sum((diff > 1e-8)+.5*(np.abs(diff) <= 1e-8))
        denominator += weight*comparable.sum()
    return float(numerator/denominator) if denominator else np.nan
