"""Piecewise exponential calculus and mixture curves, in months."""
from dataclasses import dataclass
import numpy as np
import torch


def exposure(times, cuts):
    return np.clip(np.asarray(times)[..., None] - np.asarray(cuts[:-1]), 0, np.diff(cuts))


def nll(rates, times, event, cuts):
    bounds = rates.new_tensor(cuts)
    times = times.clamp(max=float(cuts[-1]))
    lengths = (times[:, None] - bounds[:-1]).clamp(min=0)
    lengths = torch.minimum(lengths, bounds[1:] - bounds[:-1])
    # (left, right] event intervals: an event exactly at 6 belongs to (0,6].
    interval = torch.bucketize(times.contiguous(), bounds[1:], right=False).clamp(max=rates.shape[1]-1)
    return (rates * lengths).sum(1) - event * torch.log(rates.gather(1, interval[:, None]).squeeze(1).clamp(min=1e-12))


def quadrature(tau, cuts=(), discontinuities=(), order=4):
    breaks = np.unique(np.r_[0., tau, np.arange(0., tau, 1.), cuts, discontinuities])
    breaks = breaks[(breaks >= 0) & (breaks <= tau)]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(breaks) / 2
    grid = ((breaks[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
    weights = (half[:, None] * weights).ravel() / tau
    return grid, weights


@dataclass
class PathPrediction:
    expert_rates: np.ndarray  # n, 5, members, intervals
    joint_rates: np.ndarray  # n, members, intervals; zeros for ineligible patients
    expert_weights: np.ndarray  # n, 5
    eligible: np.ndarray
    cuts: list[float]

    def experts(self, times):
        e = exposure(times, self.cuts)
        return np.exp(-np.einsum("nmbk,tk->nmbt", self.expert_rates, e)).mean(axis=2)

    def curves(self, times):
        s0 = np.einsum("nmt,nm->nt", self.experts(times), self.expert_weights)
        s1 = np.exp(-np.einsum("nbk,tk->nbt", self.joint_rates, exposure(times, self.cuts))).mean(axis=1)
        s1[~self.eligible] = s0[~self.eligible]
        return s0, s1

    def q(self):
        grid, w = quadrature(self.cuts[-1], self.cuts, order=12)
        s0, s1 = self.curves(grid)
        return ((s1-s0)**2) @ w


def robust_weights(q, mean, scale, kappa, eta, eligible=None):
    q, mean, scale = np.broadcast_arrays(q, mean, scale)
    if eta <= 0 or kappa < 0 or (q < 0).any() or (scale < 0).any():
        raise ValueError("q/u/kappa must be nonnegative and eta positive")
    ell = np.maximum(-np.sqrt(q), mean-kappa*scale)
    alpha = np.clip(ell/(q+eta), 0, 1)
    alpha = np.where(q > 0, alpha, 0.)
    if eligible is not None:
        alpha = np.where(eligible, alpha, 0.)
    return alpha, ell
