"""Explicit method settings; unspecified manuscript choices are documented."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    cuts: list[float] = field(default_factory=lambda: [0, 6, 12, 24, 36, 60])
    seed: int = 20261020
    device: str = "cpu"
    threads: int = 1
    outer_folds: int = 5
    repeats: int = 20
    inner_folds: int = 3
    gain_folds: int = 3
    stack_folds: int = 3
    selection_folds: int = 3
    ensemble_members: int = 3
    bootstrap_replicates: int = 100
    top_features: int = 50
    # None means no variance prefilter. If used, fit it on each training split.
    max_candidates: int | None = None
    cox_alphas: list[float] = field(default_factory=lambda: [.001, .01, .1, 1.])
    ridge_alphas: list[float] = field(default_factory=lambda: [.001, .01, .1, 1., 10.])
    kappa_grid: list[float] = field(default_factory=lambda: [0., .35, .7, 1., 1.4])
    eta_grid: list[float] = field(default_factory=lambda: [.0001, .00025, .001])
    search_trials: int = 12
    learning_rate: float = .0005
    batch_size: int = 32
    hidden_dim: int = 32
    dropout: float = .2
    weight_decay: float = .0001
    max_epochs: int = 200
    patience: int = 20
    early_stop_fraction: float = .2
    stack_l2: float = .01
    censoring: str = "cox"
    censor_alpha: float = 1.
    g_floor: float = .025
    support_g: float = .001
    min_at_risk: int = 5
    integration_order: int = 4
    augmentation_beta: float = .25
    augmentation_severity: float = .15
    gain_augmentations: int = 1
    rna_scale: str = "log2p1"

    @property
    def tau(self):
        return float(self.cuts[-1])

    def validate(self):
        if len(self.cuts) < 2 or self.cuts[0] != 0 or any(b <= a for a, b in zip(self.cuts, self.cuts[1:])):
            raise ValueError("cuts must start at 0 and increase strictly")
        for key in ("outer_folds", "inner_folds", "gain_folds", "stack_folds", "selection_folds"):
            if getattr(self, key) < 2:
                raise ValueError(f"{key} must be >=2")
        for key in ("repeats", "ensemble_members", "bootstrap_replicates", "top_features", "search_trials", "max_epochs", "patience", "batch_size", "hidden_dim", "threads", "min_at_risk", "integration_order"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.hidden_dim % 2 or not 0 <= self.dropout < 1:
            raise ValueError("hidden_dim must be even; dropout must be in [0,1)")
        if not 0 < self.support_g <= self.g_floor <= 1:
            raise ValueError("require 0 < support_g <= g_floor <= 1")
        if not 0 < self.early_stop_fraction < .5:
            raise ValueError("early_stop_fraction must be in (0,.5)")
        if self.censoring not in {"cox", "km"} or self.rna_scale not in {"log2p1", "raw"}:
            raise ValueError("unknown censoring model or RNA scale")
        if any(x <= 0 for x in self.cox_alphas + self.ridge_alphas + self.eta_grid) or not all([self.cox_alphas, self.ridge_alphas, self.eta_grid, self.kappa_grid]):
            raise ValueError("positive nonempty penalty grids required")
        if min(self.kappa_grid) < 0 or self.augmentation_beta < 0 or self.gain_augmentations < 0 or not 0 <= self.augmentation_severity < 1:
            raise ValueError("invalid shrinkage/augmentation parameters")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
