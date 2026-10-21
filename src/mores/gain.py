"""Patient-grouped ridge gain estimation and conditional bootstrap stability."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from .augmentation import perturb
from .censoring import gain_pseudo_outcome
from .preprocessing import splits
from .training import fit_upstream


def summary(upstream, data, prediction):
    _, coverage, available = upstream.preprocessor.transform(data)
    s0, s1 = prediction.curves(upstream.networks.cfg.cuts[1:])
    q = prediction.q()
    clinical = upstream.preprocessor.clinical.transform(data.clinical)
    clinical[~data.available[:, 4]] = 0
    z = np.column_stack([s0, s1, q, available.astype(float), coverage, clinical])
    return z, q


@dataclass
class GainRecords:
    z: np.ndarray
    b: np.ndarray
    q: np.ndarray
    ids: np.ndarray
    weights: np.ndarray
    provenance: list


def cross_fitted_records(data, cfg, seed, fixed_epochs=None):
    zs, bs, qs, ids, provenance = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for f, (tr, va) in enumerate(splits(data.event, cfg.gain_folds, seed)):
        train, valid = data.subset(tr), data.subset(va)
        upstream = fit_upstream(train, cfg, seed+1009*(f+1), fixed_epochs)
        reduced = upstream.preprocessor.project(valid)
        record = {"fit_ids": train.ids.tolist(), "prediction_ids": valid.ids.tolist(),
                  "stack_folds": upstream.stack_provenance, "clipped_weight_fractions": []}
        for aug in range(cfg.gain_augmentations+1):
            inputs = valid if aug == 0 else perturb(reduced, cfg.augmentation_severity, rng, rna_scale=cfg.rna_scale)
            prediction = upstream.predict(inputs)
            z, q = summary(upstream, inputs, prediction)
            # Always use unperturbed covariates/masks in the censoring model.
            b, clip_fraction = gain_pseudo_outcome(prediction, valid, upstream.censoring)
            keep = prediction.eligible
            zs.append(z[keep]); bs.append(b[keep]); qs.append(q[keep]); ids.append(valid.ids[keep])
            record["clipped_weight_fractions"].append(clip_fraction)
        provenance.append(record)
    z, b, q, patient = np.concatenate(zs), np.concatenate(bs), np.concatenate(qs), np.concatenate(ids)
    if len(patient) < cfg.inner_folds:
        raise ValueError("Insufficient joint-eligible OOF patients to fit conditional gains")
    _, inverse, counts = np.unique(patient, return_inverse=True, return_counts=True)
    return GainRecords(z, b, q, patient, 1./counts[inverse], provenance)


@dataclass
class RidgeModel:
    scaler: StandardScaler
    ridge: Ridge

    @classmethod
    def fit(cls, z, b, weights, alpha):
        scaler = StandardScaler().fit(z, sample_weight=weights)
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(z), b, sample_weight=weights)
        return cls(scaler, ridge)

    def predict(self, z, q):
        return np.clip(self.ridge.predict(self.scaler.transform(z)), -np.sqrt(q), np.sqrt(q))


class GainEnsemble:
    def fit(self, records, cfg, seed):
        groups = np.unique(records.ids)
        if len(groups) < cfg.inner_folds:
            raise ValueError("Too few unique patients for ridge tuning")
        cv = list(GroupKFold(cfg.inner_folds).split(records.z, records.b, records.ids))
        scores = []
        self.tuning_provenance = []
        for tr, va in cv:
            self.tuning_provenance.append({"fit_ids": np.unique(records.ids[tr]).tolist(),
                                           "prediction_ids": np.unique(records.ids[va]).tolist()})
        for alpha in cfg.ridge_alphas:
            losses = []
            for tr, va in cv:
                model = RidgeModel.fit(records.z[tr], records.b[tr], records.weights[tr], alpha)
                # Tune the regression objective before feasible projection.
                prediction = model.ridge.predict(model.scaler.transform(records.z[va]))
                losses.append(np.average((records.b[va]-prediction)**2, weights=records.weights[va]))
            scores.append(np.mean(losses))
        self.alpha = cfg.ridge_alphas[int(np.argmin(scores))]
        self.point = RidgeModel.fit(records.z, records.b, records.weights, self.alpha)
        self.models, self.bootstrap_patient_counts = [], []
        rng = np.random.default_rng(seed)
        for _ in range(cfg.bootstrap_replicates):
            draw = rng.choice(groups, size=len(groups), replace=True)
            chosen, counts = np.unique(draw, return_counts=True)
            multiplicity = dict(zip(chosen, counts))
            w = records.weights * np.array([multiplicity.get(p, 0) for p in records.ids])
            keep = w > 0
            self.models.append(RidgeModel.fit(records.z[keep], records.b[keep], w[keep], self.alpha))
            self.bootstrap_patient_counts.append({str(k): int(v) for k, v in multiplicity.items()})
        self.fit_ids = groups.tolist()
        return self

    def predict(self, z, q):
        values = np.stack([model.predict(z, q) for model in self.models])
        return values.mean(axis=0), values.std(axis=0, ddof=0)
