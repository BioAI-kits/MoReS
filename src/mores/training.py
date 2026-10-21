"""Bootstrap pathway training and out-of-fold probability stacking."""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from .augmentation import perturb
from .censoring import CensoringModel
from .data import MODALITIES
from .networks import Expert, Joint
from .preprocessing import Preprocessor, splits
from .survival import PathPrediction, nll, quadrature

EXPERTS = (*MODALITIES, "clinical")


def tensors(data, preprocessor, device):
    x, coverage, available = preprocessor.transform(data)
    return ({m: torch.as_tensor(v, device=device) for m, v in x.items()},
            coverage, torch.as_tensor(available, device=device))


def eligibility(name, available):
    return available[:, :4].sum(1) >= 3 if name == "joint" else available[:, EXPERTS.index(name)].bool()


def forward(model, name, x, available, ix):
    return (model({m: a[ix] for m, a in x.items()}, available[ix]) if name == "joint"
            else model(x[name][ix]))


def train_members(data, preprocessor, cfg, seed, validation=None, fixed_epochs=None):
    torch.set_num_threads(cfg.threads)
    x, _, available = tensors(data, preprocessor, cfg.device)
    reduced = preprocessor.project(data)
    y = torch.tensor(np.minimum(data.time, cfg.tau), dtype=torch.float32, device=cfg.device)
    event = torch.tensor(data.event * (data.time <= cfg.tau), dtype=torch.float32, device=cfg.device)
    if validation is not None:
        vx, _, va = tensors(validation, preprocessor, cfg.device)
        vy = torch.tensor(np.minimum(validation.time, cfg.tau), dtype=torch.float32, device=cfg.device)
        ve = torch.tensor(validation.event * (validation.time <= cfg.tau), dtype=torch.float32, device=cfg.device)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(data), len(data)) for _ in range(cfg.ensemble_members)]
    models, epochs = {}, {}
    for ei, name in enumerate((*EXPERTS, "joint")):
        eligible = eligibility(name, available).cpu().numpy()
        models[name], epochs[name] = [], []
        if not eligible.any():
            models[name] = [None] * cfg.ensemble_members
            epochs[name] = [0] * cfg.ensemble_members
            continue
        for b, draw in enumerate(draws):
            torch.manual_seed(seed + 100*ei + b)
            model = (Joint({m: v.shape[1] for m, v in x.items()}, cfg) if name == "joint"
                     else Expert(x[name].shape[1], cfg)).to(cfg.device)
            selected = draw[eligible[draw]]
            if not len(selected):
                raise ValueError(f"Bootstrap member has no {name} inputs; more training patients are required")
            if validation is not None:
                vi = torch.where(eligibility(name, va))[0]
                if not len(vi):
                    raise ValueError(f"Early-stopping partition contains no {name} inputs")
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            limit = cfg.max_epochs if fixed_epochs is None else max(1, int(fixed_epochs[name][b]))
            best, best_epoch, stale, state = float("inf"), limit, 0, None
            aug_rng = np.random.default_rng(seed + 1000*ei + b)
            for epoch in range(1, limit+1):
                model.train()
                if cfg.augmentation_beta > 0:
                    augmented = perturb(reduced, cfg.augmentation_severity, aug_rng, rna_scale=cfg.rna_scale)
                    ax, _, aa = tensors(augmented, preprocessor, cfg.device)
                order = aug_rng.permutation(selected)
                for start in range(0, len(order), cfg.batch_size):
                    ix = torch.as_tensor(order[start:start+cfg.batch_size], device=cfg.device)
                    loss = nll(forward(model, name, x, available, ix), y[ix], event[ix], cfg.cuts).mean()
                    if cfg.augmentation_beta > 0:
                        aix = ix[eligibility(name, aa)[ix]]
                        if len(aix):
                            aug_loss = nll(forward(model, name, ax, aa, aix), y[aix], event[aix], cfg.cuts).sum()/len(ix)
                            loss = loss + cfg.augmentation_beta*aug_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {name} likelihood")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                if validation is not None:
                    model.eval()
                    with torch.no_grad():
                        score = nll(forward(model, name, vx, va, vi), vy[vi], ve[vi], cfg.cuts).mean().item()
                    if score < best - 1e-7:
                        best, best_epoch, stale = score, epoch, 0
                        state = copy.deepcopy(model.state_dict())
                    else:
                        stale += 1
                    if stale >= cfg.patience:
                        break
            if state is not None:
                model.load_state_dict(state)
            models[name].append(model.cpu().eval())
            epochs[name].append(best_epoch)
    return models, epochs


@dataclass
class NetworkBundle:
    preprocessor: Preprocessor
    models: dict
    cfg: object
    epochs: dict
    fit_ids: list

    def predict(self, data, theta=None):
        # Inference remains on CPU so saved bundles are portable.
        x, _, available = tensors(data, self.preprocessor, "cpu")
        actual = available.numpy().copy()
        n, b, k = len(data), self.cfg.ensemble_members, len(self.cfg.cuts)-1
        rates = np.zeros((n, 5, b, k))
        joint = np.zeros((n, b, k))
        with torch.no_grad():
            for j, name in enumerate(EXPERTS):
                if any(model is None for model in self.models[name]):
                    actual[:, j] = False
                    continue
                ix = torch.where(available[:, j])[0]
                for member, model in enumerate(self.models[name]):
                    if len(ix):
                        rates[ix.numpy(), j, member] = model(x[name][ix]).numpy()
            eligible = actual[:, :4].sum(1) >= 3
            if any(model is None for model in self.models["joint"]):
                eligible[:] = False
            ix = torch.as_tensor(np.flatnonzero(eligible))
            for member, model in enumerate(self.models["joint"]):
                if len(ix):
                    joint[ix.numpy(), member] = model({m: a[ix] for m, a in x.items()}, available[ix]).numpy()
        if not actual.any(axis=1).all():
            raise ValueError("No trained expert is available for some prediction patients")
        logits = np.zeros(5) if theta is None else np.asarray(theta)
        weights = softmax(np.where(actual, logits, -np.inf), axis=1)
        return PathPrediction(rates, joint, weights, eligible, self.cfg.cuts)


def _fit_networks_fixed(data, cfg, seed, fixed_epochs=None):
    if fixed_epochs is None:
        train, val = train_test_split(np.arange(len(data)), test_size=cfg.early_stop_fraction,
                                      stratify=data.event, random_state=seed)
        sub, held = data.subset(train), data.subset(val)
        pre = Preprocessor().fit(sub, cfg, seed)
        _, fixed_epochs = train_members(sub, pre, cfg, seed, validation=held)
    pre = Preprocessor().fit(data, cfg, seed)
    models, epochs = train_members(data, pre, cfg, seed, fixed_epochs=fixed_epochs)
    return NetworkBundle(pre, models, cfg, epochs, data.ids.tolist())


@dataclass
class Upstream:
    networks: NetworkBundle
    theta: np.ndarray
    censoring: CensoringModel
    stack_provenance: list

    @property
    def preprocessor(self):
        return self.networks.preprocessor

    def predict(self, data):
        return self.networks.predict(data, self.theta)


def fit_upstream(data, cfg, seed, fixed_epochs=None):
    grid, integration = quadrature(cfg.tau, cfg.cuts, data.time, cfg.integration_order)
    s = np.zeros((len(data), 5, len(grid)))
    a, w = np.zeros((len(data), len(grid))), np.zeros((len(data), len(grid)))
    usable = np.zeros((len(data), 5), bool)
    provenance = []
    for f, (tr, va) in enumerate(splits(data.event, cfg.stack_folds, seed)):
        train, valid = data.subset(tr), data.subset(va)
        nets = fit_networks(train, cfg, seed + 17*(f+1), fixed_epochs)
        prediction = nets.predict(valid)
        censor = CensoringModel().fit(train, cfg)
        s[va] = prediction.experts(grid)
        usable[va] = prediction.expert_weights > 0
        a[va], w[va], _, _ = censor.weights(valid, grid)
        provenance.append({"fit_ids": train.ids.tolist(), "prediction_ids": valid.ids.tolist()})
    def objective(theta):
        weights = softmax(np.where(usable, theta, -np.inf), axis=1)
        curve = np.einsum("nmt,nm->nt", s, weights)
        # A - 2*A*S + W*S^2 equals IPCW squared error exactly.
        error = ((a - 2*a*curve + w*curve**2) @ integration).mean()
        grad_curve = ((-2*a + 2*w*curve) * integration)/len(data)
        grad = (weights * np.einsum("nt,nmt->nm", grad_curve, s-curve[:, None, :])).sum(0)
        return error + cfg.stack_l2*np.sum(theta**2), grad + 2*cfg.stack_l2*theta
    opt = minimize(objective, np.zeros(5), jac=True, method="L-BFGS-B")
    if not opt.success:
        raise RuntimeError(f"Reference stacking failed: {opt.message}")
    nets = fit_networks(data, cfg, seed+991, fixed_epochs)
    return Upstream(nets, opt.x, CensoringModel().fit(data, cfg), provenance)

# Keep network tuning local to every upstream fit. In particular, a held-out
# gain patient cannot affect network hyperparameters through an outer fit.


def fit_networks(data, cfg, seed, fixed_epochs=None):
    from dataclasses import replace
    from .metrics import ibs
    if cfg.search_trials == 1 or fixed_epochs is not None:
        return _fit_networks_fixed(data, cfg, seed, fixed_epochs)
    rng = np.random.default_rng(seed)
    from itertools import product
    candidates = list(product([.0003, .0005, .001], [16, 32], [16, 32, 64], [.1, .2, .3], [1e-5, 1e-4, 1e-3]))
    chosen = rng.choice(len(candidates), min(cfg.search_trials, len(candidates)), replace=False)
    best_score, best_cfg, best_epochs = float("inf"), None, None
    for index in chosen:
        lr, batch, dim, drop, decay = candidates[index]
        candidate = replace(cfg, learning_rate=lr, batch_size=batch, hidden_dim=dim, dropout=drop, weight_decay=decay)
        scores, epoch_sets = [], []
        for f, (tr, va) in enumerate(splits(data.event, cfg.inner_folds, seed)):
            train, valid = data.subset(tr), data.subset(va)
            nets = _fit_networks_fixed(train, candidate, seed+31*(f+1))
            censor = CensoringModel().fit(train, candidate)
            grid, weights = quadrature(cfg.tau, cfg.cuts, np.r_[censor.knots, valid.time], cfg.integration_order)
            a, w, _, _ = censor.weights(valid, grid)
            prediction = nets.predict(valid)
            s0, s1 = prediction.curves(grid)
            # Shared network configuration selected by mean pathway IBS.
            scores.append((ibs(s0, a, w, weights)+ibs(s1, a, w, weights))/2)
            epoch_sets.append(nets.epochs)
        if np.mean(scores) < best_score:
            best_score, best_cfg = np.mean(scores), candidate
            best_epochs = {name: [int(np.median([e[name][b] for e in epoch_sets])) for b in range(cfg.ensemble_members)]
                           for name in (*EXPERTS, "joint")}
    return _fit_networks_fixed(data, best_cfg, seed+97, best_epochs)
