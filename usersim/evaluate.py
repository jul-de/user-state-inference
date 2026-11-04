"""Features, state naming, baselines and recovery metrics."""
from itertools import permutations

import numpy as np

from .simulate import BACK, CONF, DELIB, FWD, INTR, OFF, PROG, REP, PAGES


class Features:
    """Page-normalised log dwell: log(dwell) minus the population mean for that page.

    The page means are estimated from (unlabelled) training traces, so no
    ground truth is used.
    """
    def fit(self, traces):
        logs = [[] for _ in PAGES]
        for tr in traces:
            for p, d in zip(tr.pages, tr.dwell):
                logs[p].append(np.log(d))
        self.page_mean = np.array([np.mean(l) if l else 0.0 for l in logs])
        return self

    def transform(self, traces):
        actions = [np.array(tr.actions) for tr in traces]
        x = [np.log(np.array(tr.dwell)) - self.page_mean[np.array(tr.pages)] for tr in traces]
        return actions, x


def name_states(params):
    """Name fitted states from their emission profiles alone (no labels).

    Longest dwell -> interrupted, shortest -> progressing; of the remaining two,
    the one less likely to move forward -> confused, the other -> deliberating.
    Returns perm with perm[fitted_state] = named state index.
    """
    order = np.argsort(params.mu)
    prog, intr = order[0], order[-1]
    mid = [k for k in order[1:-1]]
    conf = max(mid, key=lambda k: 1 - params.B[k, FWD])
    delib = [k for k in mid if k != conf][0]
    perm = np.empty(4, dtype=int)
    perm[[prog, delib, conf, intr]] = [PROG, DELIB, CONF, INTR]
    return perm


def best_permutation(post, truth):
    """Evaluation-only: the state labelling that best matches ground truth."""
    P = np.concatenate(post)
    y = np.concatenate(truth)
    pred = P.argmax(1)
    best, best_acc = None, -1
    for perm in permutations(range(4)):
        acc = (np.array(perm)[pred] == y).mean()
        if acc > best_acc:
            best, best_acc = np.array(perm), acc
    return best


def relabel(post, perm):
    """Reorder posterior columns so column j is named state j."""
    out = []
    for g in post:
        h = np.zeros_like(g)
        for k in range(4):
            h[:, perm[k]] = g[:, k]
        out.append(h)
    return out


def rule_baseline(traces, short=6.0, long=30.0):
    """A typical analytics heuristic: thresholds on dwell, plus the action type."""
    out = []
    for tr in traces:
        g = np.zeros((len(tr.dwell), 4))
        for t, (d, a) in enumerate(zip(tr.dwell, tr.actions)):
            if d >= long:
                k = INTR
            elif a in (REP, OFF) or (a == BACK and d < short):
                k = CONF
            elif d >= short:
                k = DELIB if a == FWD else CONF
            else:
                k = PROG
            g[t, k] = 1.0
        out.append(g)
    return out


def recovery(post, truth):
    """Step-level recovery metrics against the hidden states."""
    P = np.concatenate(post)
    y = np.concatenate(truth)
    pred = P.argmax(1)
    cm = np.zeros((4, 4))
    for i, j in zip(y, pred):
        cm[i, j] += 1
    recall = cm.diagonal() / cm.sum(1).clip(1)
    precision = cm.diagonal() / cm.sum(0).clip(1)
    return {
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(recall.mean()),
        'recall': recall.tolist(),
        'precision': precision.tolist(),
        'confusion_matrix': (cm / cm.sum(1, keepdims=True).clip(1)).tolist(),
        'n_steps': int(len(y)),
    }


def calibration(post, truth, state=CONF, n_bins=10):
    """Reliability of P(state): within each probability bin, how often the state is true."""
    p = np.concatenate([g[:, state] for g in post])
    y = (np.concatenate(truth) == state).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 20:
            bins.append({'p_mean': float(p[m].mean()), 'observed': float(y[m].mean()), 'n': int(m.sum())})
    ece = sum(b['n'] * abs(b['p_mean'] - b['observed']) for b in bins) / max(1, sum(b['n'] for b in bins))
    return {'bins': bins, 'ece': float(ece)}
