"""Hidden Markov model for interaction traces, written from scratch in NumPy.

Each step emits a discrete action (categorical) and a continuous feature x
(page-normalised log dwell time, Gaussian). The model is fitted by
expectation-maximisation (Baum-Welch) on unlabelled traces, with no access
to the hidden states.

Variants used in the signal ablation:
    use_action=False   ignore actions (dwell only)
    use_dwell=False    ignore dwell (actions only)
    sequential=False   no transition structure: each step classified on its own
                       (a mixture model), so neighbouring steps give no context
"""
from dataclasses import dataclass

import numpy as np

LOG2PI = np.log(2 * np.pi)


@dataclass
class HMMParams:
    pi: np.ndarray      # (K,) initial state probabilities
    A: np.ndarray       # (K, K) transitions
    B: np.ndarray       # (K, n_actions) action probabilities
    mu: np.ndarray      # (K,) mean of x
    sd: np.ndarray      # (K,) SD of x


def pad(seqs, fill=0):
    """List of 1-D arrays -> (N, T) padded array and boolean mask."""
    T = max(len(s) for s in seqs)
    out = np.full((len(seqs), T), fill, dtype=float)
    mask = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
        mask[i, :len(s)] = True
    return out, mask


class InteractionHMM:
    def __init__(self, n_states=4, n_actions=4, use_action=True, use_dwell=True, sequential=True,
                 n_iter=200, tol=1e-4, n_restarts=5, seed=0, sd_floor=0.05):
        self.K, self.M = n_states, n_actions
        self.use_action, self.use_dwell, self.sequential = use_action, use_dwell, sequential
        self.n_iter, self.tol, self.n_restarts, self.seed, self.sd_floor = n_iter, tol, n_restarts, seed, sd_floor
        self.params = None
        self.loglik_history = []

    # ---------- emissions ----------
    def _emissions(self, p, a, x, mask):
        N, T = a.shape
        logE = np.zeros((N, T, self.K))
        if self.use_action:
            logE += np.log(p.B[:, a.astype(int)]).transpose(1, 2, 0)
        if self.use_dwell:
            z = (x[..., None] - p.mu) / p.sd
            logE += -0.5 * (z ** 2 + LOG2PI) - np.log(p.sd)
        logE[~mask] = 0.0
        # Rescale per step for numerical stability; the shift is added back to the log-likelihood.
        shift = logE.max(axis=2, keepdims=True)
        return np.exp(logE - shift), shift[..., 0]

    # ---------- forward-backward (batched over sequences, with scaling) ----------
    def _forward_backward(self, p, a, x, mask):
        N, T = a.shape
        E, shift = self._emissions(p, a, x, mask)
        alpha = np.zeros((N, T, self.K))
        c = np.ones((N, T))
        a0 = p.pi * E[:, 0]
        c[:, 0] = a0.sum(1)
        alpha[:, 0] = a0 / c[:, 0, None]
        for t in range(1, T):
            at = (alpha[:, t - 1] @ p.A) * E[:, t]
            m = mask[:, t]
            at = np.where(m[:, None], at, alpha[:, t - 1])
            c[:, t] = np.where(m, at.sum(1), 1.0)
            alpha[:, t] = at / c[:, t, None]
        beta = np.ones((N, T, self.K))
        for t in range(T - 2, -1, -1):
            bt = ((E[:, t + 1] * beta[:, t + 1]) @ p.A.T) / c[:, t + 1, None]
            beta[:, t] = np.where(mask[:, t + 1, None], bt, 1.0)
        gamma = alpha * beta
        gamma /= gamma.sum(2, keepdims=True)
        gamma[~mask] = 0.0
        loglik = float((np.log(c) * mask).sum() + (shift * mask).sum())
        return alpha, beta, c, E, gamma, loglik

    # ---------- EM ----------
    def _init_params(self, rng, a, x, mask):
        K = self.K
        xv = x[mask]
        qs = np.sort(rng.uniform(0.05, 0.95, K))
        mu = np.quantile(xv, qs) + rng.normal(0, 0.05, K)
        sd = np.full(K, xv.std() / 2)
        B = rng.dirichlet(np.ones(self.M) * 5, K)
        A = np.full((K, K), 0.2 / (K - 1)) + np.eye(K) * (0.8 - 0.2 / (K - 1))
        pi = np.ones(K) / K
        if not self.use_dwell:
            mu, sd = np.zeros(K), np.ones(K)
        if not self.use_action:
            B = np.ones((K, self.M)) / self.M
        if not self.sequential:
            A = np.tile(pi, (K, 1))
        return HMMParams(pi, A, B, mu, sd)

    def _m_step(self, p, a, x, mask, alpha, beta, c, E, gamma):
        K = self.K
        w = gamma[mask]                       # (n_steps, K)
        occ = w.sum(0) + 1e-12
        pi = gamma[:, 0].sum(0) + 1e-6
        pi /= pi.sum()
        if self.sequential:
            xi = np.zeros((K, K))
            for t in range(a.shape[1] - 1):
                m = mask[:, t + 1]
                if not m.any():
                    break
                num = alpha[m, t, :, None] * p.A[None] * (E[m, t + 1] * beta[m, t + 1])[:, None, :]
                xi += (num / c[m, t + 1, None, None]).sum(0)
            A = xi + 1e-6
            A /= A.sum(1, keepdims=True)
        else:
            pi = occ / occ.sum()
            A = np.tile(pi, (K, 1))
        B = p.B
        if self.use_action:
            av = a[mask].astype(int)
            B = np.stack([np.bincount(av, weights=w[:, k], minlength=self.M) for k in range(K)]) + 1e-3
            B /= B.sum(1, keepdims=True)
        mu, sd = p.mu, p.sd
        if self.use_dwell:
            xv = x[mask]
            mu = (w * xv[:, None]).sum(0) / occ
            sd = np.sqrt((w * (xv[:, None] - mu) ** 2).sum(0) / occ)
            sd = np.maximum(sd, self.sd_floor)
        return HMMParams(pi, A, B, mu, sd)

    def fit(self, actions, x, init=None):
        """EM on unlabelled traces. With `init` (e.g. from params_from_labels), start there
        instead of from random restarts."""
        a, mask = pad(actions)
        xp, _ = pad(x)
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(1 if init is not None else self.n_restarts):
            p = init if init is not None else self._init_params(rng, a, xp, mask)
            # hist[i] is the log-likelihood of the parameters after i M-steps, so hist[-1]
            # always belongs to the parameters that are kept.
            alpha, beta, c, E, gamma, ll = self._forward_backward(p, a, xp, mask)
            hist = [ll]
            for _ in range(self.n_iter):
                p = self._m_step(p, a, xp, mask, alpha, beta, c, E, gamma)
                alpha, beta, c, E, gamma, ll = self._forward_backward(p, a, xp, mask)
                converged = ll - hist[-1] < self.tol * abs(ll)
                hist.append(ll)
                if converged:
                    break
            if best is None or hist[-1] > best[1]:
                best = (p, hist[-1], hist)
        self.params, _, self.loglik_history = best
        return self

    def posteriors(self, actions, x):
        """Per-step posterior P(state | whole trace) for each trace."""
        a, mask = pad(actions)
        xp, _ = pad(x)
        *_, gamma, _ = self._forward_backward(self.params, a, xp, mask)
        return [gamma[i, :len(s)] for i, s in enumerate(actions)]

    def loglik(self, actions, x):
        a, mask = pad(actions)
        xp, _ = pad(x)
        return self._forward_backward(self.params, a, xp, mask)[-1]


def params_from_labels(actions, x, states, n_states=4, n_actions=4, sd_floor=0.05):
    """Maximum-likelihood HMM parameters from traces whose hidden states are known
    (e.g. a small study where sessions were coded by think-aloud or experience sampling)."""
    K = n_states
    pi = np.bincount([s[0] for s in states], minlength=K) + 1.0
    A = np.ones((K, K))
    B = np.ones((K, n_actions))
    xs = [[] for _ in range(K)]
    for a, xi, s in zip(actions, x, states):
        for t in range(len(s)):
            B[s[t], a[t]] += 1
            xs[s[t]].append(xi[t])
            if t:
                A[s[t - 1], s[t]] += 1
    mu = np.array([np.mean(v) if v else 0.0 for v in xs])
    sd = np.array([max(np.std(v), sd_floor) if len(v) > 1 else 1.0 for v in xs])
    return HMMParams(pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True), mu, sd)
