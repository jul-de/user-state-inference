import itertools

import numpy as np

from usersim.hmm import HMMParams, InteractionHMM, params_from_labels


def sample_hmm(p, n_seq, T, rng):
    acts, xs, sts = [], [], []
    for _ in range(n_seq):
        s = rng.choice(len(p.pi), p=p.pi)
        a, x, st = [], [], []
        for t in range(T):
            if t:
                s = rng.choice(len(p.pi), p=p.A[s])
            st.append(s)
            a.append(rng.choice(p.B.shape[1], p=p.B[s]))
            x.append(rng.normal(p.mu[s], p.sd[s]))
        acts.append(np.array(a)); xs.append(np.array(x)); sts.append(np.array(st))
    return acts, xs, sts


TRUE = HMMParams(pi=np.array([0.6, 0.4]), A=np.array([[0.9, 0.1], [0.2, 0.8]]),
                 B=np.array([[0.8, 0.1, 0.05, 0.05], [0.2, 0.4, 0.2, 0.2]]),
                 mu=np.array([-1.0, 1.0]), sd=np.array([0.5, 0.7]))


def brute_force_loglik(p, a, x):
    """Sum over every hidden path: the definition the forward pass must match."""
    total = 0.0
    for path in itertools.product(range(len(p.pi)), repeat=len(a)):
        pr = p.pi[path[0]]
        for t, s in enumerate(path):
            if t:
                pr *= p.A[path[t - 1], s]
            pr *= p.B[s, a[t]] * np.exp(-0.5 * ((x[t] - p.mu[s]) / p.sd[s]) ** 2) / (p.sd[s] * np.sqrt(2 * np.pi))
        total += pr
    return np.log(total)


def test_forward_matches_brute_force_and_handles_padding():
    rng = np.random.default_rng(0)
    acts, xs, _ = sample_hmm(TRUE, 3, 5, rng)
    acts[1], xs[1] = acts[1][:2], xs[1][:2]     # unequal lengths exercise the padding mask
    m = InteractionHMM(n_states=2)
    m.params = TRUE
    expected = sum(brute_force_loglik(TRUE, a, x) for a, x in zip(acts, xs))
    assert np.isclose(m.loglik(acts, xs), expected, rtol=1e-10)
    for g, a in zip(m.posteriors(acts, xs), acts):
        assert g.shape == (len(a), 2)
        assert np.allclose(g.sum(1), 1.0)


def test_em_increases_likelihood_and_recovers_parameters():
    rng = np.random.default_rng(1)
    acts, xs, _ = sample_hmm(TRUE, 400, 15, rng)
    m = InteractionHMM(n_states=2, n_restarts=3, seed=0).fit(acts, xs)
    h = np.array(m.loglik_history)
    assert np.all(np.diff(h) > -1e-6 * abs(h[:-1]))
    order = np.argsort(m.params.mu)
    assert np.allclose(m.params.mu[order], TRUE.mu, atol=0.1)
    assert np.allclose(m.params.A[np.ix_(order, order)], TRUE.A, atol=0.05)
    assert np.allclose(m.params.B[order], TRUE.B, atol=0.05)


def test_params_from_labels_recovers_generating_parameters():
    rng = np.random.default_rng(2)
    acts, xs, sts = sample_hmm(TRUE, 500, 15, rng)
    p = params_from_labels(acts, xs, sts, n_states=2)
    assert np.allclose(p.mu, TRUE.mu, atol=0.05)
    assert np.allclose(p.A, TRUE.A, atol=0.03)
    assert np.allclose(p.B, TRUE.B, atol=0.03)
