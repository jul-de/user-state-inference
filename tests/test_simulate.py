import numpy as np

from usersim.evaluate import Features, name_states, rule_baseline
from usersim.hmm import params_from_labels
from usersim.simulate import ARCHETYPES, PAGES, simulate_population, transition_probs


def test_reproducible_with_seed():
    a = simulate_population(50, 'A', seed=3)
    b = simulate_population(50, 'A', seed=3)
    assert [t.dwell for t in a] == [t.dwell for t in b]


def test_traces_are_well_formed():
    for d in 'AB':
        for t in simulate_population(300, d, seed=4):
            n = len(t.states)
            assert n == len(t.actions) == len(t.dwell) == len(t.pages) > 0
            assert all(0 <= p < len(PAGES) for p in t.pages)
            assert all(x > 0 for x in t.dwell)
            assert t.outcome in ('completed', 'abandoned', 'timeout')
            if t.outcome == 'completed':
                assert t.pages[-1] == len(PAGES) - 1 and t.actions[-1] == 0


def test_transition_probabilities_are_valid():
    for prev in range(4):
        for eff in (0.05, 0.5, 0.95):
            p = transition_probs(prev, eff, dict(care=0.5, distract=0.1))
            assert np.isclose(p.sum(), 1) and (p >= 0).all()


def test_redesign_increases_confusion_on_options_page():
    def conf_share(design):
        pop = simulate_population(800, design, seed=5)
        st = np.concatenate([np.array(t.states)[np.array(t.pages) == 4] for t in pop])
        return (st == 2).mean()
    assert conf_share('B') > conf_share('A') + 0.1


def test_name_states_uses_emission_profiles():
    pop = simulate_population(400, 'A', seed=6)
    f = Features().fit(pop)
    a, x = f.transform(pop)
    p = params_from_labels(a, x, [np.array(t.states) for t in pop])
    assert list(name_states(p)) == [0, 1, 2, 3]


def test_rule_baseline_is_one_hot():
    for g in rule_baseline(simulate_population(20, 'A', seed=7)):
        assert np.all(g.sum(1) == 1)
