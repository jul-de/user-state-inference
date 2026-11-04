"""Synthetic users moving through a small checkout flow.

Each simulated user has persona traits (tech fluency, carefulness, patience,
distractibility). At every step they are in one hidden state, and that state
(together with the page and their traits) produces what an analytics pipeline
would actually log: an action and the dwell time before it.

Hidden states
    PROGRESSING   knows what to do next and does it
    DELIBERATING  deliberately weighing options (slow, sometimes goes back to compare)
    CONFUSED      can't see how to move forward (slow, goes back, wanders, repeats clicks)
    INTERRUPTED   attention is elsewhere (very long dwell, then resumes)

Logged actions
    forward   the action that moves the task on
    back      returns to the previous page
    repeat    repeats the same click without effect
    offpath   follows a link that doesn't help with the task

The generator is deliberately *richer* than the inference model in hmm.py: its
transitions depend on the page and on each user's traits, and dwell times
depend on page complexity and fluency. The inference model is a plain,
stationary HMM fitted without labels, so recovery results reflect a
misspecified model, as they would with real users.
"""
from dataclasses import dataclass, field

import numpy as np

STATES = ['progressing', 'deliberating', 'confused', 'interrupted']
PROG, DELIB, CONF, INTR = range(4)
ACTIONS = ['forward', 'back', 'repeat', 'offpath']
FWD, BACK, REP, OFF = range(4)

PAGES = ['Home', 'Search', 'Results', 'Product', 'Options', 'Cart', 'Checkout']

# Interface designs. clarity = how obvious the forward action is (0-1);
# complexity scales reading time on the page.
DESIGNS = {
    'A': {'clarity': [0.95, 0.90, 0.80, 0.85, 0.85, 0.90, 0.85],
          'complexity': [1.0, 1.0, 1.6, 1.4, 1.2, 1.0, 1.3]},
    # Redesign: the Options page hides "Add to cart" inside a dropdown.
    'B': {'clarity': [0.95, 0.90, 0.80, 0.85, 0.45, 0.90, 0.85],
          'complexity': [1.0, 1.0, 1.6, 1.4, 1.3, 1.0, 1.3]},
}

# Persona archetypes: population share and mean traits. Individuals vary around these.
ARCHETYPES = {
    'expert':       dict(share=0.25, fluency=0.85, care=0.25, patience=0.50, distract=0.03),
    'careful':      dict(share=0.25, fluency=0.60, care=0.80, patience=0.80, distract=0.03),
    'novice':       dict(share=0.30, fluency=0.25, care=0.50, patience=0.60, distract=0.04),
    'distractible': dict(share=0.20, fluency=0.55, care=0.35, patience=0.35, distract=0.15),
}

# Action probabilities in each hidden state (rows: states, cols: ACTIONS).
ACTION_P = np.array([
    [0.90, 0.03, 0.02, 0.05],   # progressing
    [0.80, 0.15, 0.01, 0.04],   # deliberating: sometimes goes back to compare
    [0.20, 0.30, 0.25, 0.25],   # confused
    [0.70, 0.10, 0.08, 0.12],   # interrupted: resumes roughly as normal
])
# Dwell time before the action: lognormal median (seconds) and log-SD per state.
DWELL_MEDIAN = np.array([3.0, 12.0, 8.0, 55.0])
DWELL_SIGMA = np.array([0.35, 0.45, 0.55, 0.80])

MAX_STEPS = 60


@dataclass
class Trace:
    user_id: int
    archetype: str
    traits: dict
    design: str
    pages: list = field(default_factory=list)     # page index at each step
    actions: list = field(default_factory=list)
    dwell: list = field(default_factory=list)     # seconds
    states: list = field(default_factory=list)    # hidden, ground truth
    outcome: str = 'timeout'                      # completed | abandoned | timeout


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1 - p))


def draw_user(rng, archetype):
    a = ARCHETYPES[archetype]
    t = {k: float(np.clip(a[k] + rng.normal(0, 0.10), 0.02, 0.98)) for k in ('fluency', 'care', 'patience')}
    t['distract'] = float(np.clip(a['distract'] + rng.normal(0, 0.02), 0.0, 0.5))
    return t


def transition_probs(prev, eff, t):
    """P(next hidden state | previous state, effective page clarity, traits)."""
    p = np.zeros(4)
    p_intr = t['distract'] * (0.4 if prev == INTR else 1.0) + (0.20 if prev == INTR else 0.0)
    p_conf = (0.50 + 0.4 * (1 - eff)) if prev == CONF else 0.7 * (1 - eff)   # confusion tends to persist
    rest = 1 - p_conf
    p_delib = rest * t['care'] * (0.55 if prev == DELIB else 0.35)
    p[CONF], p[DELIB], p[PROG] = p_conf, p_delib, rest - p_delib
    p = (1 - p_intr) * p
    p[INTR] += p_intr
    return p / p.sum()


def simulate_user(rng, user_id, archetype, design):
    d = DESIGNS[design]
    t = draw_user(rng, archetype)
    tr = Trace(user_id, archetype, t, design)
    state, page = PROG, 0
    speed = 1.25 - 0.5 * t['fluency']   # fluent users read faster
    for _ in range(MAX_STEPS):
        eff = _sigmoid(_logit(d['clarity'][page]) + 2.5 * (t['fluency'] - 0.5))
        state = rng.choice(4, p=transition_probs(state, eff, t))
        action = rng.choice(4, p=ACTION_P[state])
        scale = 1.0 if state == INTR else d['complexity'][page] * speed
        dwell = float(DWELL_MEDIAN[state] * scale * np.exp(rng.normal(0, DWELL_SIGMA[state])))
        tr.pages.append(page); tr.actions.append(int(action)); tr.dwell.append(round(dwell, 2)); tr.states.append(int(state))

        if state == CONF and rng.random() < 0.02 + 0.12 * (1 - t['patience']):
            tr.outcome = 'abandoned'
            break
        if action == FWD:
            page += 1
            if page == len(PAGES):
                tr.outcome = 'completed'
                break
        elif action == BACK:
            page = max(0, page - 1)
    return tr


def simulate_population(n, design='A', seed=0, archetypes=None):
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES) if archetypes is None else archetypes
    shares = np.array([ARCHETYPES[a]['share'] for a in names])
    picks = rng.choice(len(names), size=n, p=shares / shares.sum())
    return [simulate_user(rng, i, names[k], design) for i, k in enumerate(picks)]
