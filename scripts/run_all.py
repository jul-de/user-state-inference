"""Run every analysis and write the data the README and the demo page use.

    python scripts/run_all.py          (about a minute)

Outputs docs/data/results.json (all reported numbers) and docs/data/bank.json
(example traces for the trace explorer). Every population has a fixed seed.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from usersim.evaluate import (Features, best_permutation, calibration, name_states,  # noqa: E402
                              recovery, relabel, rule_baseline)
from usersim.hmm import InteractionHMM, params_from_labels  # noqa: E402
from usersim.simulate import (ACTIONS, ARCHETYPES, CONF, DESIGNS, PAGES, STATES,  # noqa: E402
                              simulate_population)

OUT = ROOT / 'docs' / 'data'
N_TRAIN, N_TEST, N_LABELLED, N_AB = 3000, 1000, 25, 2000


def states_of(traces):
    return [np.array(t.states) for t in traces]


def anchored(features, labelled, train, **kw):
    """HMM anchored on a small labelled study, then refined by EM on unlabelled traffic."""
    la, lx = features.transform(labelled)
    P = params_from_labels(la, lx, states_of(labelled))
    if not kw.get('use_dwell', True):
        P.mu[:], P.sd[:] = 0.0, 1.0
    if not kw.get('use_action', True):
        P.B[:] = 1.0 / P.B.shape[1]
    if not kw.get('sequential', True):
        occ = np.bincount(np.concatenate(states_of(labelled)), minlength=4) + 1.0
        P.pi = occ / occ.sum()
        P.A = np.tile(P.pi, (4, 1))
    a, x = features.transform(train)
    return InteractionHMM(**kw).fit(a, x, init=P)


def summary(name, post, truth):
    r = recovery(post, truth)
    c = calibration(post, truth, CONF)
    return {'model': name, 'balanced_accuracy': r['balanced_accuracy'], 'accuracy': r['accuracy'],
            'recall': r['recall'], 'precision': r['precision'], 'confusion_matrix': r['confusion_matrix'],
            'ece_confused': c['ece'], 'calibration_confused': c['bins']}


def user_bootstrap(posts, truth, n=1000, seed=0):
    """Recovery metrics with 95% CIs from resampling held-out *users* (each user's steps
    stay together, since steps from one person are correlated). `posts` is a list of
    models' posteriors; the same resamples are used for every model, so the CI for a
    difference between two models is paired."""
    rng = np.random.default_rng(seed)
    preds = [[g.argmax(1) for g in post] for post in posts]

    def metrics(idx):
        y = np.concatenate([truth[i] for i in idx])
        out = []
        for pr in preds:
            p = np.concatenate([pr[i] for i in idx])
            rec = [((p == k) & (y == k)).sum() / max(1, (y == k).sum()) for k in range(4)]
            prec_c = ((p == CONF) & (y == CONF)).sum() / max(1, (p == CONF).sum())
            out.append((np.mean(rec), rec[CONF], prec_c))
        return np.array(out)

    reps = np.array([metrics(rng.integers(0, len(truth), len(truth))) for _ in range(n)])  # (n, models, 3)
    lo, hi = np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)
    return reps, lo, hi


def ratio_ci(numA, denA, numB, denB, n=1000, seed=0):
    """95% CI for sum(numB)/sum(denB) - sum(numA)/sum(denA), resampling users (keeps each
    user's steps together, so the interval reflects users, not steps)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ia = rng.integers(0, len(numA), len(numA)); ib = rng.integers(0, len(numB), len(numB))
        out.append(numB[ib].sum() / denB[ib].sum() - numA[ia].sum() / denA[ia].sum())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    train = simulate_population(N_TRAIN, 'A', seed=1)       # unlabelled traffic
    test = simulate_population(N_TEST, 'A', seed=2)         # held out
    labelled = simulate_population(N_LABELLED, 'A', seed=99)  # small coded study
    feats = Features().fit(train)
    at, xt = feats.transform(test)
    truth = states_of(test)
    res = {'meta': {'states': STATES, 'actions': ACTIONS, 'pages': PAGES, 'archetypes': list(ARCHETYPES),
                    'n_train': N_TRAIN, 'n_test': N_TEST, 'n_labelled': N_LABELLED,
                    'test_steps': int(sum(len(s) for s in truth)),
                    'true_state_share': (np.bincount(np.concatenate(truth), minlength=4) / sum(len(s) for s in truth)).tolist()}}

    # ---- 1. Recovery: baseline, unsupervised, anchored, ablations ----
    a, x = feats.transform(train)
    unsup = InteractionHMM(seed=0, n_restarts=8).fit(a, x)
    unsup_post = unsup.posteriors(at, xt)
    main_model = anchored(feats, labelled, train)
    main_post = main_model.posteriors(at, xt)
    models = [
        summary('Rule-based heuristic', rule_baseline(test), truth),
        summary('HMM, no labels', relabel(unsup_post, name_states(unsup.params)), truth),
        summary(f'HMM, anchored on {N_LABELLED} labelled users', main_post, truth),
        summary('  ... dwell time only', anchored(feats, labelled, train, use_action=False).posteriors(at, xt), truth),
        summary('  ... actions only', anchored(feats, labelled, train, use_dwell=False).posteriors(at, xt), truth),
        summary('  ... no sequence context', anchored(feats, labelled, train, sequential=False).posteriors(at, xt), truth),
    ]
    posts = [rule_baseline(test), relabel(unsup_post, name_states(unsup.params)), main_post,
             anchored(feats, labelled, train, use_action=False).posteriors(at, xt),
             anchored(feats, labelled, train, use_dwell=False).posteriors(at, xt),
             anchored(feats, labelled, train, sequential=False).posteriors(at, xt)]
    reps, lo, hi = user_bootstrap(posts, truth)
    for i, m in enumerate(models):
        m['ci95'] = {'balanced_accuracy': [lo[i, 0], hi[i, 0]], 'confused_recall': [lo[i, 1], hi[i, 1]],
                     'confused_precision': [lo[i, 2], hi[i, 2]]}
        m['ci95'] = {k: [float(v[0]), float(v[1])] for k, v in m['ci95'].items()}
    # Paired differences: anchored HMM minus the rule-based heuristic, same user resamples.
    d = reps[:, 2, :] - reps[:, 0, :]
    res['anchored_vs_rule'] = {k: {'diff': float(np.mean(d[:, j])), 'ci95': [float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]}
                               for j, k in enumerate(['balanced_accuracy', 'confused_recall', 'confused_precision'])}
    # Unsupervised ceiling: even the best possible labelling of its states.
    models[1]['balanced_accuracy_best_labelling'] = recovery(relabel(unsup_post, best_permutation(unsup_post, truth)), truth)['balanced_accuracy']
    res['models'] = models

    # ---- 2. What the unlabelled model found instead: user types ----
    # Among truly "progressing" test steps, which fitted state got them, split by archetype.
    named = name_states(unsup.params)
    raw = np.concatenate([g.argmax(1) for g in unsup_post])
    y = np.concatenate(truth)
    arch = np.concatenate([[t.archetype] * len(t.states) for t in test])
    prog = y == 0
    fitted_order = np.argsort(unsup.params.mu)  # fitted states from fastest to slowest
    res['unsupervised_split'] = {
        'fitted_state_dwell_mu': unsup.params.mu[fitted_order].tolist(),
        'fitted_state_named_as': [STATES[named[k]] for k in fitted_order],
        'by_archetype': {ar: [float(((raw == k) & prog & (arch == ar)).sum() / max(1, (prog & (arch == ar)).sum()))
                              for k in fitted_order] for ar in ARCHETYPES},
    }

    # ---- 3. How many labelled users does anchoring need? ----
    curve = []
    for n in [0, 5, 10, 25, 50, 100, 200]:
        bal, conf_rec, conf_prec = [], [], []
        for rep in range(5):
            if n == 0:
                m = InteractionHMM(seed=rep, n_restarts=4).fit(a, x)
                post = relabel(m.posteriors(at, xt), name_states(m.params))
            else:
                lab = simulate_population(n, 'A', seed=1000 + 17 * rep + n)
                post = anchored(feats, lab, train).posteriors(at, xt)
            r = recovery(post, truth)
            bal.append(r['balanced_accuracy']); conf_rec.append(r['recall'][CONF]); conf_prec.append(r['precision'][CONF])
        curve.append({'n_labelled': n, 'balanced_accuracy': float(np.mean(bal)), 'ba_min': float(np.min(bal)),
                      'ba_max': float(np.max(bal)), 'confused_recall': float(np.mean(conf_rec)),
                      'confused_precision': float(np.mean(conf_prec))})
    res['label_curve'] = curve

    # ---- 4. Where it fails across user types ----
    pers = {}
    for ar in ARCHETYPES:
        idx = [i for i, t in enumerate(test) if t.archetype == ar]
        r = recovery([main_post[i] for i in idx], [truth[i] for i in idx])
        steps = np.concatenate([truth[i] for i in idx])
        pers[ar] = {'n_users': len(idx), 'balanced_accuracy': r['balanced_accuracy'], 'recall': r['recall'],
                    'precision': r['precision'], 'share_confused': float((steps == CONF).mean())}
    res['personas'] = pers

    # ---- 5. Same behaviour, different cause ----
    # Steps with the same observation (went back after 8-14 s) whose true state differs.
    rows = []
    for i, t in enumerate(test):
        for s in range(len(t.states)):
            if t.actions[s] == 1 and 8 <= t.dwell[s] <= 14 and t.states[s] in (1, 2):
                rows.append((i, s, t.states[s], float(main_post[i][s, CONF]), float(main_post[i][s, 1])))
    rows = np.array(rows)
    sb = {'n_steps': int(len(rows))}
    for st, nm in [(1, 'deliberating'), (2, 'confused')]:
        m = rows[:, 2] == st
        sb[nm] = {'n': int(m.sum()), 'mean_p_confused': float(rows[m, 3].mean()), 'mean_p_deliberating': float(rows[m, 4].mean()),
                  'correct': float(((rows[m, 3] > rows[m, 4]) if st == 2 else (rows[m, 4] > rows[m, 3])).mean())}
    res['same_behaviour'] = sb

    # ---- 6. Interface change: design A vs B on a synthetic population ----
    ab = {}
    pops = {'A': simulate_population(N_AB, 'A', seed=10), 'B': simulate_population(N_AB, 'B', seed=11)}
    per_user = {}
    for d, pop in pops.items():
        pa, px = feats.transform(pop)
        post = main_model.posteriors(pa, px)
        tr_conf = np.zeros(len(PAGES)); inf_conf = np.zeros(len(PAGES)); cnt = np.zeros(len(PAGES))
        u_true, u_inf, u_n, u_abandon = [], [], [], []
        for t, g in zip(pop, post):
            pg = np.array(t.pages)
            st = np.array(t.states)
            np.add.at(tr_conf, pg, st == CONF)
            np.add.at(inf_conf, pg, g[:, CONF])
            np.add.at(cnt, pg, 1)
            opt = pg == PAGES.index('Options')
            u_true.append((st[opt] == CONF).sum()); u_inf.append(g[opt, CONF].sum()); u_n.append(opt.sum())
            u_abandon.append(t.outcome == 'abandoned')
        per_user[d] = {'true': np.array(u_true, float), 'inf': np.array(u_inf), 'n': np.array(u_n, float),
                       'abandon': np.array(u_abandon, float), 'arch': np.array([t.archetype for t in pop])}
        by_arch = {}
        for ar in ARCHETYPES:
            u = per_user[d]; m = u['arch'] == ar
            by_arch[ar] = {'abandon': float(u['abandon'][m].mean()),
                           'true_conf_options': float(u['true'][m].sum() / u['n'][m].sum()),
                           'inferred_conf_options': float(u['inf'][m].sum() / u['n'][m].sum())}
        ab[d] = {
            'completion': float(np.mean([t.outcome == 'completed' for t in pop])),
            'abandonment': float(np.mean([t.outcome == 'abandoned' for t in pop])),
            'median_steps': float(np.median([len(t.states) for t in pop])),
            'true_conf_by_page': (tr_conf / cnt.clip(1)).tolist(),
            'inferred_conf_by_page': (inf_conf / cnt.clip(1)).tolist(),
            'by_archetype': by_arch,
        }
    A_, B_ = per_user['A'], per_user['B']
    ab['diff'] = {}
    for key, num in [('true_conf_options', 'true'), ('inferred_conf_options', 'inf')]:
        va, vb = A_[num].sum() / A_['n'].sum(), B_[num].sum() / B_['n'].sum()
        ab['diff'][key] = {'A': float(va), 'B': float(vb), 'diff': float(vb - va), 'ci95': ratio_ci(A_[num], A_['n'], B_[num], B_['n'])}
    ones_a, ones_b = np.ones(len(A_['abandon'])), np.ones(len(B_['abandon']))
    ab['diff']['abandonment'] = {'A': float(A_['abandon'].mean()), 'B': float(B_['abandon'].mean()),
                                 'diff': float(B_['abandon'].mean() - A_['abandon'].mean()),
                                 'ci95': ratio_ci(A_['abandon'], ones_a, B_['abandon'], ones_b)}
    # Where truly confused steps on the redesigned page went (what the model called them).
    pa, px = feats.transform(pops['B'])
    postB = main_model.posteriors(pa, px)
    P = np.concatenate(postB); S = np.concatenate([t.states for t in pops['B']]); pg = np.concatenate([t.pages for t in pops['B']])
    sel = (S == CONF) & (pg == PAGES.index('Options'))
    ab['B_options_confused_read_as'] = (np.bincount(P[sel].argmax(1), minlength=4) / sel.sum()).tolist()
    res['ab'] = ab

    # ---- 7. Example traces for the explorer ----
    bank = []
    for d in DESIGNS:
        for k, ar in enumerate(ARCHETYPES):
            pop = simulate_population(8, d, seed=500 + 10 * k + (0 if d == 'A' else 5), archetypes=[ar])
            pa, px = feats.transform(pop)
            post = main_model.posteriors(pa, px)
            rule = rule_baseline(pop)
            for t, g, r in zip(pop, post, rule):
                bank.append({'design': d, 'archetype': ar, 'traits': {k2: round(v, 2) for k2, v in t.traits.items()},
                             'outcome': t.outcome, 'pages': t.pages, 'actions': t.actions, 'dwell': t.dwell,
                             'states': t.states, 'posterior': np.round(g, 3).tolist(), 'rule': r.argmax(1).tolist()})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))
    (OUT / 'bank.json').write_text(json.dumps(bank, separators=(',', ':')))
    print_report(res)


def print_report(res):
    print('\nRecovery on held-out users (balanced accuracy | confused recall / precision | ECE for P(confused))')
    for m in res['models']:
        print(f"  {m['model']:42s} {m['balanced_accuracy']:.3f} | {m['recall'][CONF]:.2f} / {m['precision'][CONF]:.2f} | {m['ece_confused']:.3f}")
    for m in res['models']:
        c = m['ci95']['balanced_accuracy']
        print(f"    {m['model'].strip():40s} balanced acc 95% CI [{c[0]:.3f}, {c[1]:.3f}]")
    print('  anchored minus rule (paired):', {k: (round(v['diff'], 3), [round(x, 3) for x in v['ci95']]) for k, v in res['anchored_vs_rule'].items()})
    print(f"  (no-label HMM with the best possible labelling: {res['models'][1]['balanced_accuracy_best_labelling']:.3f})")
    print('\nLabelled users needed:')
    for c in res['label_curve']:
        print(f"  n={c['n_labelled']:3d}  balanced acc {c['balanced_accuracy']:.3f} ({c['ba_min']:.3f}-{c['ba_max']:.3f})  confused recall {c['confused_recall']:.2f}  precision {c['confused_precision']:.2f}")
    print('\nUnlabelled model, truly-progressing steps by fitted state (fast -> slow):', res['unsupervised_split']['fitted_state_named_as'])
    for ar, v in res['unsupervised_split']['by_archetype'].items():
        print(f"  {ar:13s}", ' '.join(f'{p:.2f}' for p in v))
    print('\nBy persona:')
    for ar, v in res['personas'].items():
        print(f"  {ar:13s} bal {v['balanced_accuracy']:.3f}  confused recall {v['recall'][CONF]:.2f}  precision {v['precision'][CONF]:.2f}  share confused {v['share_confused']:.3f}")
    print('\nSame behaviour (back after 8-14 s):', json.dumps(res['same_behaviour']))
    print('\nDesign A vs B:')
    for k, v in res['ab']['diff'].items():
        print(f"  {k:22s} A {v['A']:.3f}  B {v['B']:.3f}  diff {v['diff']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for d in 'AB':
        print(f"  {d}: completion {res['ab'][d]['completion']:.3f}  by archetype:",
              {ar: round(v['abandon'], 3) for ar, v in res['ab'][d]['by_archetype'].items()})


if __name__ == '__main__':
    main()
