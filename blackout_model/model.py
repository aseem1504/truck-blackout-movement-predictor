"""Logistic regression for "did the truck move while its tracker was dark?".

Three versions live here, in the order the work actually went:

1. original_recipe   what the 2020 notebook did: ordinal codes for every label, no scaling,
                     the leaky feature in, accuracy scored on the training rows.
2. honest_pipeline   the same model family done properly: one-hot labels, scaled numbers,
                     only features known when the blackout starts, scored on held-out rows.
3. pick_threshold    turn a probability into a field decision using what a wrong call costs.

A random forest is included only as a yardstick. If it barely beats the logistic regression,
the simpler model wins, because ops can read its coefficients.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import GEO, NOMINAL, WIRE_FLAGS


def _metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {'accuracy': accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred),
            'roc_auc': roc_auc_score(y, prob),
            'brier': brier_score_loss(y, prob)}


# ---------------------------------------------------------------- 1. the original recipe
def original_recipe(X, y, seed=0):
    """Reproduce the notebook: every label turned into 1, 2, 3..., raw numbers, default LR.

    Returns the score on the training rows (what the notebook reported first) and the
    score on a 20% test split (what it reported second)."""
    Xo = X.copy()
    for c in NOMINAL:
        Xo[c] = Xo[c].astype('category').cat.codes + 1
    lr = LogisticRegression(max_iter=100)  # the 2020 default; it may not fully converge
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr.fit(Xo, y)
        train_acc = accuracy_score(y, lr.predict(Xo))
        X_tr, X_te, y_tr, y_te = train_test_split(Xo, y, test_size=0.2, random_state=seed)
        lr2 = LogisticRegression(max_iter=100).fit(X_tr, y_tr)
    return {'train_accuracy': train_acc,
            'test_accuracy': accuracy_score(y_te, lr2.predict(X_te))}


# ---------------------------------------------------------------- 2. the honest pipeline
def honest_pipeline(X, C=1.0):
    nominal = [c for c in X.columns if c in NOMINAL]
    flags = [c for c in X.columns if c in WIRE_FLAGS + GEO + ['night']]
    numeric = [c for c in X.columns if c not in nominal + flags]
    pre = ColumnTransformer([
        ('labels', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal),
        ('flags', 'passthrough', flags),          # 0/1 already; odds ratio = flag on vs off
        ('numbers', StandardScaler(), numeric),
    ])
    return Pipeline([('prep', pre), ('lr', LogisticRegression(C=C, max_iter=2000))])


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)


def cross_validated(model, X, y, folds=5, seed=0):
    """Out-of-fold probabilities: every row is scored by a model that never saw it."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    if isinstance(model, RandomForestClassifier):
        X = X.copy()
        for c in NOMINAL:
            if c in X:
                X[c] = X[c].astype('category').cat.codes
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]


def compare(X, y):
    """Logistic regression vs a random forest yardstick vs always guessing the majority."""
    base = np.full(len(y), y.mean())
    rows = [{'model': 'always guess "moved"', **_metrics(y, base)}]
    rows[0]['accuracy'] = max(y.mean(), 1 - y.mean())
    rows[0]['roc_auc'] = 0.5
    probs = {}
    for name, m in [('logistic_regression', honest_pipeline(X)), ('random_forest', forest())]:
        probs[name] = cross_validated(m, X, y)
        rows.append({'model': name, **_metrics(y, probs[name])})
    return pd.DataFrame(rows).round(3), probs


# ---------------------------------------------------------------- reading the model
def odds_ratios(X, y):
    """Fit on all rows and turn each coefficient into an odds ratio.

    Odds ratio = e to the power of the coefficient. For a 0/1 flag it means "odds of moving
    get multiplied by this much when the flag is on". For a scaled number it is the same but
    per one standard deviation increase. For a one-hot label it is versus the dropped
    baseline label."""
    pipe = honest_pipeline(X).fit(X, y)
    names = pipe.named_steps['prep'].get_feature_names_out()
    names = [n.split('__', 1)[1] for n in names]
    coef = pipe.named_steps['lr'].coef_[0]
    out = pd.DataFrame({'feature': names, 'coef': coef, 'odds_ratio': np.exp(coef)})
    return out.reindex(out.coef.abs().sort_values(ascending=False).index).reset_index(drop=True).round(3)


def rfe_ranking(X, y, n_keep=8):
    """Recursive feature elimination, as the original did, but on scaled one-hot features."""
    pipe = honest_pipeline(X)
    Z = pipe.named_steps['prep'].fit_transform(X)
    names = [n.split('__', 1)[1] for n in pipe.named_steps['prep'].get_feature_names_out()]
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=n_keep).fit(Z, y)
    return sorted(zip(rfe.ranking_, names))


# ---------------------------------------------------------------- 3. turning it into a decision
def pick_threshold(y, prob, cost_missed=5.0, cost_wasted_visit=1.0):
    """Choose the cut-off that makes the cheapest mistakes.

    Total cost = (missed hidden trucks x cost of a miss) + (wasted visits x cost of a visit)
    cost = cost_missed * FN + cost_wasted_visit * FP
    """
    y = np.asarray(y)
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = prob >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append({'threshold': t, 'flagged_pct': pred.mean(), 'missed': fn, 'wasted_visits': fp,
                     'precision': precision_score(y, pred, zero_division=0),
                     'recall': recall_score(y, pred),
                     'cost': cost_missed * fn + cost_wasted_visit * fp})
    table = pd.DataFrame(rows)
    best = table.loc[table.cost.idxmin()]
    return best, table.round(3)


def capacity_view(y, prob, shares=(0.1, 0.2, 0.3, 0.5)):
    """If ops can only visit the riskiest X% of blackouts, how many of those really moved?"""
    y = np.asarray(y)
    order = np.argsort(-prob)
    rows = []
    for s in shares:
        k = int(round(s * len(y)))
        hit = y[order[:k]]
        rows.append({'visit_top': f"{s:.0%}", 'trucks_visited': k, 'really_moved': hit.mean(),
                     'share_of_all_movers_caught': hit.sum() / y.sum(),
                     'lift_vs_random': hit.mean() / y.mean()})
    return pd.DataFrame(rows).round(2)
