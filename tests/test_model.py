import numpy as np

from blackout_model import simulate
from blackout_model.features import LEAKY, build_features, time_slot
from blackout_model.model import capacity_view, honest_pipeline, odds_ratios, pick_threshold


def _data(n=600):
    return build_features(simulate.no_info_events(n=n, seed=3))


def test_time_slot_covers_whole_day():
    assert time_slot(0) == 1
    assert time_slot(14400) == 2          # boundary that the original dropped
    assert time_slot(86399) == 6


def test_honest_features_exclude_the_answer():
    X, _ = _data()
    assert LEAKY not in X.columns
    X_leak, _ = build_features(simulate.no_info_events(n=50, seed=3), include_leaky=True)
    assert LEAKY in X_leak.columns


def test_pipeline_beats_coin_flip():
    X, y = _data()
    p = honest_pipeline(X).fit(X, y).predict_proba(X)[:, 1]
    assert ((p >= 0.5) == y).mean() > 0.6


def test_wire_cut_raises_odds():
    X, y = _data(2000)
    orx = odds_ratios(X, y).set_index('feature')
    assert orx.loc['intentional_wire_cut', 'odds_ratio'] > 1
    assert orx.loc['current_state_STOPPAGE', 'odds_ratio'] < 1


def test_threshold_moves_with_costs():
    X, y = _data()
    p = honest_pipeline(X).fit(X, y).predict_proba(X)[:, 1]
    cheap_miss, _ = pick_threshold(y, p, cost_missed=1, cost_wasted_visit=3)
    dear_miss, _ = pick_threshold(y, p, cost_missed=3, cost_wasted_visit=1)
    assert dear_miss.threshold <= cheap_miss.threshold


def test_capacity_view_top_slice_is_richer():
    X, y = _data()
    p = honest_pipeline(X).fit(X, y).predict_proba(X)[:, 1]
    cap = capacity_view(y, p)
    assert cap.loc[0, 'really_moved'] >= np.mean(y)
