"""End-to-end demo on synthetic data.

    python demo.py

1. reproduces the original 2020 notebook and shows why its 98% accuracy was fake
2. trains the honest logistic regression and checks it against a random forest
3. reads the coefficients as odds ratios
4. turns probabilities into field decisions: a visit-capacity view and a cost-based cut-off
"""
import os

import pandas as pd

from blackout_model import simulate
from blackout_model.features import build_features
from blackout_model.model import capacity_view, compare, odds_ratios, original_recipe, pick_threshold, rfe_ranking

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
os.makedirs('outputs', exist_ok=True)

events = simulate.no_info_events()
events.to_csv('outputs/no_info_events_synthetic.csv', index=False)

print("\n=== 1. The original recipe ===")
X_leak, y = build_features(events, include_leaky=True)
orig = original_recipe(X_leak, y)
print(f"blackout events: {len(y)}   moved more than 1 km while dark: {y.mean():.0%}")
print(f"accuracy on the rows it trained on: {orig['train_accuracy']:.3f}")
print(f"accuracy on a 20% test split:       {orig['test_accuracy']:.3f}")
print("Looks great. But `displacement_travelled` is the answer itself. Drop it:")
X, y = build_features(events)
print(f"same recipe, leaky feature removed: {original_recipe(X, y)['test_accuracy']:.3f}")

print("\n=== 2. Honest logistic regression (5-fold cross-validation) ===")
table, probs = compare(X, y)
print(table.to_string(index=False))
table.to_csv('outputs/model_comparison.csv', index=False)

print("\n=== 3. What the model learned (odds ratios) ===")
orx = odds_ratios(X, y)
print(orx.head(10).to_string(index=False))
orx.to_csv('outputs/odds_ratios.csv', index=False)
print("\nRFE ranking (1 = kept):")
for rank, name in rfe_ranking(X, y)[:10]:
    print(f"  {rank:>2}  {name}")

print("\n=== 4. From probability to a field decision ===")
print("a) Ops has a fixed number of field visits. Who goes first?")
cap = capacity_view(y, probs['logistic_regression'])
print(cap.to_string(index=False))
cap.to_csv('outputs/capacity_view.csv', index=False)
print("\nb) No visit cap. Pick the cut-off from what each mistake costs:")
for miss, visit in [(1, 2), (1, 1), (3, 1)]:
    best, grid = pick_threshold(y, probs['logistic_regression'], miss, visit)
    print(f"  miss : wasted visit = {miss}:{visit} -> flag above {best.threshold:.2f}: "
          f"flags {best.flagged_pct:.0%} of blackouts, precision {best.precision:.2f}, recall {best.recall:.2f}")
grid.to_csv('outputs/threshold_grid.csv', index=False)
print("\nOutputs written to ./outputs")
