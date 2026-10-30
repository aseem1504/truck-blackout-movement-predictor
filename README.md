# Truck Blackout Movement Predictor

A logistic regression that answers one question for a truck GPS fleet. When a tracker goes dark, did the truck keep moving?

Built in 2020 for a truck GPS fleet, rebuilt in 2026 for public release. All data in this repo is synthetic.

Sister repos: [truck-gps-tamper-detector](https://github.com/aseem1504/truck-gps-tamper-detector) finds the wiring faults that feed this model. [gps-device-health-scorecard](https://github.com/aseem1504/gps-device-health-scorecard) is the daily device health job on the same ping data.

## The problem

Trackers go quiet all the time. Most of it is honest. A truck sits in a garage, a SIM dies, a wire works loose. Some of it is not. A driver cuts the power wire to run an unofficial trip, or to hide a truck the bank wants back.

On the dashboard, both look the same: "No info." Ops could not visit every blackout. Lenders who financed the trucks cared about one thing. Is this truck really parked, or is someone hiding it?

The model gives every blackout a probability that the truck moved while dark. Ops visits the riskiest ones first.

## How it works

**The target.** Take the last ping before the blackout and the first ping after it. If they are more than 1 km apart, the truck moved while hidden.

**The features.** Only what is known the moment the tracker goes quiet:

- device model, SIM network, sales channel, install type
- wiring flags from the last few days (power cut while driving, ignition wire cut, loose wire and so on)
- how many blackouts this truck had in the last 30 days, and how long they lasted
- was it driving or parked just before it went dark
- time of day, and a geohash of where it went dark

**The model.** Logistic regression. Labels like network and device model are one-hot encoded. Numbers are scaled. Wiring flags stay as plain 0 or 1, so their effect reads cleanly.

Why logistic regression and not something fancier? Because ops has to trust it. Every coefficient turns into a plain sentence like "a power cut while driving makes the odds of movement 3.7 times higher". A random forest is in the repo as a yardstick. If it barely wins, the simple model stays.

## Results on synthetic data

`python demo.py` creates 3,000 blackout events. 55% of them moved more than 1 km.

| Model (5-fold cross-validation) | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Always guess "moved" | 0.55 | 0.55 | 1.00 | 0.50 |
| Logistic regression | 0.69 | 0.71 | 0.74 | 0.75 |
| Random forest | 0.67 | 0.68 | 0.76 | 0.73 |

The random forest does not beat the logistic regression. So the simpler, readable model wins.

What the model learned, as odds ratios:

| Feature | Odds ratio | In plain words |
|---|---|---|
| Parked just before going dark | 0.25 | Odds of movement drop to a quarter |
| Power cut while driving (wiring flag) | 3.7 | Odds go up 3.7 times |
| Ignition wire cut (wiring flag) | 2.3 | Odds more than double |
| Went dark at night (8 pm to 4 am) | 1.5 | Odds up by half |

The odds ratio is e raised to the power of the coefficient: odds ratio = e^coef. A value above 1 means the feature pushes toward "moved". Below 1 pushes toward "parked".

## The mistake worth talking about

The original 2020 notebook reported 98% accuracy. It was fake.

The feature list included `displacement_travelled`. That is the exact distance the target is built from. The model was being shown the answer. Rerun that recipe here and it scores 99.7%. Take that one column out and the same recipe drops to 62%.

This is called target leakage. It is easy to miss when the feature list is long and built by many helper functions. The fix is one rule. Only use what you would know at the moment you need to make the call. Here, that moment is when the tracker goes quiet.

The original had three smaller problems too, all fixed here:

- **Labels were turned into numbers.** Airtel became 1, Vodafone 2, Idea 3. That tells the model Idea is "three times" Airtel, which means nothing. One-hot encoding fixes it.
- **Nothing was scaled.** Durations in seconds sat next to 0/1 flags, so the default solver struggled and the coefficients could not be compared.
- **The cross-validation used the wrong score.** It measured root mean squared error on a yes/no classifier. That number has no useful meaning here.

## From a probability to a field decision

A model score is not a decision. Ops has to know who to visit. The demo shows two ways to decide.

**When visits are capped.** Most weeks, ops has a fixed number of field agents.

| Visit the riskiest | Of those, really moved | Share of all movers caught |
|---|---|---|
| 10% | 90% | 16% |
| 20% | 85% | 31% |
| 30% | 80% | 44% |
| 50% | 74% | 67% |

A random visit finds a moved truck 55% of the time. Sending agents to the top 10% finds one 90% of the time. That is the real value of the model. It does not catch everything. It makes each visit count.

**When visits are not capped.** Pick the cut-off from what each mistake costs.

Total cost = (missed hidden trucks x cost of a miss) + (wasted visits x cost of a visit)

`cost = cost_missed * FN + cost_wasted_visit * FP`

| A miss costs ... a wasted visit | Flag above | Share flagged | Precision | Recall |
|---|---|---|---|---|
| half of | 0.60 | 44% | 0.76 | 0.61 |
| the same as | 0.55 | 51% | 0.74 | 0.68 |
| 3 times | 0.25 | 89% | 0.59 | 0.96 |

The last row is the non-obvious one. If a missed hidden truck is very costly (say, a lender's asset that disappears), the "best" cut-off flags almost every blackout. At that point the model is not saving visits. The better product move is a cheaper first check, like an automated call to the driver, before a field visit. The model tells you when that is worth building.

## Run it

```bash
pip install -r requirements.txt
python demo.py      # full synthetic run, under 15 seconds
pytest              # checks the leakage guard, odds ratios and threshold logic
```

## Repo layout

```
blackout_model/
  features.py      feature building, geohash, time slots, the leaky-feature switch
  model.py         original recipe, honest pipeline, odds ratios, RFE, thresholds, capacity view
  simulate.py      synthetic blackout events with a known hidden truth
  geo.py           haversine distance
demo.py            end-to-end run
tests/             unit tests
data/sample/       a sample of the synthetic events
notebooks/original the 2020 notebook, cleaned for public release
```

## About the data

Nothing here comes from real vehicles. `simulate.py` builds blackout events with the same fields the production pipeline pulled. A hidden rule decides which trucks moved. It is driven by driving state, wiring flags, blackout length and time of day, with noise on top. So the results show how the method behaves, not how well it did in production.

## License

MIT. See `LICENSE`.
