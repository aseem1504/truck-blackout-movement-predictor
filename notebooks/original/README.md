# Original notebook

`03_no_info_feature_generation_and_model.ipynb` is the 2020 notebook this repo grew from, cleaned for public release.

What was removed or changed:

- Every cell output. Outputs held real query results, so none of them are here.
- Database and Mongo connection lines with hosts, users and passwords. They are replaced by `from config import *`, which reads settings from environment variables (see `../../.env.example`).
- Internal hostnames, IP addresses and API links. They are replaced by placeholders such as `GPS_API_BASE`.

What was kept: the logic, the queries, the half-finished experiments and the mistakes. Look for `displacement_travelled` in the feature list (the leak), the ordinal encoder on labels, and the RMSE cross-validation on a classifier. The main README explains each one.

Two quirks worth knowing if you read it:

- The target is flipped. In the notebook, 1 means the truck did **not** move. This repo uses 1 = moved, which reads more naturally.
- It starts with `%run WIRE_ISSUES_MODULE.py`. That rule engine lives in the sister repo, [truck-gps-tamper-detector](https://github.com/aseem1504/truck-gps-tamper-detector).

The notebook will not run without the company systems behind it. To see the ideas run end to end, use `python demo.py` in the repo root.
