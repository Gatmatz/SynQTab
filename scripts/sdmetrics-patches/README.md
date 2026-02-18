**TL; DR**: These patches are applied by running `<root_dir>/scripts/patch-sdmetrics.sh`.

### Introduction
This directory contains patch files for the `sdmetrics` library (https://github.com/sdv-dev/SDMetrics).

### What does this patch fix?
**`sdmetrics-concat-categorical.patch`** — Fixes a bug in `BaseDataAugmentationMetric.compute_breakdown()` where
`pd.concat` of real and synthetic training data causes categorical columns to fall back to `object` dtype when the
two DataFrames have different category sets (e.g. the synthetic data contains misspelled categorical values).
This subsequently crashes XGBoost with:
```
ValueError: DataFrame.dtypes for data must be int, float, bool or category.
```
The fix re-casts categorical columns back to `category` dtype after the concat operation.
