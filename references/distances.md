# Distances

sktime exposes time-series distances as pairwise transformers in `sktime.dists_kernels`. A distance matrix is the output of `fit(X).transform(X)`, not a free function return, for the estimator interface. This reference targets sktime 1.2.0.

## Euclidean panel distance

`ScipyDist` adapts `scipy.spatial.distance.cdist` and implements the tabular pairwise interface. `FlatDist` applies that distance to a panel by flattening each series.

```python
from sktime.dists_kernels import FlatDist, ScipyDist
import numpy as np

X = np.arange(12.0).reshape(3, 1, 4)
distance = FlatDist(ScipyDist(metric="euclidean")).fit(X).transform(X)
```

`X` has shape `(3, 1, 4)`. Flattened Euclidean distance between `[0, 1, 2, 3]` and `[4, 5, 6, 7]` is `8.0`, the norm of a constant difference of 4 across four points. The matrix is `(3, 3)` with zeros on the diagonal.

`AggrDist(ScipyDist())` instead averages a tabular distance over time points. Use it when the scientific distance is a mean of per-time-point distances rather than the distance of the concatenated series. `metric` accepts the `cdist` names (`"euclidean"`, `"cityblock"`, `"minkowski"`, and the rest listed in the `ScipyDist` docstring).

## DTW

```python
from sktime.dists_kernels import DtwDist

distance = DtwDist(window=None).fit(X).transform(X)
```

Call `deps("DtwDist()")` and install the returned requirements before `fit`.

- `weighted=False`, `derivative=False` is DTW.
- `derivative=True` is derivative DTW.
- `weighted=True` is weighted DTW.
- `window` is an integer Sakoe–Chiba radius, or `None` for no window. It is not a fraction of the series length. Passing `window` ignores `itakura_max_slope`.
- The built-in DTW path expects equal-length series. For unequal length, the class docstring points at `sktime.alignment` aligners (`DistFromAligner` with `AlignerDTW`).

`KNeighborsTimeSeriesClassifier(distance="dtw")` uses that distance by string name. A pairwise transformer can be passed as `distance` when you need the estimator object rather than the string.

## Using a precomputed matrix

scikit-learn estimators that accept `metric="precomputed"` need a square matrix from `transform(X)` at fit, and a rectangular matrix from `transform(X_test)` against the training panel at predict. Fit the distance on the training panel only, then transform the test panel against that training panel. Recomputing distances on the pooled train and test sample changes the neighbors.
