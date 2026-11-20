# Anomaly detection and change points

sktime detectors live in `sktime.detection`. The public `fit_predict` returns a sparse point table. Dense per-time-point output is a separate call. This reference targets sktime 1.2.0.

## HampelDetector

`deps("HampelDetector()")` is empty when the detector needs only the sktime core install. Call it before fitting.

```python
from sktime.detection.hampel import HampelDetector

detector = HampelDetector(window_length=10, n_sigma=3.0, k=1.4826)
points = detector.fit_predict(y)
scores = detector.transform_scores(y)
```

- Input is a univariate `pd.Series` (`X_inner_mtype` is `pd.Series`). `capability:multivariate` is false.
- `window_length` is the centered window. `n_sigma` scales the MAD threshold. `k=1.4826` is the Gaussian consistency constant.
- `fit_predict` returns a `DataFrame` with an integer column `ilocs`. Those values are positions in the series, not timestamps. Map them through `y.index[points["ilocs"]]` when you need the time labels.
- `transform_scores` returns one score per time point: absolute deviation divided by `k * MAD`. A single spike in an otherwise linear series lands in `ilocs` and produces the largest score.
- `predict_scores` raises `NotImplementedError` on this estimator. Call `transform_scores`.
- `fit_is_empty` is true: there is no training distribution beyond the window passed to `predict`. The threshold is local to each window, so a slow trend is not itself an anomaly, and a patch of outliers can inflate the local MAD until points are no longer flagged.

## ClaSPSegmentation

Call `deps("ClaSPSegmentation()")` and install the returned requirements before `fit`.

```python
from sktime.detection.clasp import ClaSPSegmentation

segmenter = ClaSPSegmentation(period_length=10, n_cps=1, exclusion_radius=0.05)
points = segmenter.fit_predict(y)
```

- `period_length` is the window used to build the classification score profile. `find_dominant_window_sizes` in the same module estimates it from a series.
- `n_cps` is how many change points to return.
- `_predict` returns the discovered integer locations. The public detector interface wraps point output into the sparse table, the same interface `HampelDetector.fit_predict` uses.
- After prediction the estimator stores `profiles` and `scores`. `transform_scores` is documented to return the first profile as a dense series.

Change-point locations are hypotheses about the series, not labeled events. Compare them to a known breakpoint with an explicit tolerance. `DirectedHausdorff` in `sktime.performance_metrics.detection` scores two sets of time points.

## Choosing a detector

| Question | Estimator family |
| --- | --- |
| Is this time point an outlier relative to its neighbors? | `HampelDetector`, threshold detectors |
| Where does the regime change? | `ClaSPSegmentation`, `PELT`, binary segmentation |
| Do I have supervised event labels? | A supervised detector, only if the labels are the learning target and are absent from the features |

`PELT` and the skchange adapters have their own penalty and cost arguments. Read the estimator docstring, call `deps` on the constructor, and install the returned requirements before fitting. Do not threshold an anomaly score and call the result a scientific event without a separate definition of that event.
