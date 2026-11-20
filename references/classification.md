# Time series classification

Classifiers label an entire series. They do not forecast the next time point. This reference targets sktime 1.2.0.

## Panel layout

```python
from sktime.datasets import load_unit_test, load_arrow_head

X_train, y_train = load_unit_test(split="TRAIN", return_type="numpy3D")
X_test, y_test = load_unit_test(split="TEST", return_type="numpy3D")
```

- `split` is `"TRAIN"` or `"TEST"`. Omitting it concatenates both.
- `return_type="numpy3D"` has shape `(n_instances, n_channels, n_timepoints)`. The unit-test problem is `(20, 1, 24)` for train and `(22, 1, 24)` for test.
- The default `return_type` is `nested_univ`: a `DataFrame` of `Series`. `load_arrow_head(split="TRAIN")` returns that frame.
- Unequal-length series need an estimator whose tags allow them, or a padder. Do not silently truncate to the shortest length unless the scientific question is about that prefix.

Labels from the bundled loaders are strings (`"1"`, `"2"`, …). Compare predictions to `y_test` with the same dtype.

## Interval forest

`TimeSeriesForestClassifier` (`sktime.classification.interval_based`) builds trees on the mean, standard deviation, and slope of random intervals. Call `deps` on the constructor spec and install the returned requirements before fitting.

```python
from sktime.classification.interval_based import TimeSeriesForestClassifier

clf = TimeSeriesForestClassifier(n_estimators=5, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)  # rows sum to 1
```

`n_estimators` defaults to 200. `min_interval` defaults to 3. `feature_importances_` is a frame of shape `(series_length, 3)` after a successful fit.

`DummyClassifier(strategy="prior")` predicts the training class frequencies. Use it as the baseline that ignores the series shape. Strategies include `"most_frequent"`, `"prior"`, `"stratified"`, `"uniform"`, and `"constant"`.

## Convolution and distance classifiers

Call `deps` on the constructor spec and install the returned requirements before `fit`.

| Estimator | Import | Parameter to get right |
| --- | --- | --- |
| `RocketClassifier` | `sktime.classification.kernel_based` | `num_kernels` (default 10000), `rocket_transform` in `{"rocket", "minirocket", "multirocket"}` |
| `KNeighborsTimeSeriesClassifier` | `sktime.classification.distance_based` | `distance` string such as `"euclidean"` or `"dtw"`, or a pairwise transformer. `n_neighbors` defaults to 1 |

`RocketClassifier` is the pipeline `rocket * StandardScaler(with_mean=False) * RidgeClassifierCV`. To swap the tabular classifier, build that pipeline yourself with `sktime.pipeline.make_pipeline` or the `*` operator rather than subclassing.

String distances on `KNeighborsTimeSeriesClassifier` (`"euclidean"`, `"dtw"`, `"ddtw"`, `"wdtw"`, `"wddtw"`, `"lcss"`, `"edr"`, `"erp"`, `"msm"`, `"twe"`) are the built-in equal-length univariate metrics. Pass a `sktime.dists_kernels` transformer when the collection is multivariate or unequal length, and check `capability:unequal_length` and `capability:multivariate`.

## Evaluation

Score with `clf.score(X_test, y_test)` or an explicit metric on `y_pred`. Keep the original train and test split for archive data. A random reshuffle of a fixed benchmark split is a different experiment. For repeated recordings from one subject, split by subject so windows from the same recording do not land on both sides.

`predict_proba` columns follow `classes_`. Do not assume column 0 is the positive class without reading `classes_`.
