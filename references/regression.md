# Time series regression

Regressors map a whole series to a scalar. They are not forecasters: the target is not a future index of the same series unless you built that target yourself.

## Layout

`X` uses the same panel containers as classification (`numpy3D` of shape `(n_instances, n_channels, n_timepoints)`, or `nested_univ`). `y` is a one-dimensional numeric array of length `n_instances`.

`load_covid_3month` returns a nested frame and a numeric target. The archive loaders document `split` as `"TRAIN"` or `"TEST"`.

## Estimators

`TimeSeriesForestRegressor` (`sktime.regression.interval_based`) maps each series to a scalar. The default `n_estimators` is 200. Keep `random_state` fixed when you compare forests. Call `deps` on the constructor spec and install the returned requirements before `fit`.

```python
from sktime.regression.interval_based import TimeSeriesForestRegressor

reg = TimeSeriesForestRegressor(n_estimators=5, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

`DummyRegressor` (`sktime.regression.dummy`) ignores the series. `strategy="mean"` predicts the training mean. Use it to show that a proposed model beats a constant.

`RocketRegressor` (`sktime.regression.kernel_based`) uses `num_kernels`. It is the pipeline `rocket * StandardScaler(with_mean=False) * RidgeCV`. Call `deps("RocketRegressor()")` and install the returned requirements before `fit`.

## Metrics

Predict-a-scalar tasks use a tabular metric (`sklearn.metrics.mean_absolute_error` or the sktime tabular metric wrappers), not `MeanAbsoluteError` from `sktime.performance_metrics.forecasting`. The forecasting metric aligns on a time index and will reject a vector of per-series targets.

Keep instances from the same experiment together when splitting. Random row splits are valid only when each row is an independent series.
