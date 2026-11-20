# Datasets, splits, metrics, and benchmarks

Evaluation in sktime is a temporal backtest. This reference targets sktime 1.2.0. The benchmark runner itself is documented in `references/benchmarking.md`.

## Loaders

Use a dataset class and `load`. One set name returns that object. Several names return a tuple in the same order. `keys()` lists the names that class accepts.

Forecasting classes provide `"y"` and `"X"`. `"X"` is `None` when the series has no covariates. They do not define a train/test split. Classification and regression classes also provide `"X_train"`, `"y_train"`, `"X_test"`, and `"y_test"` when the problem ships one predefined split.

| Class | Import | What `load` needs |
| --- | --- | --- |
| `Airline`, `Longley`, `Lynx`, `PBS`, `ShampooSales`, `USChange` | `sktime.datasets` | `"y"`, and `"X"` when the series has covariates. Shipped with sktime. `Longley(y_name="TOTEMP")` selects the target. |
| `HierarchicalSalesToydata` | `sktime.datasets.forecasting` | `"y"`. Shipped demo hierarchy. |
| `ACSF1`, `ArrowHead`, `BasicMotions`, `GunPoint`, `ItalyPowerDemand`, `JapaneseVowels`, `OSULeaf`, `PLAID` | `sktime.datasets` | `"X"`, `"y"`, or the predefined `"X_train"` / `"y_train"` / `"X_test"` / `"y_test"` split. Shipped with sktime. |
| `Tecator` | `sktime.datasets.regression` | `"X"`, `"y"`, and the predefined train/test names. Shipped regression problem. |
| `ForecastingData(name=...)` | `sktime.datasets` | `"y"` and `"X"`. `name` is a Monash archive id, for example `"cif_2016_dataset"`. Downloads on first use. |
| `UCRUEADataset(name=...)` | `sktime.datasets` | Same classification names as above. `name` is an archive problem, for example `"Beef"`. `list_all()` returns the valid names. `return_mtype` defaults to `"pd-multiindex"`. Downloads on first use. |
| `M5Dataset`, `Solar` | `sktime.datasets` | Remote data. `M5Dataset` downloads the competition files. `Solar` calls the Sheffield Solar API. |
| `Macroeconomic` | `sktime.datasets` | `"y"`. Reads the series shipped with statsmodels. |

## Splits

```python
from sktime.split import ExpandingWindowSplitter, temporal_train_test_split

y_train, y_test = temporal_train_test_split(y, test_size=12)
y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
```

`test_size` is a count of time points when it is an integer, and a fraction of the index when it is a float in `(0, 1)`. The default fraction when both sizes are omitted is 0.25. The split is the tail of the series, applied per series on a panel. `sktime.forecasting.model_selection.temporal_train_test_split` still exists and warns; import it from `sktime.split`.

```python
cv = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=36, step_length=12)
```

The training window grows from `initial_window`. Each test window has one index per horizon step. `step_length` moves the origin. `SlidingWindowSplitter` keeps a fixed training length instead of expanding. Both live in `sktime.split` and are re-exported from `sktime.forecasting.model_selection` for older tutorials.

Do not shuffle the time index. For several independent series, a splitter may still be random across series only when the scientific unit of independence is the series, and the time order inside each series stays intact.

## `evaluate`

```python
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
)

scores = evaluate(
    forecaster,
    cv,
    y,
    scoring=[MeanAbsoluteError(), MeanAbsolutePercentageError(symmetric=False)],
)
```

With `NaiveForecaster(strategy="last", sp=12)`, `ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=36, step_length=12)`, and `load_airline()`, the result has one row per origin. Columns include `test_MeanAbsoluteError`, `test_MeanAbsolutePercentageError`, `fit_time`, `pred_time`, `len_train_window`, and `cutoff`.

`strategy="refit"` (the default) refits on every training window. `"update"` and `"no-update_params"` pass new data through `update` instead. Use `"refit"` unless you mean to test an online update.

`MeanAbsoluteError` is in the units of `y`. Lower is better. Call it as `metric(y_true, y_pred)`.

`MeanAbsolutePercentageError(symmetric=False)` divides by `y_true` (MAPE). `symmetric=True` is sMAPE, with denominator `|y_true| + |y_pred|`, and the result is at most 2. The default is `symmetric=False`. MAPE is unstable when `y_true` is near zero; sMAPE hides the scale. Report the denominator you used.

Forecasting metrics are the wrong tool for a classifier's per-series scalar target. Use a classification score there.

## ForecastingBenchmark

```python
from sktime.benchmarking.forecasting import ForecastingBenchmark

benchmark = ForecastingBenchmark()
benchmark.add_task(load_airline, cv, [MeanAbsoluteError()], task_id="airline")
benchmark.add_estimator(forecaster, estimator_id="snaive")
results = benchmark.run()
```

A task is those three arguments: the data, the splitter, and the scorers. The horizon is `fh` on the splitter, so a different horizon is a different task. `run` returns one row per task and estimator. Metric columns are `{MetricName}_mean`, `{MetricName}_std`, and `{MetricName}_fold_{i}_test`, plus `fit_time_*`, `pred_time_*`, and `runtime_secs`. They are not named `test_MeanAbsoluteError`. That name belongs to `evaluate`. The parts of a task, including classification and regression, are in `references/benchmarking.md`.

An airline task with `initial_window=36`, `step_length=48`, `fh=[1, 2, 3]`, and a seasonal-naive estimator fills `MeanAbsoluteError_mean`. Quote the per-fold columns when a single mean hides a bad origin. The full runner, including classification, regression, and the significance analyzers, is in `references/benchmarking.md`.

`cv_global` switches on global, panel-style benchmarking. Leave it unset for a single series. When you set it, also read `cv_global_temporal`: without a temporal holdout, a global model can train on the future of the series it is then scored on.

## Reporting

State the splitter, the horizon, the metric (including `symmetric` for percentage errors), the sktime version, and the naive baseline in the same table as the candidate model. A lower error on one origin is not a model ranking. Use the fold column from `evaluate`, or a test that respects dependent errors, when you compare two forecasters.
