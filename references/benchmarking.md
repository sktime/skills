# Benchmarking

A benchmark is the cross product of two registries: estimators and tasks. Each pair is one experiment. Estimators are what you compare. A task is the protocol they all share.

This reference targets sktime 1.2.0.

## What a task is

A task is one dataset scored one way. It has three required parts:

| Part | What it fixes | What a change means |
| --- | --- | --- |
| Data | The series, panel, or hierarchy, including exogenous `X` when the loader returns it. | A different dataset is a different task. |
| Splitter | The folds. For forecasting, the horizon is `fh` on the splitter, not a separate argument of the task. | A different horizon, window, or fold count is a different task. |
| Scorers | The metrics applied to every fold of that task. | A different metric list is a different scoring of that protocol. |

`task_id` names that protocol in the result table. If you omit it, the id is `[dataset=...]_[cv_splitter=ClassName]`, and a forecasting task that sets `cv_global` appends `_[cv_global=ClassName]`. That default does not contain the horizon, the window lengths, or the metric names. Two `ExpandingWindowSplitter` tasks with different `fh` therefore collide, and the second id is rewritten with a suffix and a warning. Pass `task_id` whenever those differ.

Forecasting tasks also carry:

- `strategy`: `"refit"` (default), `"update"`, or `"no-update_params"`. `"refit"` fits again on every training window. The other two ingest the new window through `update`.
- `error_score`: `"raise"`, or a number stored when a fit fails.
- `cv_global` and `cv_global_temporal`: only for benchmarking forecasters that support global forecasting.

Classification and regression tasks are data, splitter, scorers, `task_id`, and `error_score`. They have no `strategy` and no global split.

`ForecastingBenchmark`, `ClassificationBenchmark`, and `RegressionBenchmark` each keep their own task registry. A forecasting metric cannot score a classification task, because it aligns on a time index and the classification target is one label per series.

## Registering a task

`add_task` registers one task.

```python
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.datasets import Airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
)
from sktime.split import ExpandingWindowSplitter

benchmark = ForecastingBenchmark()
cv = ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=36, step_length=12)
benchmark.add_task(
    Airline,
    cv,
    [MeanAbsoluteError(), MeanAbsolutePercentageError(symmetric=False)],
    task_id="airline-h3",
)
benchmark.add_estimator(NaiveForecaster(strategy="last", sp=12), estimator_id="snaive")
benchmark.add_estimator(NaiveForecaster(strategy="mean"), estimator_id="mean")
results = benchmark.run(output_file="forecast_benchmark.csv")
```

The same three parts can be registered separately with `add`. A dataset, a splitter, and a metric are stored by scitype, and the benchmark then builds one task for every dataset–splitter pair. Every registered metric becomes a scorer of each of those tasks. A 3-tuple is one task specification; the order does not matter, because the role comes from the scitype. The tuple must contain exactly one dataset, one metric, and one splitter.

```python
benchmark.add(Airline(), cv, MeanAbsoluteError())
```

`add_estimator` takes one estimator plus `estimator_id`, a dict of id to estimator, or a list of estimators. The id defaults to the class name. Two instances of the same class need explicit ids. `add` also accepts an estimator, a dict, or a list.

`run(output_file=None)` returns one row per task–estimator pair. A path ending in `.csv`, `.json`, or `.parquet` is also written, and the run checkpoints. `force_rerun` chooses which stored rows are recomputed. `return_data=True` on the benchmark constructor keeps predictions and ground truth in that row. `backend` is `"None"`, `"loky"`, `"multiprocessing"`, `"threading"`, `"joblib"`, `"dask"`, `"dask_lazy"`, or `"ray"`. Pass backend-specific keys through `backend_params`.

### Data

`add_task` accepts four forms. A dataset class is the form that matches the loaders in `references/datasets_evaluation.md`.

| Form | Forecasting task | Classification or regression task |
| --- | --- | --- |
| Dataset class or instance, such as `Airline` or `ArrowHead()` | `load("y")` and `load("X")`. `X` is `None` when the series has no covariates. | `load("X")` and `load("y")`. This is the full collection. The splitter cuts the folds. |
| Loader function, such as `load_airline` | Called with no arguments. A pair `(a, b)` is `y, X`. One object is `y`. | Called with no arguments. A pair `(a, b)` is `X, y`. One object is `X`. |
| Tuple of two containers | `(y, X)` | `(X, y)` |
| One container | Endogenous `y` | `X`, and `y` is missing |

The pair is read in opposite orders. `load_longley` returns `(y, X)` and belongs on a forecasting task. `load_unit_test` returns `(X, y)` and belongs on a classification or regression task.

### Scorers

Forecasting scorers are sktime forecasting metric instances. The result column uses `metric.name`, so `MeanAbsoluteError()` becomes `MeanAbsoluteError_mean`. Classification and regression scorers may be scikit-learn callables. The result column uses the callable's `__name__`.

`MeanAbsolutePercentageError(symmetric=False)` is MAPE. `symmetric=True` is sMAPE and is bounded by 2. The boolean is part of the task. Record it in `task_id`, because the default id will not.

## Forecasting versus `evaluate`

`sktime.forecasting.model_evaluation.evaluate` backtests one forecaster on one splitter. Its columns are `test_{MetricName}`, `fit_time`, `pred_time`, `len_train_window`, and `cutoff`, with one row per fold.

`ForecastingBenchmark.run` stacks every estimator and every task. Metric columns are `{MetricName}_mean`, `{MetricName}_std`, and `{MetricName}_fold_{i}_test`, plus the same pattern for `fit_time` and `pred_time`, and `runtime_secs`. There is no `test_MeanAbsoluteError` column in that table.

### Global forecasting

`cv_global` splits a panel by instance into a pretrain set and a test panel. `cv_splitter` then cuts the test panel in time, and that temporal cut still carries `fh`. The forecaster is pretrained on the pretrain set, fitted on the temporal training window, and scored on the temporal test window.

`cv_global_temporal` is a `SingleWindowSplitter` applied to the panel before the instance split. Set it when the pretrain set must not contain the future of a series that will later be scored. Leave both unset for a single series. Without the temporal holdout, a global model can train on the future of the series it is scored on.

## Classification and regression

`RegressionBenchmark` is a `ClassificationBenchmark`. Both take a splitter such as `KFold` and scorers such as `sklearn.metrics.mean_squared_error` or a classification score. Keep every series from one subject in one fold. `KFold` is appropriate only when each row is an independent series.

## Reading the table

Analyzers live in `sktime.benchmarking.analysis`. `evaluate` accepts the `DataFrame` from `run` or the path of the artifact. Ranks are computed across tasks, so one row per task is the unit. One series is one task, however many folds it has.

```python
from sktime.benchmarking.analysis import (
    AverageRank,
    CriticalDifferenceDiagram,
    FriedmanTest,
    NemenyiTest,
    WilcoxonSignedRankTest,
)

ranks = AverageRank().evaluate(results)
screen = FriedmanTest().evaluate(results)          # columns statistic, p_value
pairs = NemenyiTest().evaluate(results)            # after a significant Friedman test
wilcoxon = WilcoxonSignedRankTest().evaluate(results)
```

`AverageRank(lower_is_better=True)` gives rank 1 to the lowest score on each task. Pass `lower_is_better=False` for a score that should be maximized. The result has columns `model_id` and `rank`.

`FriedmanTest` needs more than one task. Add a task per dataset when the claim is that an estimator wins across problems.

`SignTest`, `RankSumTest`, and `TwoSampleTTest` are the other analyzers. A paired t-test on overlapping forecast origins treats dependent errors as independent. Prefer the rank tests for forecast errors, and state the dependence if you report a t-test.

## What to publish with a benchmark

- sktime version and the requirements `deps` returned for each estimator
- each `task_id`, and the splitter arguments the id does not store: horizon, window, step, and `symmetric` on a percentage error
- the naive or dummy baseline registered as an estimator on those same tasks
- per-fold columns when a mean hides one bad origin
- checkpoint identity for any foundation model
- the analyzer name and its p-value or ranks, with the number of tasks

`error_score="raise"` stops the run on the first failed fit. A numeric `error_score` fills that cell and lets the table finish. Report those cells. Dropping a failed model and ranking the rest is a different experiment.
