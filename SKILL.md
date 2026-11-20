---
name: sktime
description: Time series learning tasks with sktime, the vendor-neutral framework with scikit-learn style flexibility in composition and with the largest model zoo including the latest foundation models. Use for forecasting, classification, regression, clustering, anomaly or change-point detection, and fair and reproducible benchmarking across statistical, deep-learning, and foundation-model estimators.
license: BSD-3-Clause license
allowed-tools: Read Write Edit Bash
compatibility: Requires Python 3.11+ and sktime 1.2.0. Soft dependencies differ by estimator; pass the constructor spec to sktime.registry.deps and install the returned requirements before fitting. Network is needed for installation and for datasets that are not bundled.
metadata:
  version: "1.0"
  last-reviewed: "2026-11-06"
  upstream-version: "1.2.0"
  skill-author: jgyasu
---

# sktime

## Overview

sktime is the vendor-neutral framework package for time series learning tasks, and the only one that does all of the following:

1. **Largest model zoo.** The sktime model zoo is the largest available. It reaches almost the entire ecosystem of time-series algorithms: cutting edge foundation models, classical statistical models, distance and interval methods, and deep learning models, behind one `fit` / `predict` contract.
2. **scikit-learn style composition.** Allows pipelines, reduction, ensembling, column and hierarchy wrapping, and the `*` dunder to combine estimators.
3. **Modular extension.** `extension_templates` in the sktime repository are fill-in modules for a new estimator. Copy one when a domain needs its own forecasting strategy, a rule-based anomaly detector, a metric, or another learning task.
4. **Fair and reproducible benchmarking.** The sktime benchmarking framework allows comparison of estimators across shared tasks. The framework is fault-tolerant and crash-safe, allowing results to be persisited and the experiment to be resumed from the last safe state in the event of a crash. It also provides tools for post-hoc analysis.


## When to Use This Skill

Use this skill when:

- Forecasting, classifying, regressing, clustering, or detecting events in time series
- Selecting a model from the zoo
- Composing preprocessing, reduction, or reconciliation with a scikit-learn style pipeline
- Benchmarking estimators and reading rank or significance results
- Adding estimators from an extension template

## Installation

Targets **sktime 1.2.0** and requires **Python 3.11+**.

```bash
pip install "sktime==1.2.0"
```

Core dependencies are installed with sktime. Soft dependencies differ by estimator. Pass the constructor you intend to fit to `sktime.registry.deps`, and install every requirement it returns before calling `fit`. An empty list means the core install is enough.

```python
from sktime.registry import deps

deps("ChronosForecaster()")
deps("AutoETS(sp=12)")
```

Quote each returned requirement so the shell does not expand brackets:

```bash
pip install "requirement-returned-by-deps"
```

Curated extras such as `sktime[forecasting]` are a convenience subset. They do not cover the entire soft dependency set. Use `deps` for the estimator you are about to fit.

## Discovery

Search the sktime model zoo with `all_estimators`. The first argument is the scitype: `"forecaster"`, `"classifier"`, `"regressor"`, `"clusterer"`, `"detector"`, or `"transformer"`. `as_dataframe=True` returns columns `name` and `object`.

```python
from sktime.registry import all_estimators

forecasters = all_estimators("forecaster", as_dataframe=True)
interval_forecasters = all_estimators(
    "forecaster",
    as_dataframe=True,
    filter_tags={"capability:pred_int": True},
)
```

`filter_tags` keeps estimators whose tags match every key. Common capability filters are `capability:pred_int`, `capability:exogenous`, and `capability:missing_values`.

## Data loaders

Load data from a dataset class in `sktime.datasets`. Call `load` with the set names you need. One name returns that object. Several names return a tuple in the same order. `keys()` lists the names that class accepts.

Forecasting datasets provide `"y"` and `"X"`. `"X"` is `None` when the series has no covariates. They do not define a train/test split, so split on time after loading. Classification and regression datasets also provide `"X_train"`, `"y_train"`, `"X_test"`, and `"y_test"` when the problem ships one predefined split.

These classes ship with sktime and do not download:

- Forecasting: `Airline`, `Longley`, `Lynx`, `PBS`, `ShampooSales`, `USChange`. `HierarchicalSalesToydata` is imported from `sktime.datasets.forecasting`.
- Classification: `ACSF1`, `ArrowHead`, `BasicMotions`, `GunPoint`, `ItalyPowerDemand`, `JapaneseVowels`, `OSULeaf`, `PLAID`.
- Regression: `Tecator`, imported from `sktime.datasets.regression`.

```python
from sktime.datasets import Airline, ArrowHead, Longley

y = Airline().load("y")
y, X = Longley().load("y", "X")
X_train, y_train = ArrowHead().load("X_train", "y_train")
```

`ForecastingData` loads one named dataset from the Monash forecasting archive. `UCRUEADataset` loads one named problem from the UCR/UEA classification archive. Both download on first use. `UCRUEADataset.list_all()` returns the valid problem names. `return_mtype` defaults to `"pd-multiindex"`. `M5Dataset` and `Solar` also fetch remote data. `Macroeconomic` reads the series shipped with statsmodels.

```python
from sktime.datasets import ForecastingData, UCRUEADataset

y = ForecastingData(name="cif_2016_dataset").load("y")
X_train, y_train = UCRUEADataset(name="Beef").load("X_train", "y_train")
```

## Core capabilities

Pick the estimator with Discovery, install what `deps` returns, then use the contract for that task. Constructor arguments for particular classes are in the references.

### 1. Forecasting

See `references/forecasting.md`. A forecaster predicts a path after a cutoff.

```python
from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon([1, 2, 3], is_relative=True)
# or ForecastingHorizon(y_test.index, is_relative=False)

forecaster.fit(y_train, X=X_train, fh=fh)
y_pred = forecaster.predict()
forecaster.update(y_new)
```

- A relative step `1` is the first time point after the cutoff. An absolute horizon is a list of timestamps, usually `y_test.index`.
- `fit(..., fh=fh)` stores the horizon. `predict()` with no arguments reuses it. `predict(fh)` returns one value per step.
- Pass `X` only when `capability:exogenous` is true. Those rows must cover the forecast timestamps and be known when the forecast is issued.
- `predict_interval` and `predict_quantiles` are available when `capability:pred_int` is true.

Families behind that contract are statistical forecasters, reductions of a tabular regressor, deep learning, and foundation models. Record the checkpoint name when the estimator is a foundation model.

### 2. Classification and regression

See `references/classification.md` and `references/regression.md`. Both tasks score a whole series. Classification returns a label. Regression returns a scalar.

```python
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
```

Search `"classifier"` or `"regressor"`. Families include distance, interval, dictionary or kernel, deep learning, and a constant baseline.

### 3. Clustering and detection

See `references/clustering.md` and `references/detection.md`.

Clustering assigns each series a label. `fit` learns the partition and `predict` returns one label per series. Detection returns events. `fit_predict` returns a table of locations, and a score per time point is a separate call.

Search `"clusterer"` or `"detector"`. Clustering families include distance and feature-based methods. Detection covers point anomalies, change points, and segments.

### 4. Composition

See `references/composition.md`, `references/transformations.md`, and `references/distances.md`.

```python
composed = y_transformer * forecaster
```

`*` chains transformers and one forecaster into a `TransformedTargetForecaster`. `ForecastingPipeline` transforms exogenous `X` and ends in a forecaster that accepts `X`. `make_reduction` turns a scikit-learn regressor into a forecaster. `ReconcilerForecaster` makes hierarchical forecasts sum to their parents. Fit learned transformers inside the pipeline so each backtest fold refits them.

### 5. Benchmarking

See `references/benchmarking.md`. A benchmark crosses estimators with tasks. A task is the data, the splitter, and the scorers. For forecasting, the horizon is `fh` on the splitter. `dataset` and the two forecasters below are objects the caller already chose.

```python
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter

benchmark = ForecastingBenchmark()
benchmark.add_task(
    dataset,
    ExpandingWindowSplitter(fh=[1, 2, 3], initial_window=36, step_length=12),
    [MeanAbsoluteError()],
    task_id="task",
)
benchmark.add_estimator(forecaster_a, estimator_id="a")
benchmark.add_estimator(forecaster_b, estimator_id="b")
results = benchmark.run(output_file="results.csv")
```

`ForecastingBenchmark`, `ClassificationBenchmark`, and `RegressionBenchmark` share `add_task`, `add_estimator`, and `run`. `run` writes the table and still returns it. Forecasting columns are `{Metric}_mean`, `{Metric}_std`, and `{Metric}_fold_{i}_test`. `evaluate` on one forecaster is a different table: one row per fold and columns such as `test_MeanAbsoluteError`.

Analyzers in `sktime.benchmarking.analysis` consume that table, or the path written by `run`:

| Analyzer | Role |
| --- | --- |
| `AverageRank` | Mean rank of each estimator across tasks. Rank the direction the metric requires. |
| `FriedmanTest` | Omnibus test that all estimators perform equally. |
| `NemenyiTest` | Post-hoc pairwise comparison after a significant Friedman test. |
| `WilcoxonSignedRankTest`, `SignTest`, `RankSumTest`, `TwoSampleTTest` | Paired or two-sample comparisons. |
| `CriticalDifferenceDiagram` | Rank diagram with a critical-difference bar. |

```python
from sktime.benchmarking.analysis import AverageRank, FriedmanTest

ranks = AverageRank().evaluate(results)
friedman = FriedmanTest().evaluate(results)
```

Add one task per dataset. A single series cannot support a Friedman test across datasets. Keep a naive or dummy baseline in the same benchmark as the other candidates.

### 6. Extension templates

See `references/extensions.md`.

Copy the template from the [sktime `extension_templates` directory](https://github.com/sktime/sktime/tree/v1.2.0/extension_templates) that matches the learning task. `forecasting_simple.py` and `forecasting.py` are the fill-ins for a domain forecasting strategy. `detection.py` is the fill-in for a rule-based anomaly, change-point, or segment detector. Work through the `todo` comments, set the tags, and check the class with `sktime.utils.estimator_checks.check_estimator`. The template is not an importable estimator.

## Best practices

1. **Cut on time.** Use `temporal_train_test_split` or a window splitter. A random split of overlapping windows leaks the future into training.
2. **Publish the horizon.** Fit and score the same `fh` you will report. Report error at each step before averaging.
3. **Keep a naive or dummy baseline on the same task.** The reference for that task names the baseline class.
4. **Fit transformers inside the pipeline** so each backtest fold learns its own scale, seasonality, or imputation.
5. **Install what `deps` returns before `fit`.** Pass the constructor spec you will actually use.
6. **Set the seasonal period from the series frequency.** See `references/forecasting.md`.
7. **Treat intervals as model statements.** Check coverage on the benchmark before quoting a band.
8. **Record the checkpoint** when the estimator is a foundation model. The interface is the same; the weights are not part of the sktime release.

## Reference documentation

- `references/forecasting.md` — horizon, exogenous input, and the forecaster families
- `references/classification.md` — collection classifiers and the constant baseline
- `references/regression.md` — series-to-scalar regression
- `references/clustering.md` — clusterers and their metrics
- `references/detection.md` — anomaly points, change points, and scores
- `references/transformations.md` — deseasonalizing, detrending, differencing, imputation
- `references/distances.md` — panel distances
- `references/composition.md` — pipelines, reduction, reconciliation, the registry
- `references/benchmarking.md` — shared-protocol benchmarks and significance tools
- `references/extensions.md` — extension templates for a new estimator
- `references/datasets_evaluation.md` — loaders, splitters, and metrics

## Additional resources

- Documentation: https://www.sktime.net/
- API reference: https://www.sktime.net/docs/api-reference/
- Estimator search: https://www.sktime.net/models/
- Extension templates: https://github.com/sktime/sktime/tree/v1.2.0/extension_templates
- Adding an estimator: https://www.sktime.net/docs/developer-guide/add-estimators/
- Adding a dataset: https://www.sktime.net/docs/developer-guide/add-dataset/
- GitHub: https://github.com/sktime/sktime
