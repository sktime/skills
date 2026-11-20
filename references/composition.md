# Composition, reconciliation, and the registry

sktime builds estimators by arithmetic and by explicit pipeline classes. This reference targets sktime 1.2.0.

## Forecaster pipelines

The `*` operator chains transformers in application order, then a forecaster. The product of series transformers and one forecaster is a `TransformedTargetForecaster`: each transformer is fit on `y`, applied, and inverted after the inner forecaster predicts.

```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.detrend import Deseasonalizer, Detrender

forecaster = (
    Deseasonalizer(model="multiplicative", sp=12)
    * Detrender(PolynomialTrendForecaster(degree=1))
    * NaiveForecaster(strategy="last")
)
```

An equivalent explicit form is `TransformedTargetForecaster(steps=[("deseason", ...), ("detrend", ...), ("forecast", ...)])`.

`ForecastingPipeline` is the other pipeline: steps before the forecaster transform **exogenous** `X`, and the last step is the forecaster. Use it when imputation or scaling belongs on `X` rather than on `y`. `NaiveForecaster` cannot sit at the end of a pipeline that must pass `X` through, because that forecaster does not accept exogenous input.

`make_pipeline` in `sktime.pipeline` builds classifier, regressor, and transformer pipelines. A trailing scikit-learn estimator is wrapped so that a time-series feature transformer can feed it.

## Reduction

```python
from sklearn.linear_model import Ridge
from sktime.forecasting.compose import make_reduction

forecaster = make_reduction(Ridge(), strategy="recursive", window_length=5)
forecaster.fit(y_train, X=X_train, fh=fh)
y_pred = forecaster.predict(X=X_test)
```

`window_length` is the number of lags of `y`. `"recursive"` predicts one step and appends it. `"direct"` fits a separate regressor per horizon step. On the Longley split (`test_size=3`), pass `X_test` at predict time so the exogenous rows cover the stored absolute horizon.

Pick `window_length` from the dependence you care about (one seasonal cycle is a common start), and keep it inside the training backtest. A window chosen on the test period is a tuned hyperparameter.

## Hierarchical reconciliation

```python
from sktime.datasets import load_hierarchical_sales_toydata
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster

y = load_hierarchical_sales_toydata()
forecaster = ReconcilerForecaster(
    forecaster=NaiveForecaster(strategy="drift"),
    method="bu",
)
forecaster.fit(y, fh=[1, 2, 3])
y_pred = forecaster.predict()
```

`load_hierarchical_sales_toydata` returns a `DataFrame` with index names `Product line`, `Product group`, and `Date`, and a `Sales` column. The bundled frame has 240 rows at the product-group level.

`method="bu"` forecasts the bottom series and sums them to the parents. Other methods on the class are `"ols"`, `"wls_var"`, `"wls_str"`, `"mint_cov"`, `"mint_shrink"`, and `"td_fcst"`. Reconciliation inserts aggregate nodes, and the prediction row count is the number of nodes times the horizon. For this toy series and `fh=[1, 2, 3]` that is 7 nodes and 21 rows.

Sum-consistent forecasts are a constraint, not evidence that the bottom model is well specified. Check the bottom-level error as well as the reconciled totals.

## Registry

```python
from sktime.registry import all_estimators

frame = all_estimators("forecaster", as_dataframe=True)
```

`as_dataframe=True` yields columns `name` and `object`. The first argument is a scitype string: `"forecaster"`, `"classifier"`, `"regressor"`, `"clusterer"`, `"detector"`, `"transformer"`, and others listed by `all_estimators` itself when called without a filter.

Tags on an instance (`estimator.get_tags()`) answer capability questions: `capability:pred_int`, `capability:exogenous`, `capability:missing_values`, `capability:multivariate`, and `requires-fh-in-fit`. Read the tag from the installed object. A tag in an older tutorial can disagree with the version you imported.

`sktime.registry.deps` returns the soft dependencies of a constructor spec. Install that list before `fit`. `imports` returns the import block for the same spec.
