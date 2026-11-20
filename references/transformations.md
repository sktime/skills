# Transformations

Transformers change a series and, when they support it, invert the change. Fit them on the training window. This reference targets sktime 1.2.0. Call `deps` on the transformer constructor and install the returned requirements.

## Deseasonalize and detrend

```python
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.detrend import Deseasonalizer, Detrender

yt = Deseasonalizer(sp=12, model="additive").fit_transform(y)
resid = Detrender(PolynomialTrendForecaster(degree=1)).fit_transform(y)
```

- `Deseasonalizer` estimates a seasonal component of period `sp` with `statsmodels` and subtracts it (`model="additive"`) or divides by it (`model="multiplicative"`). `inverse_transform` puts the component back. `sp=1` is a no-op period.
- `Detrender` fits a forecaster and returns residuals. With the default forecaster, or with `PolynomialTrendForecaster(degree=1)`, that is a linear trend. Residuals are `y - fitted` for the additive model.
- `ConditionalDeseasonalizer` and `STLTransformer` live in the same module. STL needs a long enough series for the seasonal window; check the statsmodels error rather than padding silently.

Inside a forecaster, chain with `*` so the inverse runs at predict time:

```python
from sktime.forecasting.naive import NaiveForecaster

forecaster = (
    Deseasonalizer(model="multiplicative", sp=12)
    * Detrender(PolynomialTrendForecaster(degree=1))
    * NaiveForecaster(strategy="last")
)
```

The product is a `TransformedTargetForecaster`. Multiplicative deseasonalizing requires strictly positive values.

## Differencing

```python
from sktime.transformations.series.difference import Differencer

diff = Differencer(lags=1).fit_transform(y)
seasonal = Differencer(lags=[1, 12]).fit_transform(y)
```

`lags=1` is a first difference. `lags=[1, 12]` differences at lag 1 and then at lag 12. Repeating a lag (`lags=[1, 1]`) is a second difference.

`na_handling` defaults to `"fill_zero"`. On airline data the differenced series kept length 144 and the first value was 0. `"drop_na"` shortens the series. `"keep_na"` leaves a leading NaN. A model that cannot accept NaN or a rewritten index will disagree with `"fill_zero"` about the first observation; choose the handling explicitly.

## Imputation

```python
from sktime.transformations.series.impute import Imputer

filled = Imputer(method="linear").fit_transform(y)
```

`method` includes `"drift"`, `"linear"`, `"nearest"`, `"constant"`, `"mean"`, `"median"`, `"bfill"`, and `"ffill"`.

- `"linear"` interpolates inside the series passed to `transform`. It does not extrapolate; remaining edges are filled by forward then back fill.
- `"mean"` and `"median"` use the values seen in `fit`. To compute them on the transform batch instead, wrap the imputer so that `fit` sees that batch (`FitInTransform` in the compose module).

Fit the imputer on the training window when the filled values are inputs to a model that is then scored on a later window. Computing a mean on the full series, including the test period, leaks that period into the features.

## Feature extraction

`Catch22`, `TSFresh`, and `SummaryTransformer` turn a panel into a tabular feature matrix that a scikit-learn model can use. Call `deps` on the constructor and install the returned requirements before `fit`. Prefer a pipeline (`features * sklearn_estimator` via `sktime.pipeline.make_pipeline`) so the feature fit stays inside the training fold.
