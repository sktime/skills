# Forecasting

sktime forecasters predict at a `ForecastingHorizon` relative to a cutoff. This reference targets sktime 1.2.0. Statistical, deep-learning, and foundation-model forecasters share the interface. Call `deps` on the constructor spec and install the returned requirements before `fit`.

## Horizon and cutoff

```python
from sktime.forecasting.base import ForecastingHorizon

fh_rel = ForecastingHorizon([1, 2, 3], is_relative=True)
# fh=[1, 2, 3] passed to predict is the same relative horizon
```

- Relative step `1` is the first index after the cutoff, `2` the second, and so on.
- An absolute horizon is built from timestamps that exist in the series frequency, usually `y_test.index` after `temporal_train_test_split`.
- `fit(..., fh=fh)` stores the horizon. A later `predict()` with no arguments reuses it.
- `predict(fh=[1, 2, 3])` returns a series of length 3, one value per step, indexed in the series frequency.

`temporal_train_test_split(y, test_size=12)` on `load_airline()` holds out the last 12 months. Build the absolute horizon from `y_test.index` so the prediction index matches the test index.

## NaiveForecaster

`from sktime.forecasting.naive import NaiveForecaster`

| `strategy` | What is extrapolated | NaN behaviour |
| --- | --- | --- |
| `"last"` | Last non-missing value, or the last value of each season when `sp > 1` | Robust to NaN |
| `"mean"` | Mean of the window (`window_length=None` uses the whole series) | Robust to NaN |
| `"drift"` | Line through the first and last window points | Not robust to NaN |

There is no `"seasonal_last"` strategy. Seasonality is `strategy="last"` plus `sp`. For monthly airline data, `sp=12`. After `fit` on all but the last 12 months, `predict` of those 12 months equals `y_train.iloc[-12:]`. The first three relative steps equal `y_train.iloc[-12:-9]`.

`capability:exogenous` is false, and the tag `capability:multivariate` is false on the univariate implementation. Multivariate columns are handled by an internal column ensemble when the input is multivariate; pass a `DataFrame` only after checking the installed tags if each column must be a separate series.

`predict_interval(fh, coverage=0.9)` returns columns `(variable_name, 0.9, "lower")` and `(variable_name, 0.9, "upper")`. The interval is a residual-scale Gaussian band. It is not a calibrated split conformal interval unless you wrap the forecaster in a conformal estimator from `sktime.forecasting.conformal`.

`update(new_y)` appends observations and moves `cutoff`. Call `predict` again after `update`; the stored relative horizon is applied from the new cutoff.

## ARIMA: statsmodels and pmdarima

```python
from sktime.forecasting.arima import ARIMA, AutoARIMA, StatsModelsARIMA
```

| Class | Backend | Notes |
| --- | --- | --- |
| `StatsModelsARIMA` | `statsmodels.tsa.arima.model.ARIMA` | `order=(p, d, q)`, `seasonal_order=(P, D, Q, s)`. No `suppress_warnings` argument. |
| `ARIMA`, `AutoARIMA` | pmdarima | Same order tuples. `suppress_warnings=True` is a pmdarima argument. |

`StatsModelsARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12))` was fit on the airline training window and `predict(fh)` returned 6 finite values for a 6-month holdout. `ARIMA(order=(1, 1, 0), seasonal_order=(0, 0, 0, 0), suppress_warnings=True)` returned 3 finite relative steps. Do not pass `p=`, `d=`, and `q=` as separate constructor arguments.

Seasonal differencing needs enough history. `seasonal_order` period `12` on a series shorter than two seasons will fail or warn inside statsmodels; lengthen the training window or drop the seasonal difference.

Exogenous rows align with the target index. `fit(y_train, X=X_train)` and `predict(fh, X=X_test)` require `X_test` to contain the forecast timestamps. `temporal_train_test_split(y, X, test_size=3)` on `load_longley()` produces that alignment. The prediction index matches `y_test`.

## ETS

`from sktime.forecasting.ets import AutoETS`

- `auto=False` (the default) uses the `error`, `trend`, `seasonal`, and `sp` arguments as a fixed ETS structure. `"add"`, `"mul"`, or `None`.
- `auto=True` searches structures and ignores the manual component strings. Pass `sp` for the seasonal period.
- `n_jobs` controls the automatic search.

`AutoETS(auto=False, error="add", trend="add", seasonal="add", sp=12)` fixes the structure. `AutoETS(auto=True, sp=12, n_jobs=1)` selects one. Compare either against seasonal naive on the same horizon before treating the selected ETS as an improvement. Call `deps("AutoETS(auto=True, sp=12)")` and install the returned requirements before fitting.

## Reduction and direct strategies

`make_reduction(estimator, strategy="recursive", window_length=10)` builds lag features and calls a scikit-learn regressor.

- `"recursive"` trains one model for the one-step target and feeds predictions back. Error accumulates with the horizon.
- `"direct"` trains one model per horizon step. It needs enough rows to fill `window_length` plus the largest step.
- `"dirrec"` and `"multioutput"` are the other `strategy` values.

`make_reduction(Ridge(), strategy="recursive", window_length=5)` was fit on the Longley training split with `X_train` and predicted from `X_test`. The stored `fh` was the absolute Longley test index.

Recursive forecasts of the target use only past `y` plus exogenous rows at the target time. Lag any exogenous series yourself when the scientific lag is not contemporaneous. Do not pass future `y` inside `X`.

## Foundation models

`ChronosForecaster`, `MOIRAIForecaster`, `TimesFMForecaster`, `TiRexForecaster`, and the other checkpoint forecasters in `sktime.forecasting` use the same `fh` contract. Call `deps("ChronosForecaster()")` and install the returned requirements. Record the checkpoint name. Score them in `ForecastingBenchmark` next to a seasonal-naive baseline. A zero-shot model still needs that temporal comparison.

Sources: [forecasting API](https://www.sktime.net/docs/api-reference/forecasting/), sktime 1.2.0 `NaiveForecaster`, `StatsModelsARIMA`, `AutoETS`, and `make_reduction` docstrings.
