# Extension templates

sktime's third interface, after the model zoo and composition, is a fill-in template per learning task. Use one when the zoo has no estimator for a domain rule: a forecasting strategy that encodes a known schedule, a threshold detector for a lab instrument, a custom metric, a distance, or a splitter.

Templates are not installed as importable classes. Copy the file, rename the class, and fill in the `todo` comments. They live in the sktime repository at `extension_templates/` for the 1.2.0 tag:

https://github.com/sktime/sktime/tree/v1.2.0/extension_templates

The developer walkthrough is https://www.sktime.net/en/stable/developer_guide/add_estimators.html

## Pick the file

| Task | Template |
| --- | --- |
| Forecaster, smallest complete class | `forecasting_supersimple.py` |
| Forecaster, no composition | `forecasting_simple.py` |
| Forecaster, probabilistic or composite | `forecasting.py` |
| Anomaly, change point, or segment detector | `detection.py` |
| Classifier | `classification.py` |
| Early classifier | `early_classification.py` |
| Clusterer | `clustering.py` |
| Transformer | `transformer_supersimple.py`, `transformer_simple.py`, or `transformer.py` |
| Pairwise distance, tabular or panel | `dist_kern_tab.py`, `dist_kern_panel.py` |
| Forecasting metric, including hierarchical | `metric_forecasting.py`, `metric_forecasting_hierarchical.py` |
| Detection metric | `metric_detection.py` |
| Splitter, aligner, parameter estimator, catalogue | `split.py`, `alignment.py`, `param_est.py`, `catalogue.py` |

The 1.2.0 tree has no `regression.py`. A series-to-scalar regressor implements `sktime.regression.base.BaseRegressor`. Follow the same steps as `classification.py`: inner data type, `_fit`, `_predict`, tags, and `get_test_params`.

## Fill-in rules

The template states these constraints:

- Implement the mandatory private methods. For a forecaster those are `_fit(self, y, X=None, fh=None)` and `_predict(self, fh=None, X=None)`. For a detector they are `_fit(self, X, y=None)` and `_predict(self, X)`.
- Do not write the reserved attributes (`is_fitted`, `_is_fitted`, `cutoff`, `_fh`, `_tags`, and the others named at the top of the file).
- Prefix extra helpers with `_custom` so they do not override `BaseEstimator` internals.
- Set `_tags` for the capabilities you actually implement: multivariate input, missing values, prediction intervals, exogenous variables, and `python_dependencies` for any package the new class imports.
- Replace the placeholder docstring and parameter list.
- Implement `get_test_params` so `check_estimator` can build a small instance.

```python
from sktime.utils.estimator_checks import check_estimator

check_estimator(MyForecaster, raise_exceptions=True)
```

A local class that passes `check_estimator` can stay in the project as a library. Contributing it to sktime follows the developer guide and uses the repository copyright header.

## Dependencies of a custom estimator

Set the tag to the PEP 440 strings the class needs:

```python
_tags = {
    "python_dependencies": ["some-package>=1.2"],
}
```

Once the class is importable, callers install its soft dependencies with `deps`, the same way as for a zoo estimator. A class that exists only in the working file is invisible to `deps` until it can be imported by name.

## After the template

- A forecasting strategy still has to honor `fh` and the cutoff. Predict every requested step, and do not read `y` past the cutoff.
- A detector returns the sparse point table the base class expects. Map integer locations back to the time index when you report them.
- A metric receives aligned `y_true` and `y_pred`. State whether lower is better.
- Run the new estimator through `ForecastingBenchmark` or the matching benchmark class against a naive or dummy baseline before treating it as an improvement.
