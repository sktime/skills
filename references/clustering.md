# Time series clustering

Clustering groups series. It does not use a forecast horizon. This reference targets sktime 1.2.0. Call `deps("TimeSeriesKMeans()")` and install the returned requirements before `fit`.

## TimeSeriesKMeans

```python
from sktime.clustering.k_means import TimeSeriesKMeans

clusterer = TimeSeriesKMeans(
    n_clusters=3,
    metric="euclidean",
    averaging_method="mean",
    n_init=10,
    max_iter=300,
    random_state=0,
)
labels = clusterer.fit_predict(X)
centers = clusterer.cluster_centers_
```

Parameter names that differ from other libraries:

- The distance argument is `metric`, not `distance`. Strings include `"euclidean"`, `"dtw"`, `"erp"`, `"edr"`, `"lcss"`, `"squared"`, `"ddtw"`, `"wdtw"`, and `"wddtw"`.
- `averaging_method` is `"mean"` or `"dba"`. There is no `"ba"` alias in this class. `"dba"` is the barycenter averaging used with elastic distances.
- `init_algorithm` is `"kmeans++"`, `"random"`, `"forgy"`, or a center array of shape `(n_clusters, n_channels, series_length)`.
- `cluster_centers_` has shape `(n_clusters, n_channels, n_timepoints)`.
- `labels_` has shape `(n_instances,)`.

`X` should be a panel: `numpy3D` `(n_instances, n_channels, n_timepoints)` for equal length. Unequal-length collections need a distance that declares that capability.

`n_init` defaults to 10 and `max_iter` to 300. Set both explicitly for a smoke test, and raise them for a comparison you will report.

## Other clusterers

`sktime.clustering` also contains `TimeSeriesKMedoids`, kernel k-means, `TimeSeriesDBSCAN`, agglomerative clustering, and k-shapes. Call `all_estimators("clusterer")`, then `deps` on the constructor you select, and install the returned requirements.

## Comparing a clustering to labels

If a labeled problem is used only as a sanity check, compare assignments with a label-invariant score (adjusted Rand index, or a clustering accuracy that optimizes the label permutation). Raw accuracy against the original class ids is meaningless unless you have aligned the permutation. Do not treat a cluster id as a scientific class without an external description of the cluster.
