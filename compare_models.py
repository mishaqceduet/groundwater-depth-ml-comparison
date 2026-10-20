#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surface-informed machine-learning comparison for water-table depth.

This research workflow compares several models for estimating depth to water
(WTD) from terrain and proximity-to-water features in a selected US study area.

Workflow
--------
1. Download groundwater-level observations from USGS Water Data.
2. Download a DEM and mapped waterbodies/streams for the study area.
3. Derive terrain predictors (slope, topographic wetness index, HAND) and
   distances to mapped surface-water features.
4. Match well observations to raster predictors and apply transparent filters.
5. Train six model configurations on a shared train/test split.
6. Evaluate held-out wells, inspect residuals, compare spatial predictions,
   and run data-availability and monotonicity stress tests.

Important scientific caveats
----------------------------
- The well-depth and aquifer filters are approximations, not a definitive
  hydrogeologic classification.
- Shoreline pixels are assigned WTD = 0 as a proxy assumption. This is not a
  direct groundwater measurement and may introduce bias.
- Random train/test splits can be optimistic when nearby wells are spatially
  autocorrelated. Spatially blocked validation is recommended for research use.
- Monotonic constraints encode hypotheses about predictor-response directions;
  they should be justified for each study area and tested, not treated as truth.
- Predictions are extrapolations from available wells and proxy data. Maps are
  model outputs, not validated groundwater observations.

The script downloads live data and may require network access. Review data
providers' terms and cite the underlying datasets and methods in publications.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.features
import xgboost as xgb
from dataretrieval import waterdata as wd
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import py3dep
import pynhd
import whitebox


# =============================================================================
# 1. Study configuration
# =============================================================================
# Edit these four coordinates to select a different study area. Coordinates
# are geographic (longitude/latitude, EPSG:4326), ordered west, south, east,
# north. Start with a region that contains enough usable groundwater wells.
LON_MIN = -99.6
LON_MAX = -98.6
LAT_MIN = 46.85
LAT_MAX = 47.85
BBOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

# Data and modeling settings. These are explicit assumptions to make runs
# reproducible and easy to adapt; they are not universal hydrogeologic values.
CONFINED_DEPTH_CUTOFF_FT = 100.0
STREAM_THRESHOLD_CELLS = 1000
DEM_RESOLUTION_M = 30
CELL_SIZE_M = 30.0
TEST_SIZE = 1 / 3
RANDOM_SEED = 42
N_TREES = 300
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.1
N_SHORELINE_PROXIES = 300
SHORELINE_DILATION_CELLS = 2
TWI_PERTURBATION = 2.0

FEATURES = [
    "elevation",
    "slope",
    "twi",
    "dist_to_water",
    "dist_to_stream",
    "hand",
]

# Constraint order must exactly match FEATURES:
# +1: prediction must not decrease as feature increases;
# -1: prediction must not increase as feature increases;
#  0: no monotonic direction imposed.
# These are hypotheses: elevation (+), TWI (-), and distance/HAND (+) are
# specified here, while slope is unconstrained. Revisit these assumptions for
# every new region and target definition.
MONOTONE_CONSTRAINTS = (1, 0, -1, 1, 1, 1)


# =============================================================================
# 2. Download and prepare source data
# =============================================================================
def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Check that bbox is ordered (west, south, east, north)."""
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(
            f"Invalid BBOX {bbox}; expected (west, south, east, north)."
        )


def _geometry_lon_lat(geometry):
    """Extract longitude/latitude from a point-like geometry or coordinate pair."""
    if hasattr(geometry, "x") and hasattr(geometry, "y"):
        return float(geometry.x), float(geometry.y)
    return float(geometry[0]), float(geometry[1])


def get_wells_live(bbox):
    """Fetch USGS water-level measurements and return one record per well.

    The target `wtd_ft` is depth to water in feet below land surface. For each
    monitoring location, this workflow keeps the minimum reported measurement,
    following the source notebook's attempt to reduce pumping-related effects.
    It then removes wells explicitly coded as confined (C/M) and wells with
    construction depth greater than CONFINED_DEPTH_CUTOFF_FT, when those fields
    are available from the API.

    These filters are approximate. Missing metadata does not prove a well is
    unconfined, and minimum recorded depth is not guaranteed to be natural.
    """
    bbox_list = list(bbox)
    sites, _ = wd.get_monitoring_locations(site_type="Well", bbox=bbox_list)
    gwl, _ = wd.get_field_measurements(bbox=bbox_list, parameter_code="72019")

    if gwl.empty:
        raise RuntimeError("USGS returned no water-level measurements for BBOX.")
    required_gwl = {"monitoring_location_id", "value"}
    if not required_gwl.issubset(gwl.columns):
        raise RuntimeError(
            f"Unexpected USGS water-level fields; missing {required_gwl - set(gwl.columns)}"
        )
    if "monitoring_location_id" not in sites or "geometry" not in sites:
        raise RuntimeError("USGS site response lacks monitoring_location_id or geometry.")

    gwl["wtd_ft"] = pd.to_numeric(gwl["value"], errors="coerce")
    per_site = (
        gwl.dropna(subset=["wtd_ft"])
        .groupby("monitoring_location_id", as_index=False)["wtd_ft"]
        .min()
    )

    aquifer_col = "aquifer_type_code" if "aquifer_type_code" in sites.columns else None
    depth_col = "well_constructed_depth" if "well_constructed_depth" in sites.columns else None
    site_cols = ["monitoring_location_id", "geometry"]
    site_cols += [c for c in (aquifer_col, depth_col) if c is not None]
    wells = per_site.merge(sites[site_cols], on="monitoring_location_id", how="inner")

    drop_mask = pd.Series(False, index=wells.index)
    if aquifer_col:
        # Codes C and M are treated as likely confined/mixed for this workflow.
        drop_mask |= wells[aquifer_col].isin(["C", "M"])
    else:
        warnings.warn("Aquifer type is unavailable; aquifer-code filter was skipped.")
    if depth_col:
        construction_depth = pd.to_numeric(wells[depth_col], errors="coerce")
        drop_mask |= construction_depth > CONFINED_DEPTH_CUTOFF_FT
    else:
        warnings.warn("Well construction depth is unavailable; depth filter was skipped.")

    print(
        f"Well filter: {len(wells)} candidates; {int(drop_mask.sum())} removed; "
        f"{int((~drop_mask).sum())} retained before coordinate/raster checks."
    )
    wells = wells.loc[~drop_mask].copy()

    coordinates = wells["geometry"].apply(_geometry_lon_lat)
    wells["lon"] = coordinates.apply(lambda pair: pair[0])
    wells["lat"] = coordinates.apply(lambda pair: pair[1])
    wells = wells[["monitoring_location_id", "lon", "lat", "wtd_ft"]].dropna()
    wells = wells.reset_index(drop=True)

    if wells.empty:
        raise RuntimeError(
            "No wells remain after filtering. Choose a different BBOX or review "
            "the filter settings and available USGS metadata."
        )
    return wells


def _read_raster_masked(path):
    """Read a raster band and convert its declared NoData value to NaN."""
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        if src.nodata is not None:
            array[array == src.nodata] = np.nan
        return array, src.transform, src.crs


def get_terrain_live(bbox, output_dir: Path, resolution_m=DEM_RESOLUTION_M):
    """Download a DEM and derive elevation, slope, TWI, and HAND rasters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dem_path = output_dir / "dem.tif"
    filled_path = output_dir / "dem_filled.tif"
    slope_path = output_dir / "slope.tif"
    sca_path = output_dir / "specific_contributing_area.tif"
    twi_path = output_dir / "twi.tif"
    d8_path = output_dir / "d8_flow_accumulation.tif"
    streams_path = output_dir / "streams_dem.tif"
    hand_path = output_dir / "hand.tif"

    dem = py3dep.get_dem(list(bbox), resolution=int(resolution_m))
    dem.rio.to_raster(dem_path)

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(output_dir))
    wbt.fill_depressions(str(dem_path), str(filled_path))
    wbt.slope(str(filled_path), str(slope_path), units="degrees")
    # D-infinity flow accumulation is used for TWI in this workflow.
    wbt.d_inf_flow_accumulation(
        str(filled_path), str(sca_path), out_type="Specific Contributing Area"
    )
    wbt.wetness_index(str(sca_path), str(slope_path), str(twi_path))

    # HAND extraction uses D8 flow accumulation, so it is computed separately.
    wbt.d8_flow_accumulation(str(filled_path), str(d8_path), out_type="cells")
    wbt.extract_streams(
        str(d8_path), str(streams_path),
        threshold=STREAM_THRESHOLD_CELLS, zero_background=True,
    )
    wbt.elevation_above_stream(str(filled_path), str(streams_path), str(hand_path))

    elevation, transform, crs = _read_raster_masked(filled_path)
    slope, _, _ = _read_raster_masked(slope_path)
    twi, _, _ = _read_raster_masked(twi_path)
    hand, _, _ = _read_raster_masked(hand_path)
    return elevation, slope, twi, hand, transform, crs


def get_polygon_mask_live(layer_name, bbox, out_shape, transform, crs, all_touched=False):
    """Download a National Hydrography Dataset layer and rasterize it."""
    features = pynhd.WaterData(layer_name, crs=crs).bybox(list(bbox), box_crs=4326)
    if features.empty:
        warnings.warn(f"No features returned for hydrography layer {layer_name}.")
        return np.zeros(out_shape, dtype=bool)

    shapes = [(geometry, 1) for geometry in features.geometry if geometry is not None]
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(bool)


def build_feature_data(bbox, output_dir: Path):
    """Download inputs, derive rasters, and sample raster predictors at wells."""
    wells = get_wells_live(bbox)
    elevation, slope, twi, hand, transform, crs = get_terrain_live(bbox, output_dir)

    waterbody_mask = get_polygon_mask_live(
        "nhdwaterbody", bbox, elevation.shape, transform, crs
    )
    stream_mask = get_polygon_mask_live(
        "nhdflowline_network", bbox, elevation.shape, transform, crs, all_touched=True
    )

    # Euclidean distance in raster cells multiplied by the assumed cell size.
    # This approximates map distance; use a suitable projected CRS and verify
    # actual pixel dimensions when applying the workflow to other regions.
    dist_to_water = distance_transform_edt(~waterbody_mask) * CELL_SIZE_M
    dist_to_stream = distance_transform_edt(~stream_mask) * CELL_SIZE_M

    print(f"Raster shape: {elevation.shape}")
    print(f"Mapped waterbody fraction: {waterbody_mask.mean():.4f}")
    print(f"Mapped stream fraction: {stream_mask.mean():.4f}")
    print(f"Elevation range: {np.nanmin(elevation):.1f}–{np.nanmax(elevation):.1f} m")
    print(f"HAND range: {np.nanmin(hand):.1f}–{np.nanmax(hand):.1f} m")

    # Convert well coordinates from geographic coordinates to the DEM CRS.
    to_raster_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_raster_crs.transform(wells["lon"].to_numpy(), wells["lat"].to_numpy())
    col_float, row_float = ~transform * (x, y)
    row_idx = np.floor(row_float).astype(int)
    col_idx = np.floor(col_float).astype(int)

    # Drop wells outside the raster rather than clipping them to an edge cell.
    inside = (
        (row_idx >= 0) & (row_idx < elevation.shape[0])
        & (col_idx >= 0) & (col_idx < elevation.shape[1])
    )
    wells = wells.loc[inside].copy()
    row_idx, col_idx = row_idx[inside], col_idx[inside]
    if wells.empty:
        raise RuntimeError("No well locations fall inside the downloaded DEM grid.")

    wells["row"] = row_idx
    wells["col"] = col_idx
    wells["elevation"] = elevation[row_idx, col_idx]
    wells["slope"] = slope[row_idx, col_idx]
    wells["twi"] = twi[row_idx, col_idx]
    wells["dist_to_water"] = dist_to_water[row_idx, col_idx]
    wells["dist_to_stream"] = dist_to_stream[row_idx, col_idx]
    wells["hand"] = hand[row_idx, col_idx]

    n_before = len(wells)
    wells = wells.dropna(subset=FEATURES + ["wtd_ft"]).reset_index(drop=True)
    print(f"Wells removed because a predictor was NoData: {n_before - len(wells)}")
    if len(wells) < 6:
        raise RuntimeError(
            f"Only {len(wells)} usable wells remain; too few for a meaningful "
            "train/test comparison. Choose a region with more wells."
        )

    feature_arrays = {
        "elevation": elevation,
        "slope": slope,
        "twi": twi,
        "dist_to_water": dist_to_water,
        "dist_to_stream": dist_to_stream,
        "hand": hand,
    }
    masks = {"waterbody": waterbody_mask, "stream": stream_mask}
    return wells, feature_arrays, masks


# =============================================================================
# 3. Shared evaluation and prediction helpers
# =============================================================================
def evaluate_predictions(y_true, y_pred, model_name):
    """Return standard regression metrics for held-out observations."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        raise ValueError(f"Not enough valid predictions to evaluate {model_name}.")

    observed, predicted = y_true[valid], y_pred[valid]
    corr = pearsonr(observed, predicted).statistic if np.std(observed) and np.std(predicted) else np.nan
    row = {
        "model": model_name,
        "n_test": int(valid.sum()),
        "MAE_ft": mean_absolute_error(observed, predicted),
        "RMSE_ft": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
        "Pearson_r": corr,
    }
    print(
        f"{model_name:38s} n={row['n_test']:4d} | "
        f"MAE={row['MAE_ft']:7.2f} ft | RMSE={row['RMSE_ft']:7.2f} ft | "
        f"R²={row['R2']:6.2f} | r={row['Pearson_r']:6.2f}"
    )
    return row


def predict_full_grid(model, feature_arrays, feature_names=FEATURES):
    """Predict each raster cell where every required predictor is available."""
    shape = feature_arrays[feature_names[0]].shape
    flat = pd.DataFrame({name: feature_arrays[name].ravel() for name in feature_names})
    valid = flat.notna().all(axis=1) & np.isfinite(flat).all(axis=1)
    prediction = np.full(len(flat), np.nan, dtype=float)
    if valid.any():
        prediction[valid.to_numpy()] = model.predict(flat.loc[valid, feature_names])
    return prediction.reshape(shape)


def make_shoreline_proxy_data(feature_arrays, waterbody_mask, rng):
    """Sample near-shore raster cells and assign proxy WTD = 0 ft.

    These are pseudo-observations, not measured groundwater levels. Their
    influence depends on the mapped waterbody layer, shoreline buffer width,
    number sampled, and the assumption that WTD is zero at the shoreline.
    """
    shoreline = (
        binary_dilation(waterbody_mask, iterations=SHORELINE_DILATION_CELLS)
        & ~waterbody_mask
    )
    rows, cols = np.where(shoreline)
    if len(rows) == 0:
        warnings.warn("No shoreline cells found; proxy-trained models will use wells only.")
        return pd.DataFrame(columns=FEATURES + ["wtd_ft"])

    n_proxy = min(N_SHORELINE_PROXIES, len(rows))
    chosen = rng.choice(len(rows), size=n_proxy, replace=False)
    proxy_rows, proxy_cols = rows[chosen], cols[chosen]
    proxy = pd.DataFrame({
        name: feature_arrays[name][proxy_rows, proxy_cols] for name in FEATURES
    })
    proxy["wtd_ft"] = 0.0
    proxy = proxy.dropna(subset=FEATURES)
    print(f"Shoreline proxy samples retained: {len(proxy)} (assigned target = 0 ft)")
    return proxy


def plot_input_data(wells, feature_arrays, masks, output_dir):
    """Save a quick visual quality-control figure for terrain, wells, and water."""
    elevation = feature_arrays["elevation"]
    twi = feature_arrays["twi"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axes[0].imshow(elevation, cmap="terrain")
    axes[0].scatter(
        wells["col"], wells["row"], c=wells["wtd_ft"], cmap="viridis_r",
        edgecolor="white", s=28,
    )
    axes[0].set_title("Elevation and wells\n(color = observed WTD, ft)")
    fig.colorbar(im0, ax=axes[0], shrink=0.75, label="Elevation (m)")
    im1 = axes[1].imshow(twi, cmap="Blues")
    axes[1].set_title("Topographic wetness index (TWI)")
    fig.colorbar(im1, ax=axes[1], shrink=0.75, label="TWI")
    axes[2].imshow(masks["waterbody"], cmap="Blues")
    axes[2].set_title("Mapped waterbodies")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "input_data_qc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. Train comparable model configurations
# =============================================================================
def run_model_comparison(wells, feature_arrays, masks, output_dir):
    """Train A–F using one fixed well train/test split and compare behavior."""
    # A fixed split makes model configurations comparable on the same held-out
    # wells. It does not guarantee spatial independence between train and test.
    train_df, test_df = train_test_split(
        wells, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"\nTrain wells: {len(train_df)} | held-out test wells: {len(test_df)}")
    if len(test_df) < 2:
        raise RuntimeError("Test split has fewer than two wells.")

    rng = np.random.default_rng(RANDOM_SEED)
    proxy_df = make_shoreline_proxy_data(feature_arrays, masks["waterbody"], rng)
    train_real = train_df[FEATURES + ["wtd_ft"]].copy()
    train_with_proxy = pd.concat([train_real, proxy_df], ignore_index=True)
    if train_with_proxy.empty:
        raise RuntimeError("No usable training data remain.")

    models = {}
    test_predictions = {}
    prediction_grids = {}
    metrics = []

    # A — Random forest, well observations only.
    # This is the baseline: it measures what the selected terrain predictors
    # can learn from the real well labels without shoreline pseudo-observations.
    models["A"] = RandomForestRegressor(
        n_estimators=N_TREES, random_state=RANDOM_SEED, n_jobs=-1
    )
    models["A"].fit(train_df[FEATURES], train_df["wtd_ft"])

    # B — Random forest plus shoreline pseudo-observations.
    # A versus B isolates the effect of adding proxy labels, with the algorithm
    # and predictor set otherwise held constant.
    models["B"] = RandomForestRegressor(
        n_estimators=N_TREES, random_state=RANDOM_SEED, n_jobs=-1
    )
    models["B"].fit(train_with_proxy[FEATURES], train_with_proxy["wtd_ft"])

    # C — XGBoost plus the same well + shoreline training data.
    # B versus C compares tree-ensemble learning strategies under the same
    # training-label setup, although hyperparameters are not fully tuned.
    models["C"] = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, random_state=RANDOM_SEED,
        objective="reg:squarederror", n_jobs=-1,
    )
    models["C"].fit(train_with_proxy[FEATURES], train_with_proxy["wtd_ft"])

    # D — XGBoost with monotonic constraints.
    # C versus D tests whether imposing expected directions changes predictions
    # and held-out performance. Constraints are assumptions, not a guarantee of
    # better accuracy or physical correctness.
    models["D"] = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=RANDOM_SEED, objective="reg:squarederror", n_jobs=-1,
    )
    models["D"].fit(train_with_proxy[FEATURES], train_with_proxy["wtd_ft"])

    # E — Simple linear regression using TWI only.
    # This deliberately simple benchmark asks how much of the observed WTD
    # pattern can be represented by one linear relationship with TWI.
    models["E"] = LinearRegression()
    models["E"].fit(train_df[["twi"]], train_df["wtd_ft"])

    # F — D retrained on half of its training rows.
    # This is a sensitivity test for reduced training-data availability. Since
    # the sampled rows include proxy labels, it is half of the combined
    # well-plus-proxy training table, not necessarily half the real wells.
    half_train = train_with_proxy.sample(
        frac=0.5, random_state=RANDOM_SEED
    )
    models["F"] = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=RANDOM_SEED, objective="reg:squarederror", n_jobs=-1,
    )
    models["F"].fit(half_train[FEATURES], half_train["wtd_ft"])

    # Use the exact same held-out well rows for all six model evaluations.
    for label, model in models.items():
        if label == "E":
            test_pred = model.predict(test_df[["twi"]])
            grid = predict_full_grid(
                model, {"twi": feature_arrays["twi"]}, feature_names=["twi"]
            )
        else:
            test_pred = model.predict(test_df[FEATURES])
            grid = predict_full_grid(model, feature_arrays)
        test_predictions[label] = test_pred
        prediction_grids[label] = grid
        metrics.append(evaluate_predictions(
            test_df["wtd_ft"], test_pred,
            {
                "A": "A: Random forest, wells only",
                "B": "B: Random forest + shoreline proxies",
                "C": "C: XGBoost + shoreline proxies",
                "D": "D: Constrained XGBoost + proxies",
                "E": "E: Linear regression, TWI only",
                "F": "F: Constrained XGBoost, half training rows",
            }[label],
        ))

    metrics_df = pd.DataFrame(metrics).set_index("model")
    metrics_df.to_csv(output_dir / "model_metrics.csv")
    plot_evaluation(test_df, test_predictions, metrics_df, output_dir)
    plot_prediction_maps(prediction_grids, output_dir)
    plot_spatial_prediction_correlations(prediction_grids, output_dir)
    analyze_baseline_residuals(test_df, test_predictions["A"], output_dir)
    run_twi_monotonicity_test(
        models["C"], models["D"], feature_arrays, output_dir
    )

    # Export well-level predictions and residuals for transparent follow-up.
    prediction_table = test_df[["monitoring_location_id", "lon", "lat", "wtd_ft"]].copy()
    for label, prediction in test_predictions.items():
        prediction_table[f"pred_{label}_ft"] = prediction
        prediction_table[f"residual_{label}_ft"] = prediction - test_df["wtd_ft"].to_numpy()
    prediction_table.to_csv(output_dir / "heldout_well_predictions.csv", index=False)

    return metrics_df, prediction_grids


# =============================================================================
# 5. Diagnostics and figures
# =============================================================================
def plot_evaluation(test_df, predictions, metrics_df, output_dir):
    """Compare observed and predicted WTD on shared axes."""
    labels = {
        "A": "A: RF, wells only",
        "B": "B: RF + proxies",
        "C": "C: XGBoost + proxies",
        "D": "D: constrained XGBoost",
        "E": "E: linear, TWI only",
        "F": "F: half training rows",
    }
    observed = test_df["wtd_ft"].to_numpy()
    all_values = np.concatenate([observed] + [np.asarray(v) for v in predictions.values()])
    low, high = np.nanmin(all_values), np.nanmax(all_values)
    padding = 0.05 * (high - low if high > low else 1.0)
    limits = (low - padding, high + padding)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, labels):
        predicted = predictions[key]
        ax.plot(limits, limits, "--", color="gray", linewidth=1.3)
        ax.scatter(observed, predicted, s=30, alpha=0.8, edgecolor="white", linewidth=0.4)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[key])
        row = metrics_df.loc[metrics_df.index.str.startswith({
            "A": "A:", "B": "B:", "C": "C:", "D": "D:", "E": "E:", "F": "F:"
        }[key])].iloc[0]
        ax.text(
            0.04, 0.96,
            f"MAE={row['MAE_ft']:.1f} ft\nr={row['Pearson_r']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
        )
    for ax in axes[1, :]:
        ax.set_xlabel("Observed WTD (ft)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted WTD (ft)")
    fig.suptitle("Held-out well predictions (dashed line = perfect prediction)")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_baseline_residuals(test_df, baseline_predictions, output_dir):
    """Export and summarize the largest baseline errors without excluding them."""
    residuals = test_df[
        ["monitoring_location_id", "lon", "lat", "wtd_ft"] + FEATURES
    ].copy()
    residuals["predicted_wtd_ft"] = baseline_predictions
    residuals["residual_ft"] = residuals["predicted_wtd_ft"] - residuals["wtd_ft"]
    residuals["absolute_error_ft"] = residuals["residual_ft"].abs()
    residuals = residuals.sort_values("absolute_error_ft", ascending=False)
    residuals.to_csv(output_dir / "baseline_test_residuals.csv", index=False)

    n_test = len(residuals)
    top_n = max(1, int(np.ceil(0.2 * n_test)))
    total_abs_error = residuals["absolute_error_ft"].sum()
    top_share = (
        residuals["absolute_error_ft"].head(top_n).sum() / total_abs_error
        if total_abs_error > 0 else np.nan
    )
    print(
        f"\nBaseline residual diagnostic: the worst {top_n}/{n_test} held-out wells "
        f"account for {top_share:.1%} of total absolute error."
    )
    print("Largest baseline errors (all held-out wells remain in evaluation):")
    print(residuals.head(min(10, n_test)).round(2).to_string(index=False))


def plot_spatial_prediction_correlations(prediction_grids, output_dir):
    """Compare similarity among full-grid predictions over shared valid cells."""
    grid_table = pd.DataFrame({
        label: grid.ravel() for label, grid in prediction_grids.items()
    })
    correlation = grid_table.corr(min_periods=2)
    correlation.to_csv(output_dir / "spatial_prediction_correlations.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(correlation.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns)
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Pairwise spatial correlation of predicted WTD")
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_prediction_correlations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_maps(prediction_grids, output_dir):
    """Plot all model predictions with a common color scale."""
    labels = list(prediction_grids)
    low = min(np.nanmin(prediction_grids[key]) for key in labels)
    high = max(np.nanmax(prediction_grids[key]) for key in labels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    image = None
    for ax, label in zip(axes.flat, labels):
        image = ax.imshow(prediction_grids[label], cmap="viridis_r", vmin=low, vmax=high)
        ax.set_title({
            "A": "A: RF, wells only",
            "B": "B: RF + shoreline proxies",
            "C": "C: XGBoost + proxies",
            "D": "D: constrained XGBoost",
            "E": "E: linear, TWI only",
            "F": "F: half training rows",
        }[label])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.8, label="Predicted WTD (ft)")
    fig.suptitle("Predicted water-table depth (common color scale)")
    fig.savefig(output_dir / "predicted_wtd_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_twi_monotonicity_test(unconstrained_model, constrained_model, feature_arrays, output_dir):
    """Perturb TWI and report where predicted WTD increases.

    Because the constrained model has a -1 TWI constraint, increasing TWI
    should not increase predictions, provided the feature order and XGBoost
    constraint configuration are correct. This tests model behavior for this
    synthetic perturbation; it does not validate whether the assumed direction
    is appropriate in the real landscape.
    """
    baseline_uncon = predict_full_grid(unconstrained_model, feature_arrays)
    baseline_con = predict_full_grid(constrained_model, feature_arrays)
    wetter_features = dict(feature_arrays)
    wetter_features["twi"] = feature_arrays["twi"] + TWI_PERTURBATION
    wetter_uncon = predict_full_grid(unconstrained_model, wetter_features)
    wetter_con = predict_full_grid(constrained_model, wetter_features)

    def increase_fraction(before, after):
        valid = np.isfinite(before) & np.isfinite(after)
        if not valid.any():
            return np.nan
        return float(np.mean(after[valid] > before[valid] + 1e-10))

    result = pd.DataFrame([
        {
            "model": "C: unconstrained XGBoost",
            "fraction_cells_with_increased_WTD": increase_fraction(baseline_uncon, wetter_uncon),
        },
        {
            "model": "D: constrained XGBoost",
            "fraction_cells_with_increased_WTD": increase_fraction(baseline_con, wetter_con),
        },
    ])
    result.to_csv(output_dir / "twi_monotonicity_test.csv", index=False)
    print("\nTWI perturbation test: increase TWI everywhere by "
          f"{TWI_PERTURBATION}; fraction of valid cells with increased predicted WTD:")
    print(result.to_string(index=False, formatters={
        "fraction_cells_with_increased_WTD": lambda value: f"{value:.2%}"
    }))


# =============================================================================
# 6. Main workflow
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare ML methods for terrain-informed water-table-depth prediction."
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for downloaded rasters, metrics, predictions, and figures.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_bbox(BBOX)
    print("=" * 78)
    print("SURFACE-INFORMED WATER-TABLE DEPTH: MACHINE-LEARNING COMPARISON")
    print("=" * 78)
    print(f"Study area (west, south, east, north): {BBOX}")
    print(f"Random seed: {RANDOM_SEED} | DEM target resolution: {DEM_RESOLUTION_M} m")
    print(f"Outputs will be written to: {output_dir}")
    print("\nWorkflow: data acquisition -> terrain predictors -> well matching ->")
    print("          shared train/test split -> model comparison -> diagnostics")

    wells, feature_arrays, masks = build_feature_data(BBOX, output_dir)
    wells.to_csv(output_dir / "prepared_well_data.csv", index=False)
    plot_input_data(wells, feature_arrays, masks, output_dir)

    metrics, _ = run_model_comparison(wells, feature_arrays, masks, output_dir)
    print("\nHeld-out performance summary:")
    print(metrics.round(3).to_string())
    print(f"\nCompleted. Review the tables and figures in: {output_dir}")


if __name__ == "__main__":
    main()
