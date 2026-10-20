# Comparing Machine-Learning Methods for Water-Table Depth

## Project purpose

This project explores how different machine-learning choices affect estimates of groundwater depth to water (WTD) in a selected study area in the United States. It is designed as a transparent starting point for hydrology and hydrogeology research: the same study area and held-out wells are used to compare models, while individual experiments change the learning algorithm, training labels, constraints, or amount of training data.

The central research question is not simply **“Which model has the lowest error?”** It is also:

- How much can terrain and proximity-to-water variables explain observed water-table depth?
- Does adding shoreline-based pseudo-observations change model performance or spatial patterns?
- Do Random Forest and XGBoost learn different relationships from the same predictors and labels?
- Do monotonic constraints make predictions more consistent with selected physical expectations?
- How sensitive are results to having fewer training examples?
- Do different models produce similar maps, or do they infer substantially different spatial patterns?
- Where are errors concentrated, and what does that suggest about data coverage, missing predictors, or local hydrogeology?

The code was adapted from a small-area, classroom-scale workflow inspired by published work on combining groundwater wells and surface-water proxies. It is now organized as a standalone research script rather than a course assignment. The workflow is intentionally modest in scale; it is not a validated national groundwater model.

## Model comparison

All models use a shared train/test split of the available well observations. The test wells are held out from model fitting and are used to compare predictions against observed WTD. The default split is random and reproducible (`RANDOM_SEED = 42`).

| Label | Method | Training data | What the comparison tests |
|---|---|---|---|
| **A** | Random Forest regression | Training wells only | Baseline performance using measured well data |
| **B** | Random Forest regression | Training wells + shoreline pseudo-observations | Effect of adding the surface-water proxy labels while retaining the same algorithm |
| **C** | XGBoost regression | Training wells + shoreline pseudo-observations | Difference between two tree-ensemble methods under the same broad training-data setup |
| **D** | Monotonic XGBoost regression | Same as C, with monotonic constraints | Effect of constraining selected predictor-response directions |
| **E** | Linear regression using TWI only | Training wells only | Simple benchmark for a single-predictor linear relationship |
| **F** | Monotonic XGBoost regression | Half of the combined training table used by D | Sensitivity to reduced training-data availability |

Model F samples half of the combined well-plus-proxy training table, so it does **not** necessarily represent half of the measured wells. This distinction matters when interpreting the stress test.

## Workflow: what happens and why

### 1. Select a study area

The study area is a geographic bounding box in longitude/latitude. The default box is approximately one degree across. Edit `LON_MIN`, `LON_MAX`, `LAT_MIN`, and `LAT_MAX` near the top of `compare_models.py`.

Choose an area with enough groundwater wells for a meaningful train/test comparison. A small box may return too few wells, and some wells may be removed because of aquifer filters or missing raster values. The script prints counts so the amount of usable data is visible.

### 2. Retrieve groundwater-level observations

The script queries USGS Water Data for well locations and water-level measurements (parameter code `72019`). It converts the reported water-level values to numeric depth-to-water values in feet and keeps the minimum recorded value for each monitoring location.

The workflow attempts to reduce the influence of wells in confined aquifers by removing records coded as confined/mixed (`C`/`M`) and wells whose construction depth exceeds `CONFINED_DEPTH_CUTOFF_FT`, where the corresponding metadata are available.

These are practical filters, not a definitive hydrogeologic classification. Missing aquifer metadata does not mean a well is unconfined, and the minimum observed depth is not guaranteed to represent an undisturbed or natural water table. For research publication, inspect the USGS fields, data quality, measurement dates, and assumptions in the selected region.

### 3. Build terrain and surface-water predictors

The script retrieves a digital elevation model (DEM) from USGS 3DEP via `py3dep`, then uses WhiteboxTools to derive:

- **Elevation** — land-surface elevation.
- **Slope** — slope in degrees.
- **Topographic Wetness Index (TWI)** — a terrain-based index often used to describe relative wetness potential.
- **Height Above Nearest Drainage (HAND)** — relative height above the derived drainage network.
- **Distance to mapped waterbodies** — Euclidean raster distance to mapped NHD waterbody cells.
- **Distance to mapped streams** — Euclidean raster distance to mapped NHD flowline cells.

The terrain calculations also create intermediate rasters in the output directory. These files can be useful when checking the workflow.

Distances are calculated using `CELL_SIZE_M = 30`. This is an approximation: the actual grid transform and projection should be checked for each study area, especially if the workflow is moved to a region or grid with different cell dimensions.

### 4. Match wells to raster cells

Each well coordinate is transformed into the DEM coordinate reference system and matched to the corresponding raster cell. Wells outside the raster or with missing predictor values are dropped. The prepared table is saved as `prepared_well_data.csv`.

This step connects measured groundwater conditions to the terrain features the models can learn from. It also introduces uncertainty: raster resolution, coordinate transforms, geolocation error, and raster NoData can affect the features assigned to each well.

### 5. Create shoreline pseudo-observations

The mapped waterbody mask is dilated to identify cells near waterbody edges. A sample of those shoreline cells is assigned a target depth-to-water value of **0 ft** and added to the training data for models B, C, D, and F.

These rows are **pseudo-observations, not measured groundwater levels**. They encode an assumption that groundwater is at land surface near mapped waterbody shorelines. The assumption may be reasonable in some settings but wrong in others, and the proxy labels can strongly influence predictions if they are numerous relative to real wells. Model A and model E do not use these proxy labels.

The script records the number of proxy rows retained. A useful next experiment is to vary the number of proxies, shoreline buffer width, and assumed target value, then see whether results are robust.

### 6. Fit and compare models

The same well train/test split is used for all models, and each model is evaluated against the same held-out well observations. The primary metrics are:

- **MAE (ft)** — mean absolute error; lower is better.
- **RMSE (ft)** — root mean squared error; penalizes large errors more strongly.
- **R²** — coefficient of determination; can be negative when predictions are worse than the test-set mean benchmark.
- **Pearson correlation (r)** — linear association between observed and predicted values. High correlation does not necessarily mean small errors or unbiased predictions.

The current model settings are simple, fixed comparison settings, not a hyperparameter optimization study. Therefore, the results compare these particular configurations, not the maximum possible performance of each algorithm.

### 7. Inspect the outputs, not just the score

The script creates:

- **Observed-versus-predicted scatter plots** — show bias, spread, and performance across the range of observed depths.
- **Predicted WTD maps** — show spatial patterns across the raster domain, using a common color scale to support comparison.
- **Spatial prediction correlation matrix** — measures how similar the models' maps are across cells where predictions are available.
- **Baseline residual table** — identifies the largest errors for Model A without removing those wells from evaluation.
- **TWI monotonicity stress test** — increases TWI uniformly and checks how often predicted WTD increases.
- **CSV tables** — export metrics, held-out predictions, residuals, spatial correlations, and the monotonicity check.

The residual diagnostic is deliberately descriptive: all held-out wells remain in the reported evaluation. Removing difficult test observations after inspecting model errors would bias the evaluation and could make performance look artificially better.

## Monotonic constraints: what they mean

For Model D and Model F, XGBoost receives a constraint for each predictor, in the same order as `FEATURES`:

```python
MONOTONE_CONSTRAINTS = (1, 0, -1, 1, 1, 1)
```

In this configuration, increasing elevation is constrained not to decrease predicted WTD; slope has no imposed direction; increasing TWI is constrained not to increase predicted WTD; and increasing distance to water, distance to stream, or HAND is constrained not to decrease predicted WTD.

These directions are hypotheses chosen for this experiment. They should not be assumed to hold universally. The TWI perturbation test checks whether the fitted model follows the specified TWI direction; it does not establish that the constraint is physically correct for the study area.

## How to run

### Requirements

Use Python 3.10 or newer in an environment with the packages listed in `requirements.txt`. WhiteboxTools and the data-access packages may need additional setup depending on your operating system.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run from the repository directory:

```bash
python compare_models.py --output-dir results
```

The script downloads current data from external services, so an internet connection is required. Some downloads or terrain-processing steps may take time. The output directory will contain intermediate rasters and model outputs.

For a Colab run, upload `compare_models.py` and `requirements.txt`, install the requirements in a notebook cell, then run:

```python
!python compare_models.py --output-dir results
```

If a package is already installed, the install command will generally leave it available; however, dependencies and external services can change, so record the environment used for any published analysis.

## Output files

| File | Contents |
|---|---|
| `prepared_well_data.csv` | Wells retained after filtering and predictor extraction |
| `model_metrics.csv` | Held-out MAE, RMSE, R², and Pearson correlation for models A–F |
| `heldout_well_predictions.csv` | Observed and predicted WTD, plus residuals, for every model |
| `baseline_test_residuals.csv` | Model A test residuals sorted by absolute error |
| `spatial_prediction_correlations.csv` | Correlation matrix for full-grid predictions |
| `twi_monotonicity_test.csv` | Fraction of cells where a TWI increase raised predicted WTD |
| `input_data_qc.png` | Initial terrain/well/waterbody quality-control figure |
| `observed_vs_predicted.png` | Scatter plots for held-out observations |
| `predicted_wtd_maps.png` | Six full-grid model prediction maps |
| `spatial_prediction_correlations.png` | Heatmap of spatial prediction similarity |
| `dem*.tif`, `slope.tif`, `twi.tif`, etc. | Intermediate terrain rasters used by the workflow |

## Important limitations and research cautions

1. **Random split versus spatial validation.** Nearby wells can share hydrogeologic conditions. A random split may put nearby wells in both training and test data and overstate performance for new locations. Use spatial blocking or leave-location-out validation for stronger generalization claims.
2. **Small or uneven well samples.** Results depend on the number, distribution, depth range, and quality of wells. A region with sparse or clustered wells may produce unstable or spatially biased maps.
3. **Target definition and units.** WTD is represented as feet below land surface. Confirm that the selected USGS parameter and units are consistent for the queried records before interpreting results.
4. **Confined-well screening.** Construction depth is not the same as the depth of the screened interval or a reliable confined/unconfined label. The current rule is only an approximation.
5. **Pseudo-label assumptions.** Shoreline WTD = 0 ft is a modeling assumption. Test its influence rather than treating proxy labels as observations.
6. **Feature completeness.** The predictors are primarily topographic and distance-based. They omit many potentially important controls, including geology, soil properties, pumping, recharge, climate, aquifer thickness, and human water management.
7. **Map interpretation.** Full-grid predictions can look spatially detailed even where observations are sparse. The maps should be interpreted alongside data coverage, residuals, and uncertainty—not as direct measurements.
8. **Model fairness.** The algorithms have not been exhaustively tuned, and Model E intentionally uses only TWI. Differences therefore reflect both model structure and the stated predictor/training-data design.
9. **Reproducibility.** A fixed random seed supports repeatable splits and sampling, but live source data may change. Archive source data, retrieval dates, package versions, and configuration values for a publishable experiment.
10. **Spatial distance assumptions.** The code assumes a 30 m cell size for distance calculations. Verify projected cell dimensions and use a distance method suitable for the study area's CRS before interpreting distances quantitatively.

## Promising next research steps

- Replace the random split with spatial-block cross-validation and compare the results.
- Repeat the experiment across multiple study areas with different terrain, geology, well density, and hydroclimate.
- Add geologic, soil, recharge, climate, and aquifer predictors where reliable data are available.
- Compare model performance across shallow and deep WTD ranges, and inspect whether models systematically overpredict or underpredict extremes.
- Run ablation experiments: remove one feature at a time to assess the information contributed by each predictor.
- Test alternative proxy strategies, including different shoreline buffers, proxy counts, and proxy target values.
- Tune model hyperparameters using training data only, with a validation strategy that avoids test-set leakage.
- Quantify uncertainty with ensembles, quantile models, bootstrap resampling, or prediction intervals.
- Compare the predicted maps with independent datasets where possible.
- Report both error metrics and spatial behavior; one score cannot establish that a map is hydrogeologically credible.

## Data sources and citation

The script accesses data through these services and software:

- USGS Water Data / `dataretrieval` for groundwater monitoring locations and water-level measurements.
- USGS 3DEP through `py3dep` for elevation data.
- National Hydrography Dataset services through `pynhd` for mapped waterbodies and flowlines.
- WhiteboxTools for terrain processing.
- scikit-learn and XGBoost for regression models.

For public research, cite the datasets, software, and scientific methods actually used in your final analysis. Record retrieval dates and versions. Check each provider's current terms and attribution guidance.

## Repository status

This is a research prototype and educational comparison workflow, not a production groundwater model. Validate API behavior, data assumptions, spatial validation, and scientific interpretation before using its results in a publication or decision-making context.
