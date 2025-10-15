This folder contains a compiled diagnostic report for discussion with your professor.

Included files:
- prob_regression_summary.json (regression metrics)
- results_summary_classification.json (classification metrics)
- LogisticRegression_calibration.png (regression)
- LightGBM_calibration.png (regression)
- MLP_calibration.png (regression)
- per_map_bootstrap_bar_ci.png
- combined_reliability.png (generated: regression + classification side-by-side)
- diagnostics_table.csv (summary table of key metrics for classification and regression models)

How to regenerate combined plot:
- Run the script `Phase2/Regression/cs2_map_probability_regression.ipynb` cells for plotting or use Python to load the pngs and combine them.

Notes:
- The classification results file is copied and renamed to `results_summary_classification.json` for convenience.
