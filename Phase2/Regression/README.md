Artifacts and quick usage
=========================

This folder contains artifacts produced by the CS2 map probability regression pipeline.

Files of interest
- `preprocessor.pkl` : Fitted ColumnTransformer (StandardScaler for numerics + OneHotEncoder for `map`). Load with joblib and use to transform raw feature DataFrames.
- `global_calibrated_model.pkl` : Canonical copy of the global calibrated classifier. If missing, `LightGBM_calibrated.pkl` or other `*_calibrated.pkl` files are present and can be used.
- `per_map_bootstrap_results.csv` : Bootstrap results (mean_diff, ci_low, ci_high, p_value) for per-map vs global log-loss comparisons.
- `per_map_bootstrap_bar_ci.png` : Visualization showing mean_diff ± CI per map.
- `per_map_routing_recommendations*.csv` : Candidate routing recommendations and final conservative recommendations.
- `prob_regression_summary.json` : Summary of model evaluation and per-map metrics.

Quick example (python)
----------------------
import joblib
import pandas as pd

pre = joblib.load('Phase2/Regression/preprocessor.pkl')
model = joblib.load('Phase2/Regression/global_calibrated_model.pkl')
# X_raw is your DataFrame with the expected feature columns
X_proc = pre.transform(X_raw)
probs = model.predict_proba(X_proc)[:,1]

Notes
-----
- The `preprocessor.pkl` preserves OneHotEncoder categories and feature ordering. Always use it when evaluating or retraining models to ensure deterministic transforms.
- If you regenerate bootstrap results or re-train models, keep the preprocessor and global model in sync.

Contact
-------
If you want me to also save alternative canonical names or to write a small wrapper loader utility, tell me which format you prefer.
