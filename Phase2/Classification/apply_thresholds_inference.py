"""Inference helper updated to support Phase2 artifact layout and per-map focal models.

This exposes a main() function and can be run from the workspace root. It will:
 - locate the preprocessor and global model (search Phase2/models then models)
 - prefer per-map focal calibrated models (per_map_*_focal_calibrated.pkl), then per_map_*_calibrated, then global
 - load per-map thresholds if available (Phase2/per_map_threshold_report.json or per_map_threshold_report.json)
 - read a sample CSV and produce inference_with_thresholds_output.csv
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Optional


def find_artifact(filename_candidates):
    """Search for filename candidates in likely directories and return the first existing path or None."""
    roots = ['Phase2/models', 'models', '.']
    for root in roots:
        for candidate in filename_candidates:
            path = os.path.join(root, candidate)
            if os.path.exists(path):
                return path
    return None


def load_models_and_thresholds():
    # locate preprocessor and global model
    preproc_path = find_artifact(['preprocessor.pkl', 'models/preprocessor.pkl'])
    # prefer Phase2 global model paths
    global_model_path = find_artifact(['lightgbm_model.pkl', 'lightgbm_group_calibrated.pkl', 'xgboost_model.pkl'])

    preprocessor = joblib.load(preproc_path) if preproc_path and os.path.exists(preproc_path) else None
    global_model = joblib.load(global_model_path) if global_model_path and os.path.exists(global_model_path) else None

    # load per-map models: prefer focal calibrated, then calibrated, then tuned
    per_map_models = {}
    search_dirs = ['Phase2/models', 'models']
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith('per_map_') and fname.endswith('_focal_calibrated.pkl'):
                mapkey = fname.replace('per_map_','').replace('_focal_calibrated.pkl','').replace('_',' ').title()
                per_map_models[mapkey] = joblib.load(os.path.join(d, fname))
    # fallback: load non-focal calibrated per-map models if not present
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith('per_map_') and fname.endswith('_calibrated.pkl'):
                mapkey = fname.replace('per_map_','').replace('_calibrated.pkl','').replace('_',' ').title()
                if mapkey not in per_map_models:
                    per_map_models[mapkey] = joblib.load(os.path.join(d, fname))

    # thresholds
    thr_path = None
    # prefer Phase2 threshold report
    if os.path.exists('Phase2/per_map_threshold_report.json'):
        thr_path = 'Phase2/per_map_threshold_report.json'
    elif os.path.exists('per_map_threshold_report.json'):
        thr_path = 'per_map_threshold_report.json'
    thresholds = json.load(open(thr_path)) if thr_path else {}

    return preprocessor, global_model, per_map_models, thresholds


def infer(sample_input: str = 'sample_inference_rows.csv', output_path: str = 'inference_with_thresholds_output.csv') -> None:
    if not os.path.exists(sample_input):
        raise FileNotFoundError(f"Sample input not found: {sample_input}")

    preprocessor, global_model, per_map_models, thresholds = load_models_and_thresholds()
    if preprocessor is None:
        raise RuntimeError('Preprocessor artifact not found in Phase2/models or models')

    raw = pd.read_csv(sample_input)
    X_proc = preprocessor.transform(raw)

    preds = []
    probs = []

    # For each row, route to per-map model if available else global, and apply per-map threshold if present
    for i, row in raw.iterrows():
        m = row['map']
        map_key = str(m).title()
        model = per_map_models.get(map_key, global_model)
        if model is None:
            raise RuntimeError('No model available (global or per-map)')

        # prepare a per-row processed vector and adjust its dimension to the model's expectation
        X_row = X_proc[i:i+1]
        if hasattr(model, 'n_features_in_'):
            expected = int(getattr(model, 'n_features_in_'))
            if X_row.shape[1] != expected:
                # pad or truncate
                if X_row.shape[1] < expected:
                    pad = np.zeros((1, expected - X_row.shape[1]))
                    X_row = np.hstack([X_row, pad])
                else:
                    X_row = X_row[:, :expected]
    for i, row in raw.iterrows():
        m = row['map']
        map_key = str(m).title()
        # choose model: prefer per-map focal calibrated, then per-map calibrated, then global
        model = per_map_models.get(map_key) if (per_map_models := per_map_models if 'per_map_models' in locals() else {}) else None
        # The above uses locals check; now load fresh per_map_models from the loader to be safe
        _, global_model, per_map_models, thresholds = load_models_and_thresholds()
        model = per_map_models.get(map_key, global_model)
        if model is None:
            raise RuntimeError('No model available (global or per-map)')

        try:
            p = model.predict_proba(X_row)[:,1][0]
        except Exception:
            try:
                p = float(model.decision_function(X_row)[0])
            except Exception:
                p = float(model.predict(X_row)[0])

        # threshold: prefer per-map threshold if present, else 0.5
        t = 0.5
        thr_entry = thresholds.get(map_key, {})
        if 'per_map' in thr_entry and 'threshold' in thr_entry['per_map']:
            t = float(thr_entry['per_map']['threshold'])
        elif 'global' in thr_entry and 'threshold' in thr_entry['global']:
            t = float(thr_entry['global']['threshold'])

        pred = int(p >= t)
        preds.append(pred)
        probs.append(p)

    out = raw.copy()
    out['pred_probability'] = probs
    out['pred_label'] = preds
    out.to_csv(output_path, index=False)
    print('Wrote', output_path)


def main(sample_input: Optional[str] = 'sample_inference_rows.csv'):
    infer(sample_input=sample_input)


if __name__ == '__main__':
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else 'sample_inference_rows.csv'
    main(sample)