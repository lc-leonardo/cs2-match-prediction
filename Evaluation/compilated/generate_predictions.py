import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'map_picks_last6m_top50_ml_ready.csv'
CLS_MODELS_DIR = ROOT / 'Classification' / 'models'
REG_MODELS_DIR = ROOT / 'Regression'
DST = Path(__file__).resolve().parent

print('Loading dataset', DATA_PATH)
df = pd.read_csv(DATA_PATH)
# select validation + test rows
mask = df['split_row'].isin(['val','test'])
df_sel = df[mask].copy()
print('Selected rows:', len(df_sel))
# reset index so positional arrays from models align with dataframe rows
df_sel = df_sel.reset_index(drop=True)

# Load preprocessors if available
cls_preproc_path = CLS_MODELS_DIR / 'preprocessor.pkl'
reg_preproc_path = REG_MODELS_DIR / 'preprocessor.pkl'
cls_preproc = None
reg_preproc = None
if cls_preproc_path.exists():
    cls_preproc = joblib.load(cls_preproc_path)
    print('Loaded classification preprocessor')
else:
    print('Classification preprocessor not found at', cls_preproc_path)
if reg_preproc_path.exists():
    reg_preproc = joblib.load(reg_preproc_path)
    print('Loaded regression preprocessor')
else:
    print('Regression preprocessor not found at', reg_preproc_path)

# Helper: apply preprocessor and get X
FEATURES = [
    'map_number','team_A_rank','team_B_rank','rank_diff','abs_rank_diff',
    'picked_by_is_A','is_decider','map_winrate_A','map_winrate_B','recent_form_A','recent_form_B','map'
]

# Classification predictions
cls_models = {
    'Logistic Regression': CLS_MODELS_DIR / 'logistic_regression_model.pkl',
    'Random Forest': CLS_MODELS_DIR / 'random_forest_model.pkl',
    'LightGBM': CLS_MODELS_DIR / 'lightgbm_model.pkl',
    'XGBoost': CLS_MODELS_DIR / 'xgboost_model.pkl',
    'MLP': CLS_MODELS_DIR / 'mlp_model.pkl'
}
cls_out_rows = []
for name,path in cls_models.items():
    if not path.exists():
        print('Missing model', path)
        continue
    model = joblib.load(path)
    print('Loaded', name)
    # apply preprocessor if available
    if cls_preproc is not None:
        X = cls_preproc.transform(df_sel)
    else:
        X = df_sel[FEATURES].copy()
    # predict_proba
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        # assume column 1 is prob of winner_is_A==1
        p1 = probs[:,1]
    else:
        # fallback to decision_function + sigmoid
        try:
            dfun = model.predict(X)
            p1 = dfun
        except Exception:
            p1 = np.zeros(len(df_sel))
    preds = (p1 >= 0.5).astype(int)
    for i, row in df_sel.iterrows():
        cls_out_rows.append({
            'match_id': row['match_id'], 'map': row['map'], 'team_A': row['team_A'], 'team_B': row['team_B'],
            'winner': row['winner'], 'winner_is_A': row['winner_is_A'], 'split_row': row['split_row'],
            'model': name, 'pred_prob_A': float(p1[i]), 'pred_winner_is_A': int(preds[i])
        })

cls_df_out = pd.DataFrame(cls_out_rows)
cls_csv = DST / 'classification_predictions_val_test.csv'
cls_df_out.to_csv(cls_csv, index=False)
print('Wrote', cls_csv)

# Regression predictions: load the calibrated models (if available)
reg_models = {
    'LogisticRegression': REG_MODELS_DIR / 'LogisticRegression_calibrated.pkl',
    'LightGBM': REG_MODELS_DIR / 'LightGBM_calibrated.pkl',
    'MLP': REG_MODELS_DIR / 'MLP_calibrated.pkl'
}
reg_out_rows = []
for name,path in reg_models.items():
    if not path.exists():
        print('Missing regression model', path)
        continue
    model = joblib.load(path)
    print('Loaded', name)
    if reg_preproc is not None:
        X = reg_preproc.transform(df_sel)
    else:
        X = df_sel[FEATURES].copy()
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        p1 = probs[:,1]
    else:
        # if regressor that outputs probability-like value in predict
        try:
            p1 = model.predict(X)
        except Exception:
            p1 = np.zeros(len(df_sel))
    for i, row in df_sel.iterrows():
        reg_out_rows.append({
            'match_id': row['match_id'], 'map': row['map'], 'team_A': row['team_A'], 'team_B': row['team_B'],
            'winner': row['winner'], 'winner_is_A': row['winner_is_A'], 'split_row': row['split_row'],
            'model': name, 'pred_prob_A': float(p1[i])
        })

reg_df_out = pd.DataFrame(reg_out_rows)
reg_csv = DST / 'regression_predictions_val_test.csv'
reg_df_out.to_csv(reg_csv, index=False)
print('Wrote', reg_csv)

print('Done')
