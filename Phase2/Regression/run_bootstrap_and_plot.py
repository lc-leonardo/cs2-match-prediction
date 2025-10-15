"""Recompute per-map bootstrap results and regenerate CI bar plot.

Usage: run this script from project root (it will look for Phase2/Regression and data files).
Adjust B and MIN_TEST_COUNT below or pass via environment variables by editing this file.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

# Parameters you can tweak
B = 2000  # number of bootstrap resamples
MIN_TEST_COUNT = 20  # minimum test rows to attempt bootstrap
CI_LO_P = 2.5
CI_HI_P = 97.5

# Paths
ROOT = os.getcwd()
DATA_PATH = os.path.join('Phase2', 'map_picks_last6m_top50_ml_ready.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'map_picks_last6m_top50_ml_ready.csv'

BOOT_OUT_DIR = os.path.join('Phase2', 'Regression')
os.makedirs(BOOT_OUT_DIR, exist_ok=True)
BOOT_OUT_CSV = os.path.join(BOOT_OUT_DIR, f'per_map_bootstrap_results_B{B}.csv')
BOOT_OUT_CANON = os.path.join(BOOT_OUT_DIR, 'per_map_bootstrap_results.csv')
PLOT_OUT = os.path.join(BOOT_OUT_DIR, 'per_map_bootstrap_bar_ci.png')

print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Determine features similar to notebook
feature_cols = [
    'map','map_number','team_A_rank','team_B_rank','rank_diff','abs_rank_diff','picked_by_is_A','is_decider',
    'map_winrate_A','map_winrate_B','recent_form_A','recent_form_B',
    'elo_A','elo_B','map_elo_A','map_elo_B','h2h_rate_A_vs_B','last5_map_rate_A','last5_map_rate_B',
    'elo_diff','map_elo_diff','last5_map_diff','h2h_diff','h2h_raw','map_count_train','map_age_days','new_map_flag','team_A_te','team_B_te',
]
feature_cols = [c for c in feature_cols if c in df.columns]
print('Using features:', feature_cols)
X = df[feature_cols].copy()
y = df['winner_is_A'].copy()

# split masks
if 'split_row' in df.columns:
    train_mask = df['split_row']=='train'
    val_mask = df['split_row']=='val'
    test_mask = df['split_row']=='test'
else:
    # fallback: simple split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # create masks of indices
    train_mask = df.index.isin(X_train.index)
    val_mask = pd.Series(False, index=df.index)
    test_mask = df.index.isin(X_test.index)

# Preprocessor
categorical_features = ['map'] if 'map' in feature_cols else []
numerical_features = [c for c in feature_cols if c not in categorical_features]
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# Fit preprocessor on training rows
X_train = X[train_mask]
preprocessor.fit(X_train)
X_proc = preprocessor.transform(X)

# Helper to get processed rows
def proc_rows(mask):
    return X_proc[mask.values]

# Try to load the global calibrated model (prefer CalibratedClassifierCV wrapper)
global_calib_path = os.path.join(BOOT_OUT_DIR, 'LightGBM_calibrated.pkl')
global_model = None
if os.path.exists(global_calib_path):
    try:
        global_model = joblib.load(global_calib_path)
        print('Loaded global calibrated model from', global_calib_path)
    except Exception as e:
        print('Failed to load global calibrated model:', e)
# fallback: look for any *_calibrated.pkl
if global_model is None:
    for fname in os.listdir(BOOT_OUT_DIR):
        if fname.endswith('_calibrated.pkl'):
            try:
                global_model = joblib.load(os.path.join(BOOT_OUT_DIR, fname))
                print('Loaded global calibrated model from', fname)
                break
            except Exception:
                continue

# If still not found, train a simple LightGBM on training data and calibrate on val (if exists)
if global_model is None:
    print('No calibrated model found; training a LightGBM and calibrating on validation (if present)')
    X_tr = proc_rows(train_mask)
    y_tr = y[train_mask]
    base = LGBMClassifier(random_state=42)
    base.fit(X_tr, y_tr)
    # if we have val set, calibrate
    if val_mask.sum() > 0:
        from sklearn.calibration import CalibratedClassifierCV
        calib = CalibratedClassifierCV(base, cv='prefit', method='sigmoid')
        X_val_proc = proc_rows(val_mask)
        calib.fit(X_val_proc, y[val_mask])
        global_model = calib
    else:
        global_model = base
    # persist the model
    try:
        joblib.dump(global_model, global_calib_path)
        print('Saved newly created calibrated model to', global_calib_path)
    except Exception as e:
        print('Could not save model:', e)

# Determine candidate maps: use maps present in test slice with at least MIN_TEST_COUNT
map_counts = df.loc[test_mask, 'map'].value_counts()
candidate_maps = map_counts[map_counts >= MIN_TEST_COUNT].index.tolist()
print('Candidate maps for bootstrap (test_count >=', MIN_TEST_COUNT, '):', candidate_maps)

results = []
start = time.time()
for m in candidate_maps:
    # indices of rows in df for this map and test set
    map_test_mask = (df['map'] == m) & test_mask
    test_idxs = np.where(map_test_mask.values)[0]
    test_count = len(test_idxs)
    if test_count == 0:
        continue
    # processed arrays for test slice
    X_test_map = X_proc[test_idxs]
    y_test_map = y.iloc[test_idxs].values
    # global predictions
    try:
        p_global = global_model.predict_proba(X_test_map)[:,1]
    except Exception as e:
        # try fallback predict
        try:
            p_global = global_model.predict(X_test_map).astype(float)
        except Exception as e2:
            print('Global model predict failed for map', m, e, e2)
            p_global = np.full(test_count, np.nan)
    # train per-map model on training rows for this map
    map_train_mask = (df['map'] == m) & train_mask
    train_idxs = np.where(map_train_mask.values)[0]
    if len(train_idxs) < 10:
        print('Skipping map', m, 'insufficient train rows:', len(train_idxs))
        results.append({'map': m, 'test_count': test_count, 'mean_diff': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'global_logloss': np.nan, 'per_map_logloss': np.nan})
        continue
    X_map_train = X_proc[train_idxs]
    y_map_train = y.iloc[train_idxs].values
    try:
        per_model = LGBMClassifier(random_state=42)
        per_model.fit(X_map_train, y_map_train)
        p_permap = per_model.predict_proba(X_test_map)[:,1]
    except Exception as e:
        print('Per-map model failed for', m, e)
        results.append({'map': m, 'test_count': test_count, 'mean_diff': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': np.nan, 'global_logloss': np.nan, 'per_map_logloss': np.nan})
        continue
    # compute point estimates
    try:
        g_ll = log_loss(y_test_map, p_global)
    except Exception:
        g_ll = np.nan
    try:
        p_ll = log_loss(y_test_map, p_permap)
    except Exception:
        p_ll = np.nan
    # bootstrap
    diffs = []
    rng = np.random.default_rng(42)
    for b in range(B):
        # resample indices relative to the test slice
        res_idx = rng.integers(0, test_count, size=test_count)
        yb = y_test_map[res_idx]
        pg = p_global[res_idx]
        pp = p_permap[res_idx]
        # clamp probabilities to (eps,1-eps) to avoid log_loss errors
        eps = 1e-12
        pg = np.clip(pg, eps, 1-eps)
        pp = np.clip(pp, eps, 1-eps)
        try:
            gll_b = log_loss(yb, pg)
            pll_b = log_loss(yb, pp)
            diffs.append(gll_b - pll_b)
        except Exception:
            diffs.append(np.nan)
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        mean_diff = np.nan
        ci_low = np.nan
        ci_high = np.nan
        pval = np.nan
    else:
        mean_diff = float(np.mean(diffs))
        ci_low = float(np.percentile(diffs, CI_LO_P))
        ci_high = float(np.percentile(diffs, CI_HI_P))
        # p-value: fraction of diffs <= 0 (null that per-map is no better than global)
        pval = float((diffs <= 0).mean())
    print(f"Map={m}, test_count={test_count}, global_ll={g_ll:0.4f}, per_map_ll={p_ll:0.4f}, mean_diff={mean_diff:0.4f}, ci=({ci_low:0.4f},{ci_high:0.4f}), p={pval:0.3f}")
    results.append({'map': m, 'test_count': int(test_count), 'mean_diff': mean_diff, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pval, 'global_logloss': g_ll, 'per_map_logloss': p_ll})

# write outputs
df_out = pd.DataFrame(results).sort_values('mean_diff', ascending=False).reset_index(drop=True)
df_out.to_csv(BOOT_OUT_CSV, index=False)
# copy to canonical filename
try:
    df_out.to_csv(BOOT_OUT_CANON, index=False)
except Exception:
    pass
# also write json
with open(os.path.join(BOOT_OUT_DIR, f'per_map_bootstrap_results_B{B}.json'), 'w') as f:
    json.dump(df_out.to_dict(orient='records'), f, indent=2)

# regenerate plot (same logic as notebook cell)
import matplotlib.pyplot as plt
if not df_out.empty:
    df_b = df_out.copy()
    df_b = df_b.dropna(subset=['mean_diff']).copy()
    labels = df_b['map'].astype(str).tolist()
    means = df_b['mean_diff'].values
    if 'ci_low' in df_b.columns and 'ci_high' in df_b.columns:
        err_lower = means - df_b['ci_low'].values
        err_upper = df_b['ci_high'].values - means
    else:
        err_lower = np.zeros_like(means)
        err_upper = np.zeros_like(means)
    colors = []
    for low, high in zip(df_b.get('ci_low', pd.Series([np.nan]*len(df_b))), df_b.get('ci_high', pd.Series([np.nan]*len(df_b)))):
        if pd.isna(low) or pd.isna(high):
            colors.append('#bbbbbb')
        elif low > 0:
            colors.append('#2ca02c')
        elif high < 0:
            colors.append('#d62728')
        else:
            colors.append('#ff7f0e')
    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(labels)), 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[err_lower, err_upper], align='center', alpha=0.9, color=colors, capsize=6)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean diff (global_logloss - per_map_logloss)')
    ax.set_title(f'Per-map bootstrap mean diff with {100-(CI_LO_P)}% CI (B={B})')
    plt.tight_layout()
    fig.savefig(PLOT_OUT, bbox_inches='tight', dpi=150)
    print('Saved regenerated plot to', PLOT_OUT)

print('Done. Time elapsed:', time.time() - start)
