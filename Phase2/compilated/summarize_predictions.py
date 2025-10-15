import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from pathlib import Path

DST = Path(__file__).resolve().parent
cls_csv = DST / 'classification_predictions_val_test.csv'
reg_csv = DST / 'regression_predictions_val_test.csv'

summaries = []
if cls_csv.exists():
    cls = pd.read_csv(cls_csv)
    for model in cls['model'].unique():
        dfm = cls[cls['model']==model]
        y_true = dfm['winner_is_A'].astype(int).values
        y_prob = dfm['pred_prob_A'].astype(float).values
        acc = accuracy_score(y_true, (y_prob>=0.5).astype(int))
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        brier = brier_score_loss(y_true, y_prob)
        try:
            ll = log_loss(y_true, np.vstack([1-y_prob, y_prob]).T)
        except Exception:
            ll = np.nan
        summaries.append({'task':'classification','model':model,'accuracy':acc,'roc_auc':auc,'brier':brier,'log_loss':ll})

if reg_csv.exists():
    reg = pd.read_csv(reg_csv)
    for model in reg['model'].unique():
        dfm = reg[reg['model']==model]
        y_true = dfm['winner_is_A'].astype(int).values
        y_prob = dfm['pred_prob_A'].astype(float).values
        acc = accuracy_score(y_true, (y_prob>=0.5).astype(int))
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        brier = brier_score_loss(y_true, y_prob)
        try:
            ll = log_loss(y_true, np.vstack([1-y_prob, y_prob]).T)
        except Exception:
            ll = np.nan
        summaries.append({'task':'regression','model':model,'accuracy':acc,'roc_auc':auc,'brier':brier,'log_loss':ll})

out_df = pd.DataFrame(summaries)
out_csv = DST / 'prediction_summary_metrics.csv'
out_df.to_csv(out_csv, index=False)
print('Wrote', out_csv)
print(out_df.to_string(index=False))
