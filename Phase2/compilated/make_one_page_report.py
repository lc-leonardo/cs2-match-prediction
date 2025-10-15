import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DST = Path(__file__).resolve().parent
combined_img = DST / 'combined_reliability.png'
metrics_csv = DST / 'prediction_summary_metrics.csv'
cls_csv = DST / 'classification_predictions_val_test.csv'
reg_csv = DST / 'regression_predictions_val_test.csv'
out_pdf = DST / 'one_page_report.pdf'

# canvas size (A4-like at 150 dpi)
W, H = 1240, 1754
canvas = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(canvas)

# fonts (fallback to default PIL font)
try:
    font_b = ImageFont.truetype('arial.ttf', 20)
    font_s = ImageFont.truetype('arial.ttf', 14)
except Exception:
    font_b = ImageFont.load_default()
    font_s = ImageFont.load_default()

# Title
d.text((40,20), 'Probability model diagnostics — concise report', fill='black', font=font_b)

# Place combined image
if combined_img.exists():
    im = Image.open(combined_img)
    # resize to fit width
    max_w = W - 80
    aspect = im.height / im.width
    new_w = max_w
    new_h = int(new_w * aspect)
    imr = im.resize((new_w, new_h))
    canvas.paste(imr, (40,60))
    y = 60 + new_h + 10
else:
    d.text((40,60), 'combined_reliability.png not found', fill='red', font=font_s)
    y = 120

# Metrics table
if metrics_csv.exists():
    dfm = pd.read_csv(metrics_csv)
    # show top 6 rows
    text = 'Key metrics (val/test):\n'
    for i, row in dfm.iterrows():
        text += f"{row['task'][:3]} | {row['model'][:12]:12s} | acc={row['accuracy']:.3f} auc={row['roc_auc']:.3f} brier={row['brier']:.3f}\n"
    d.text((40,y), text, fill='black', font=font_s)
    y += 16 * (len(dfm)+2)
else:
    d.text((40,y), 'prediction_summary_metrics.csv not found', fill='red', font=font_s)
    y += 40

# Example tables: pick 6 example rows for Logistic models
try:
    cls = pd.read_csv(cls_csv)
    reg = pd.read_csv(reg_csv)
    cls_log = cls[cls['model']=='Logistic Regression'].head(6)[['match_id','team_A','team_B','winner','pred_prob_A']]
    reg_log = reg[reg['model']=='LogisticRegression'].head(6)[['match_id','team_A','team_B','winner','pred_prob_A']]

    d.text((40,y), 'Classification examples (Logistic Regression):', fill='black', font=font_b)
    y += 24
    tx = ''
    for idx, r in cls_log.iterrows():
        tx += f"{int(r['match_id'])} | {r['team_A']} vs {r['team_B']} | actual: {r['winner']} | p(A)={r['pred_prob_A']:.2f}\n"
    d.text((40,y), tx, fill='black', font=font_s)
    y += 16 * (len(cls_log)+1)

    d.text((40,y), 'Regression examples (LogisticRegression calibrated):', fill='black', font=font_b)
    y += 24
    tx = ''
    for idx, r in reg_log.iterrows():
        tx += f"{int(r['match_id'])} | {r['team_A']} vs {r['team_B']} | actual: {r['winner']} | p(A)={r['pred_prob_A']:.2f}\n"
    d.text((40,y), tx, fill='black', font=font_s)
    y += 16 * (len(reg_log)+1)
except Exception as e:
    d.text((40,y), f'Failed to create example tables: {e}', fill='red', font=font_s)

# save as PDF
canvas.save(out_pdf, 'PDF', resolution=150)
print('Saved', out_pdf)
