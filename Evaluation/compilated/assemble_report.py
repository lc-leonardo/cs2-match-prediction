import os
import json
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_REG = os.path.join(ROOT, 'Regression')
SRC_CLS = os.path.join(ROOT, 'Classification', 'models')
DST = os.path.abspath(os.path.dirname(__file__))

# files to copy
files_to_copy = [
    os.path.join(SRC_REG, 'prob_regression_summary.json'),
    os.path.join(SRC_REG, 'LogisticRegression_calibration.png'),
    os.path.join(SRC_REG, 'LightGBM_calibration.png'),
    os.path.join(SRC_REG, 'MLP_calibration.png'),
    os.path.join(SRC_REG, 'per_map_bootstrap_bar_ci.png'),
    os.path.join(SRC_CLS, 'results_summary.json')
]

copied = []
for src in files_to_copy:
    if os.path.exists(src):
        dst = os.path.join(DST, os.path.basename(src))
        shutil.copyfile(src, dst)
        copied.append(dst)
        print('Copied', src, '->', dst)
    else:
        print('Missing', src)

# Build diagnostics table from JSON summaries
reg_json = os.path.join(DST, 'prob_regression_summary.json')
cls_json = os.path.join(DST, 'results_summary.json')
rows = []
if os.path.exists(reg_json):
    with open(reg_json,'r',encoding='utf-8') as f:
        r = json.load(f)
    # pick main models
    for name in ['LogisticRegression','LightGBM','MLP']:
        if name in r.get('models',{}):
            m = r['models'][name]
            rows.append({'task':'regression','model':name,
                         'log_loss':m.get('log_loss'),
                         'brier':m.get('brier'),
                         'roc_auc':m.get('roc_auc')})

if os.path.exists(cls_json):
    with open(cls_json,'r',encoding='utf-8') as f:
        c = json.load(f)
    final = c.get('final_test_results',{})
    for name,metrics in final.items():
        rows.append({'task':'classification','model':name,
                     'accuracy':metrics.get('accuracy'),
                     'roc_auc':metrics.get('roc_auc'),
                     'brier':metrics.get('brier_score')})

if rows:
    df = pd.DataFrame(rows)
    csv_path = os.path.join(DST, 'diagnostics_table.csv')
    df.to_csv(csv_path, index=False)
    print('Wrote', csv_path)

# Combine calibration images side-by-side if available
left_paths = []
for name in ['LogisticRegression_calibration.png','LightGBM_calibration.png','MLP_calibration.png']:
    p = os.path.join(DST, name)
    if os.path.exists(p):
        left_paths.append(p)

# For classification, try to use a classifier calibration plot if present in Regression folder or Classification models
cls_plot = None
for candidate in ['lightgbm_group_calibrated.png','calibration.png','group_calibration.png']:
    p = os.path.join(SRC_CLS, candidate)
    if os.path.exists(p):
        cls_plot = p
        break

# fallback: if classification models directory has any pngs, pick first
if cls_plot is None:
    for f in os.listdir(SRC_CLS):
        if f.lower().endswith('.png'):
            cls_plot = os.path.join(SRC_CLS,f)
            break

images = []
for p in left_paths:
    try:
        images.append(Image.open(p))
    except Exception as e:
        print('Failed to open',p,e)

if cls_plot:
    try:
        images.append(Image.open(cls_plot))
    except Exception as e:
        print('Failed to open classification plot',cls_plot,e)

if images:
    # create a horizontal montage
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    montage = Image.new('RGB', (total_width, max_height), color=(255,255,255))
    x_offset = 0
    for im in images:
        montage.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    out_path = os.path.join(DST, 'combined_reliability.png')
    montage.save(out_path)
    print('Saved combined image to', out_path)
else:
    print('No images found to combine')

print('Done')
