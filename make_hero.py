import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

data = [
    {'model': 'LR', 'features': 'fp_only', 'random': 0.906, 'target': 0.783},
    {'model': 'LR', 'features': 'fp_esm',  'random': 0.922, 'target': 0.776},
    {'model': 'RF', 'features': 'fp_only', 'random': 0.954, 'target': 0.828},
    {'model': 'RF', 'features': 'fp_esm',  'random': 0.982, 'target': 0.781},
]

MODEL_LABEL   = {'LR': 'Log. Regression', 'RF': 'Random Forest'}
FEATURE_LABEL = {'fp_only': 'drug structure only', 'fp_esm': 'drug + protein seq.'}

COLOR_FP_ONLY = '#E07B39'
COLOR_FP_ESM  = '#2E86AB'

x_left  = 0.0
x_right = 1.0
x_left_label_val  = -0.04
x_left_label_name = -0.17
x_right_label_val  = 1.04
x_right_label_name = 1.18

left_val_y = {
    ('LR', 'fp_only'): 0.902,
    ('LR', 'fp_esm'):  0.926,
    ('RF', 'fp_only'): 0.954,
    ('RF', 'fp_esm'):  0.986,
}
left_name_y = {
    ('LR', 'fp_only'): 0.900,
    ('LR', 'fp_esm'):  0.924,
    ('RF', 'fp_only'): 0.952,
    ('RF', 'fp_esm'):  0.984,
}

right_val_y = {
    ('RF', 'fp_only'): 0.828,
    ('LR', 'fp_only'): 0.787,
    ('RF', 'fp_esm'):  0.779,
    ('LR', 'fp_esm'):  0.771,
}
right_name_y = {
    ('RF', 'fp_only'): 0.828,
    ('LR', 'fp_only'): 0.787,
    ('RF', 'fp_esm'):  0.779,
    ('LR', 'fp_esm'):  0.771,
}

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

for row in data:
    color    = COLOR_FP_ONLY if row['features'] == 'fp_only' else COLOR_FP_ESM
    ls       = '--' if row['model'] == 'LR' else '-'
    lw       = 1.5  if row['model'] == 'LR' else 2.5
    dot_size = 60   if row['model'] == 'LR' else 100

    ax.plot([x_left, x_right], [row['random'], row['target']],
            color=color, linestyle=ls, linewidth=lw, zorder=2)
    ax.scatter([x_left],  [row['random']], color=color, s=dot_size, zorder=3)
    ax.scatter([x_right], [row['target']], color=color, s=dot_size, zorder=3)

for row in data:
    color = COLOR_FP_ONLY if row['features'] == 'fp_only' else COLOR_FP_ESM
    key   = (row['model'], row['features'])
    label = f"{MODEL_LABEL[row['model']]}  {FEATURE_LABEL[row['features']]}"

    ax.text(x_left_label_val,  left_val_y[key],  f"{row['random']:.3f}",
            ha='right', va='center', fontsize=10, color=color)
    ax.text(x_left_label_name, left_name_y[key], label,
            ha='right', va='center', fontsize=9.5, color=color, fontweight='bold')

    ax.text(x_right_label_val,  right_val_y[key],  f"{row['target']:.3f}",
            ha='left', va='center', fontsize=10, color=color)
    ax.text(x_right_label_name, right_name_y[key], label,
            ha='left', va='center', fontsize=9.5, color=color, fontweight='bold')

ax.plot([x_left,  x_left],  [0.72, 1.007], color='#555555', linewidth=1.2, zorder=1)
ax.plot([x_right, x_right], [0.72, 1.007], color='#555555', linewidth=1.2, zorder=1)

ax.text(x_left,  1.020, 'Random Split',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#222222')
ax.text(x_left,  1.016, '(optimistic)',
        ha='center', va='top', fontsize=9, color='#888888', style='italic')

ax.text(x_right, 1.020, 'Held-out Target Split',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#222222')
ax.text(x_right, 1.016, '(97 unseen kinases)',
        ha='center', va='top', fontsize=9, color='#888888', style='italic')

ax.annotate(
    'adding protein seq. hurts AUROC\nby 0.047 on new kinases (Random Forest)',
    xy=(x_right, (0.828 + 0.781) / 2),
    xytext=(x_right + 0.30, 0.900),
    fontsize=9, color='#888888', ha='left', va='center',
    linespacing=1.4,
    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2,
                    connectionstyle='arc3,rad=-0.2'),
)

ax.set_xlim(-0.72, 1.95)
ax.set_ylim(0.72, 1.044)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(False)

ax.set_title(
    'Protein Sequence Improves Drug Binding Prediction, Except on Unseen Targets',
    fontsize=13, fontweight='bold', pad=8, loc='center'
)
ax.text(0.5, 1.008,
        'Drug structure alone vs. drug + protein sequence (ESM2) — predicting drug-kinase binding · AUROC across random and held-out protein splits',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9.5, color='#888888', style='italic')

plt.savefig('figures/fig_hero.png', dpi=150, bbox_inches='tight',
            transparent=False, facecolor='white')
print('Saved figures/fig_hero.png')
