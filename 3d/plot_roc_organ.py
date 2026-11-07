# plot_roc_organ.py
# OrganMNIST3D ROC curves in the same style as plot_roc_combined_breast.py:
# mean ± std across all 4 seeds, Original / +512 / +1024 overlaid.
#
# Uses the best-epoch test predictions stored in each checkpoint
# (best_ckpt_<run>_seed<s>.pt -> 'all_probs', 'all_labels'), so no inference
# is re-run and the labels are exactly the ones used in training.
#
# OrganMNIST3D has 11 classes, so each seed's curve is the macro-average of the
# 11 one-vs-rest ROC curves (AUC = macro one-vs-rest, as in your tables).
#
# Figures:
#   roc_combined_organmnist3d.png  - all configs, mean ± std over seeds
#   roc_perclass_organmnist3d.png  - one panel per config, each class mean ± std over seeds
#
# Usage: python3 plot_roc_organ.py <folder with best_ckpt_*.pt>

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

SAVE_PATH  = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else \
             "/your/path/classification_organmnist3d"
OUTPUT_DIR = SAVE_PATH
SEEDS      = [0, 1, 2, 3]
CLASS_NAMES = ["liver", "kidney-right", "kidney-left", "femur-right", "femur-left", "bladder",
               "heart", "lung-right", "lung-left", "spleen", "pancreas"]
NC = len(CLASS_NAMES)
MEAN_FPR   = np.linspace(0, 1, 200)

CONFIGS = [   # run name in checkpoint file, colour, legend label (same colours as Breast)
    ("Original Sequence", "#1f77b4", "Original Sequence"),
    ("512 Extra Tokens",  "#ff7f0e", "+512 Tokens"),
    ("1024 Extra Tokens", "#2ca02c", "+1024 Tokens"),
]

def load_seed(run_name, seed):
    path = os.path.join(SAVE_PATH, f"best_ckpt_{run_name.replace(' ', '_')}_seed{seed}.pt")
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        return None
    ck = torch.load(path, weights_only=False, map_location='cpu')
    if 'all_probs' not in ck:
        print(f"  {os.path.basename(path)} has no all_probs -- skipped")
        return None
    p = np.asarray(ck['all_probs'], dtype=float)
    y = np.asarray(ck['all_labels']).astype(int).ravel()
    return p, y, ck.get('auc'), ck.get('epoch')

def interp_roc(y_bin, score):
    fpr, tpr, _ = roc_curve(y_bin, score)
    t = np.interp(MEAN_FPR, fpr, tpr); t[0] = 0.0
    return t

# ── collect all seeds ─────────────────────────────────────────────────────────
results = {}
print(f"{'config':20s} {'seed':>4} {'epoch':>5} {'AUC recomputed':>15} {'AUC in ckpt':>12}")
for run_name, _, _ in CONFIGS:
    macro_tprs, aucs, class_tprs = [], [], {c: [] for c in range(NC)}
    class_aucs = {c: [] for c in range(NC)}
    for seed in SEEDS:
        r = load_seed(run_name, seed)
        if r is None:
            continue
        p, y, stored, ep = r
        per_class = []
        for c in range(NC):
            if (y == c).sum() == 0:
                continue
            t = interp_roc((y == c).astype(int), p[:, c])
            per_class.append(t); class_tprs[c].append(t)
            class_aucs[c].append(roc_auc_score((y == c).astype(int), p[:, c]))
        macro_tprs.append(np.mean(per_class, axis=0))
        a = roc_auc_score(y, p, multi_class='ovr', average='macro')
        aucs.append(a)
        flag = "" if stored is None or abs(a - stored) < 1e-3 else "   <-- MISMATCH"
        print(f"{run_name:20s} {seed:>4} {str(ep):>5} {a:>15.4f} "
              f"{(f'{stored:.4f}' if stored is not None else 'n/a'):>12}{flag}")
    results[run_name] = (macro_tprs, aucs, (class_tprs, class_aucs))

# ── figure 1: combined, same layout as the BreastMNIST figure ─────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for run_name, color, label in CONFIGS:
    tprs, aucs, _ = results[run_name]
    if not tprs:
        print(f"  Skipping {label} -- no valid seeds"); continue
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs, axis=0)                     # same std as the Breast script
    ax.plot(MEAN_FPR, mean_tpr, color=color, lw=2,
            label=f"{label} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    ax.fill_between(MEAN_FPR, np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1), color=color, alpha=0.15)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title(f'ROC Curves — OrganMNIST3D (macro-averaged, one-vs-rest)\n'
             f'(mean ± std across {len(SEEDS)} seeds)', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "roc_combined_organmnist3d.png")
plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nSaved: {out1}")

# ── figure 2: per-class, also mean ± std across all seeds ─────────────────────
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6 * len(CONFIGS), 6), squeeze=False)
cmap = plt.get_cmap('tab20')
for ax, (run_name, _, label) in zip(axes[0], CONFIGS):
    _, _, (class_tprs, class_aucs) = results[run_name]
    for c in range(NC):
        if not class_tprs[c]:
            continue
        T = np.array(class_tprs[c]); m = T.mean(0); m[-1] = 1.0; s = T.std(0)
        ax.plot(MEAN_FPR, m, color=cmap(c), lw=1.8,
                label=f"{CLASS_NAMES[c]} (AUC={np.mean(class_aucs[c]):.2f}±{np.std(class_aucs[c]):.2f})")
        ax.fill_between(MEAN_FPR, np.maximum(m - s, 0), np.minimum(m + s, 1),
                        color=cmap(c), alpha=0.10)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{label}\n(per class, mean ± std across {len(SEEDS)} seeds)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "roc_perclass_organmnist3d.png")
plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved: {out2}")
