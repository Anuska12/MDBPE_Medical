# plot_roc_acdc.py
# ACDC k-fold ROC curves in the same style as the BreastMNIST / DermaMNIST figures:
# mean ± std across 5 folds × 4 seeds (n = 20), Original / +1024 / +2048 overlaid.
#
# The k-fold models (model_<run>_fold<f>_seed<s>.pt) store weights but not
# predictions, so this script re-runs each saved best-AUC model on its own fold's
# test patients (same folds as training) to get the class probabilities.
# It reuses the data / model code from acdc_kfold_inference_time.py, which was
# checked to be identical to the k-fold training script -> keep both files in
# the same folder (~/scratch/codes/4d).
#
# Each run's curve is the macro-average of the 5 one-vs-rest ROC curves;
# legend AUC = macro one-vs-rest AUC (as in training), mean ± np.std over runs.
# The recomputed AUC of each run is printed next to the 'best_auc' stored in the
# model file, so you can check the figure matches your table.
#
# Figures (saved in SAVE_PATH = ~/scratch/codes/4d/k_fold):
#   roc_combined_acdc.png   - all configs
#   roc_perclass_acdc.png   - one panel per config, each class mean ± std
# Predictions are cached in acdc_kfold_probs.npz, so re-plotting is instant.
#
# Usage (GPU job recommended; works on CPU, just slowly):
#   python3 -u plot_roc_acdc.py            # compute (or load cached) + plot

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import acdc_kfold_inference_time as A          # verified-identical data + model code

DEVICE   = A.DEVICE
SEEDS    = [0, 1, 2, 3]
N_FOLDS  = 5
MEAN_FPR = np.linspace(0, 1, 200)
CACHE    = os.path.join(A.SAVE_PATH, "acdc_kfold_probs.npz")
CLASS_NAMES = sorted(A.CLASS_MAP, key=A.CLASS_MAP.get)     # ['DCM','HCM','MINF','NOR','RV']
NC = A.NUM_CLASSES

CONFIGS = [   # run name (as in model file names), colour, legend label
    ("Original",   "#1f77b4", "Original Sequence"),
    ("MDBPE 1024", "#ff7f0e", "+1024 Tokens"),
    ("MDBPE 2048", "#2ca02c", "+2048 Tokens"),
]

@torch.no_grad()
def predict(model, head, items):
    model.eval(); head.eval()
    out = []
    for s, p in items:
        s, p = s.unsqueeze(0).to(DEVICE), p.unsqueeze(0).to(DEVICE)
        out.append(torch.softmax(head(model.get_features(s[:, :-1], p[:, :-1])), 1)[0].cpu().numpy())
    return np.array(out)

# ── 1. predictions for every (config, fold, seed) ─────────────────────────────
if os.path.exists(CACHE):
    print("Loading cached predictions:", CACHE)
    z = np.load(CACHE, allow_pickle=True)
    preds = z['preds'].item()
else:
    print("Device:", DEVICE, flush=True)
    preds = {}
    for name, _, _ in CONFIGS:
        base = A.BASE_PATHS[name]
        try:
            num_tokens = torch.load(os.path.join(base, "metadata.pt"),
                                    weights_only=False).get('MAXIMUM_NUMBER_OF_TOKENS', A.FALLBACK_TOKENS[name])
        except Exception:
            num_tokens = A.FALLBACK_TOKENS[name]
        pooled = A.build_pooled_file_label_list(base)
        labels = np.array([l for _, l in pooled])
        pre = A.get_pre_loaded_tokens(base, num_tokens)
        print(f"\n{name}: {len(pooled)} patients, num_tokens={num_tokens}, tokenising...", flush=True)
        cache = {f: A.tokenise(torch.load(f, weights_only=False), num_tokens, pre) for f, _ in pooled}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(pooled)), labels)):
            items = [cache[pooled[i][0]] for i in test_idx]
            y = labels[test_idx]
            for seed in SEEDS:
                mpath = os.path.join(A.SAVE_PATH, f"model_{name.replace(' ', '_')}_fold{fold}_seed{seed}.pt")
                if not os.path.exists(mpath):
                    print(f"  missing {os.path.basename(mpath)}"); continue
                ck = torch.load(mpath, weights_only=False, map_location='cpu')
                model = A.TransformerDecoderModel(num_tokens).to(DEVICE)
                head  = A.ClassificationHead(NC).to(DEVICE)
                model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
                preds[(name, fold, seed)] = (predict(model, head, items), y, ck.get('best_auc'))
                print(f"  fold {fold} seed {seed} done", flush=True)
                del model, head
        del cache
    np.savez(CACHE, preds=np.array(preds, dtype=object))
    print("cached predictions ->", CACHE)

# ── 2. ROC per run + check against stored best_auc ────────────────────────────
print(f"\n{'config':12s} {'fold':>4} {'seed':>4} {'AUC recomputed':>15} {'best_auc stored':>16}")
results = {}
for name, _, _ in CONFIGS:
    macro_tprs, aucs = [], []
    class_tprs = {c: [] for c in range(NC)}; class_aucs = {c: [] for c in range(NC)}
    for (n, fold, seed), (p, y, stored) in sorted(preds.items(), key=lambda kv: kv[0][1:]):
        if n != name:
            continue
        per_class = []
        for c in range(NC):
            if (y == c).sum() == 0 or (y != c).sum() == 0:
                continue
            fpr, tpr, _ = roc_curve((y == c).astype(int), p[:, c])
            t = np.interp(MEAN_FPR, fpr, tpr); t[0] = 0.0
            per_class.append(t); class_tprs[c].append(t)
            class_aucs[c].append(roc_auc_score((y == c).astype(int), p[:, c]))
        macro_tprs.append(np.mean(per_class, axis=0))
        a = roc_auc_score(y, p, multi_class='ovr', average='macro')
        aucs.append(a)
        flag = "" if stored is None or abs(a - stored) < 1e-3 else "   <-- MISMATCH"
        st = f"{stored:.4f}" if stored is not None else "n/a"
        print(f"{name:12s} {fold:>4} {seed:>4} {a:>15.4f} {st:>16}{flag}")
    results[name] = (macro_tprs, aucs, class_tprs, class_aucs)
    print(f"{name:12s} mean AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} over {len(aucs)} runs\n")

# ── 3. figure 1: combined, same layout as BreastMNIST / DermaMNIST ────────────
fig, ax = plt.subplots(figsize=(8, 7))
for name, color, label in CONFIGS:
    tprs, aucs, _, _ = results[name]
    if not tprs:
        print(f"  Skipping {label} -- no runs"); continue
    m = np.mean(tprs, axis=0); m[-1] = 1.0
    s = np.std(tprs, axis=0)
    ax.plot(MEAN_FPR, m, color=color, lw=2,
            label=f"{label} (AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f})")
    ax.fill_between(MEAN_FPR, np.maximum(m - s, 0), np.minimum(m + s, 1), color=color, alpha=0.15)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — ACDC (macro-averaged, one-vs-rest)\n'
             f'(mean ± std across {N_FOLDS} folds × {len(SEEDS)} seeds, n={N_FOLDS*len(SEEDS)})', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out1 = os.path.join(A.SAVE_PATH, "roc_combined_acdc.png")
plt.savefig(out1, dpi=150, bbox_inches='tight'); plt.close()
print("Saved:", out1)

# ── 4. figure 2: per class, mean ± std across runs ────────────────────────────
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6 * len(CONFIGS), 6), squeeze=False)
cmap = plt.get_cmap('tab10')
for ax, (name, _, label) in zip(axes[0], CONFIGS):
    _, _, class_tprs, class_aucs = results[name]
    for c in range(NC):
        if not class_tprs[c]:
            continue
        T = np.array(class_tprs[c]); m = T.mean(0); m[-1] = 1.0; s = T.std(0)
        ax.plot(MEAN_FPR, m, color=cmap(c), lw=1.8,
                label=f"{CLASS_NAMES[c]} (AUC={np.mean(class_aucs[c]):.2f}±{np.std(class_aucs[c]):.2f})")
        ax.fill_between(MEAN_FPR, np.maximum(m - s, 0), np.minimum(m + s, 1), color=cmap(c), alpha=0.10)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{label}\n(per class, mean ± std across {N_FOLDS*len(SEEDS)} runs)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = os.path.join(A.SAVE_PATH, "roc_perclass_acdc.png")
plt.savefig(out2, dpi=150, bbox_inches='tight'); plt.close()
print("Saved:", out2)
