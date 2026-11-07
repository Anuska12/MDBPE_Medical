# -*- coding: utf-8 -*-
# attention_rollout_acdc.py
#
# Attention rollout + ground-truth-region overlap for the ACDC k-fold models,
# using the SAME method as the BreastMNIST / DermaMNIST rollout scripts:
#   - hook-based extraction on the real nn.TransformerEncoderLayer forward
#   - MAX over heads, 0.5*A + 0.5*I, row-normalised, multiplied through the layers
#   - CLS row, skipping CLS (0) and SOS (1), projected onto each token's anchor position
#   - Gaussian smoothing (sigma=1, in-plane only) + min-max normalisation
#   - overlap = share of the attention mass that falls inside the region of interest
#
# Differences forced by the data (state these in the thesis):
#   1. Region of interest = ACDC's MANUAL segmentation (RV + myocardium + LV, all
#      foreground labels) instead of an Otsu mask, since ACDC ships expert masks.
#      The mask is resampled to the 32x27x2 grid exactly like the image
#      (trilinear, align_corners=False), then thresholded at 0.5.
#   2. ACDC masks exist only for the ED and ES frames. Overlap is computed on
#      the frames of the 8-frame grid that ARE those frames exactly (frame index
#      divisible by the temporal stride; ED = frame 1 is always included), and
#      averaged over them. No mask is copied to a neighbouring frame.
#   3. Every patient's model input is checked against its raw NIfTI volume
#      (base-token voxels must match), so the patient <-> file mapping is
#      verified rather than assumed.
#
# All 5 folds x 4 seeds x 3 configs; each test patient is evaluated by the
# 4 seed models of the fold where it is held out.
#
# Keep this file in ~/scratch/codes/4d next to acdc_kfold_inference_time.py
# (data + model code are imported from it; that file was checked to be
# identical to the k-fold training script).
#
# usage:
#   python3 -u attention_rollout_acdc.py 0          # fold 0 (all configs, all seeds)
#   python3 -u attention_rollout_acdc.py all        # every fold, one after another
#   python3 -u attention_rollout_acdc.py summary    # overlap table + t-tests from the saved npz files
#   add  --limit 3  to process only 3 patients per run (quick test)

import os, sys, glob, re
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold
import acdc_kfold_inference_time as A

OUTPUT_DIR = os.path.join(A.SAVE_PATH, "attention_rollout")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE  = A.DEVICE
SEEDS   = A.SEEDS
N_FOLDS = A.N_FOLDS
H, W, D, T = A.TARGET_H, A.TARGET_W, A.TARGET_D, A.TARGET_T
CONFIGS = [   # key, run name (as in the model file names)
    ('original', 'Original'),
    ('1024',     'MDBPE 1024'),
    ('2048',     'MDBPE 2048'),
]
CLASS_NAMES = sorted(A.CLASS_MAP, key=A.CLASS_MAP.get)
PATIENT_CACHE = os.path.join(OUTPUT_DIR, "patients_grid_and_masks.pt")

# ── 1. patients: model-grid volume (as in the processor) + GT masks on that grid ──
def resample(vol3d):
    """(h,w,d) float -> (H,W,D), the processor's per-frame trilinear resize."""
    t = torch.tensor(vol3d).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(H, W, D), mode='trilinear', align_corners=False).squeeze()

def load_acdc_4d(patient_dir):
    """Copied from the ACDC processor / acdc_test_export.py."""
    files_4d = glob.glob(os.path.join(patient_dir, "*_4d.nii.gz"))
    if len(files_4d) == 0:
        return None, None
    data = nib.load(files_4d[0]).get_fdata()
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-8:
        return None, None
    data = (data - mn) / (mx - mn)
    vol = torch.stack([resample(data[:, :, :, t]) for t in range(data.shape[3])], dim=-1)
    step = max(1, data.shape[3] // T)
    vol = vol[:, :, :, ::step][:, :, :, :T]
    return (vol * 255).long().numpy().astype(np.int16), step

def load_gt_masks(patient_dir, step):
    """{grid frame k: bool (H,W,D)} for every GT frame that lands exactly on the grid."""
    masks = {}
    for f in sorted(glob.glob(os.path.join(patient_dir, "*_frame*_gt.nii.gz"))):
        frame = int(re.search(r"_frame(\d+)_gt", os.path.basename(f)).group(1))   # 1-based
        if (frame - 1) % step != 0 or (frame - 1) // step >= T:
            continue                                  # not one of the 8 sampled frames
        k = (frame - 1) // step
        fg = (nib.load(f).get_fdata() > 0).astype(np.float32)   # RV + myo + LV
        masks[k] = (resample(fg) > 0.5).numpy()
    return masks

def build_patients():
    if os.path.exists(PATIENT_CACHE):
        return torch.load(PATIENT_CACHE, weights_only=False)
    P = {}
    for split, root in (('train', A.ACDC_ROOT_TRAIN), ('test', A.ACDC_ROOT_TEST)):
        for i, d in enumerate(sorted(glob.glob(os.path.join(root, "patient*")))):
            pid = int(re.search(r"patient(\d+)", d).group(1))
            vol, step = load_acdc_4d(d)
            if vol is None:
                continue
            group = None
            cfg = os.path.join(d, "Info.cfg")
            if os.path.exists(cfg):
                for line in open(cfg):
                    if line.startswith("Group:"):
                        group = line.split(":")[1].strip()
            P[pid] = dict(split=split, pos=i, vol=vol, step=step,
                          masks=load_gt_masks(d, step),
                          label=A.CLASS_MAP.get(group, None))
    torch.save(P, PATIENT_CACHE)
    return P

# ── 2. patient <-> file mapping, verified on the base-token voxels ─────────────
def match_frac(data, vol):
    hit = n = 0
    for c, x, y, z, t in data:
        if c < A.BASE_TOKENS:
            n += 1; hit += int(vol[x, y, z, t] == c)
    return hit / max(n, 1)

def map_files(pooled, P):
    """file path -> patient id. Tries the position-based assumption of the
    training script first, searches all patients only if it fails."""
    by_pos = {(p['split'], p['pos']): pid for pid, p in P.items()}
    out, remapped, label_bad = {}, 0, 0
    for fpath, lab in pooled:
        split = 'test' if os.sep + 'tokensequences_test' + os.sep in fpath else 'train'
        idx = int(os.path.basename(fpath).replace(".dat", ""))
        data = torch.load(fpath, weights_only=False)
        pid = by_pos.get((split, idx))
        if pid is None or match_frac(data, P[pid]['vol']) < 0.99:
            scores = {q: match_frac(data, P[q]['vol']) for q in P}
            pid = max(scores, key=scores.get)
            if scores[pid] < 0.99:
                raise RuntimeError(f"{fpath}: no patient volume matches (best {scores[pid]:.3f})")
            remapped += 1
        if P[pid]['label'] is not None and P[pid]['label'] != lab:
            label_bad += 1
        out[fpath] = pid
    print(f"  mapping verified: {len(out)} files, {remapped} not at the assumed position, "
          f"{label_bad} label mismatches vs Info.cfg", flush=True)
    if label_bad:
        print("  !! WARNING: training labels differ from Info.cfg for some patients -- check before using results")
    return out

# ── 3. rollout ─────────────────────────────────────────────────────────────────
def register_attention_hooks(model):
    """Same as the verified Derma/Breast scripts (MAX over heads)."""
    attn_maps, handles, originals = [], [], []
    for layer in model.transformer.layers:
        orig = layer.self_attn.forward
        originals.append((layer.self_attn, orig))
        def patched(*args, orig=orig, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return orig(*args, **kwargs)
        layer.self_attn.forward = patched
        def hook(module, inp, out):
            if isinstance(out, tuple) and out[1] is not None:
                attn_maps.append(out[1].detach().max(dim=1).values[0])   # (L, L)
        handles.append(layer.self_attn.register_forward_hook(hook))
    return attn_maps, handles, originals

@torch.no_grad()
def rollout_and_pred(model, head, seq, pos):
    """Returns (pred, CLS rollout vector over the input tokens).
    Only the CLS row of  A_L ... A_1  is needed, so it is computed as a
    row vector (e0^T A_L ... A_1) -- identical to row 0 of the full matrix
    product in the Derma/Breast scripts, without the L^3 matmul."""
    attn_maps, handles, originals = register_attention_hooks(model)
    model.eval(); head.eval()
    feats = model.get_features(seq.unsqueeze(0).to(DEVICE), pos.unsqueeze(0).to(DEVICE))
    pred = int(head(feats).argmax(dim=1).item())
    for h in handles: h.remove()
    for m, f in originals: m.forward = f
    if not attn_maps:
        raise RuntimeError("No attention maps captured.")
    v = torch.zeros(attn_maps[0].size(0), device=attn_maps[0].device); v[0] = 1.0
    for Am in reversed(attn_maps):
        Am.mul_(0.5); Am.diagonal().add_(0.5)
        Am.div_(Am.sum(dim=-1, keepdim=True))
        v = v @ Am
    return pred, v[2:].cpu().numpy()          # skip CLS (0) and SOS (1)

def heatmap(cls_vec, data):
    """Project onto anchor positions (mean per position), smooth in-plane, min-max."""
    hm = np.zeros((H, W, D, T)); cnt = np.zeros((H, W, D, T))
    tok = np.asarray([d[1:5] for d in data[:len(cls_vec)]], dtype=int)
    np.add.at(hm,  (tok[:, 0], tok[:, 1], tok[:, 2], tok[:, 3]), cls_vec[:len(tok)])
    np.add.at(cnt, (tok[:, 0], tok[:, 1], tok[:, 2], tok[:, 3]), 1)
    hm = hm / np.maximum(cnt, 1)
    hm = gaussian_filter(hm, sigma=(1.0, 1.0, 0, 0))
    hm = hm - hm.min()
    return hm / hm.max() if hm.max() > 0 else hm

def overlap(hm, masks):
    vals = []
    for k, m in masks.items():
        a = hm[:, :, :, k]
        if a.sum() > 0:
            vals.append(float(a[m].sum() / a.sum()))
    return float(np.mean(vals)) if vals else None

def load_model(name, fold, seed, num_tokens):
    ck = torch.load(os.path.join(A.SAVE_PATH, f"model_{name.replace(' ', '_')}_fold{fold}_seed{seed}.pt"),
                    weights_only=False, map_location='cpu')
    model = A.TransformerDecoderModel(num_tokens).to(DEVICE)
    head = A.ClassificationHead(A.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
    return model.eval(), head.eval()

def config_setup(name):
    base = A.BASE_PATHS[name]
    try:
        nt = torch.load(os.path.join(base, "metadata.pt"),
                        weights_only=False).get('MAXIMUM_NUMBER_OF_TOKENS', A.FALLBACK_TOKENS[name])
    except Exception:
        nt = A.FALLBACK_TOKENS[name]
    pooled = A.build_pooled_file_label_list(base)
    labels = np.array([l for _, l in pooled])
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                 .split(np.zeros(len(pooled)), labels))
    return base, nt, pooled, folds, A.get_pre_loaded_tokens(base, nt)

def run_folds(folds_wanted, limit=None):
    P = build_patients()
    n_mask = sum(1 for p in P.values() if p['masks'])
    print(f"{len(P)} patients, {n_mask} with >=1 GT frame on the grid; "
          f"GT frames used per patient: {np.bincount([len(p['masks']) for p in P.values()])}", flush=True)
    results = {(f, s): {} for f in folds_wanted for s in SEEDS}
    for key, name in CONFIGS:
        base, nt, pooled, folds, pre = config_setup(name)
        print(f"\n{name}: {len(pooled)} patients, num_tokens={nt}", flush=True)
        fmap = map_files(pooled, P)
        for fold in folds_wanted:
            test_idx = folds[fold][1][:limit] if limit else folds[fold][1]
            items = []
            for i in test_idx:
                fpath = pooled[i][0]
                data = torch.load(fpath, weights_only=False)
                seq, pos = A.tokenise(data, nt, pre)
                items.append((fmap[fpath], data[:-1], seq[:-1], pos[:-1]))   # training uses [:, :-1]
            for seed in SEEDS:
                model, head = load_model(name, fold, seed, nt)
                J, OV, PR, TR = [], [], [], []
                for pid, data, seq, pos in items:
                    pred, v = rollout_and_pred(model, head, seq, pos)
                    ov = overlap(heatmap(v, data), P[pid]['masks'])
                    if ov is None:
                        continue
                    J.append(pid); OV.append(ov); PR.append(pred); TR.append(P[pid]['label'])
                results[(fold, seed)][key] = dict(j=np.array(J), ov=np.array(OV),
                                                  pred=np.array(PR), true=np.array(TR))
                print(f"  fold {fold} seed {seed}: n={len(J)}, mean overlap {np.mean(OV):.4f}, "
                      f"acc {np.mean(np.array(PR) == np.array(TR)):.3f}", flush=True)
                del model, head
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
    for (fold, seed), r in results.items():
        out = os.path.join(OUTPUT_DIR, f"overlap_preds_acdc_fold{fold}_seed{seed}.npz")
        np.savez(out, **{f"{k}_{f}": r[k][f] for k in r for f in ('j', 'ov', 'pred', 'true')})
        print("saved", out)

def summary():
    runs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "overlap_preds_acdc_fold*_seed*.npz")))
    print(f"{len(runs)} runs found (expect {N_FOLDS * len(SEEDS)})")
    means = {k: [] for k, _ in CONFIGS}; pooled = {k: [] for k, _ in CONFIGS}
    for f in runs:
        z = np.load(f)
        for k, _ in CONFIGS:
            means[k].append(z[f"{k}_ov"].mean()); pooled[k].append(z[f"{k}_ov"])
    means = {k: np.array(v) for k, v in means.items()}
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    print(f"\nGT-REGION OVERLAP ({len(runs)} runs, n={len(pooled['original'])} pooled)")
    for k, name in CONFIGS:
        print(f"  {name:<12} per-run {means[k].mean():.4f} ± {means[k].std():.4f}   "
              f"pooled {pooled[k].mean():.4f} ± {pooled[k].std():.4f}")
    print("\nSignificance (independent t-test):")
    for a, b in [('original', '1024'), ('original', '2048'), ('1024', '2048')]:
        t1, p1 = ttest_ind(means[a], means[b]); t2, p2 = ttest_ind(pooled[a], pooled[b])
        print(f"  {a} vs {b}: per-run t={t1:.4f}, p={p1:.4g} | pooled t={t2:.4f}, p={p2:.4g}")
    np.savez(os.path.join(OUTPUT_DIR, "overlap_summary_acdc_allruns.npz"),
             **{f"means_{k}": means[k] for k in means}, **{f"pooled_{k}": pooled[k] for k in pooled})

# ── 4. figure for one patient (used by rescue_failure_acdc.py) ─────────────────
def plot_final_layer_comparison(pid, fold, seed, cfg_key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P = build_patients(); p = P[pid]
    k = min(p['masks'])                                   # ED (first GT frame on the grid)
    m = p['masks'][k]
    z = int(np.argmax(m.reshape(-1, D).sum(0)))           # slice with the larger heart area
    panels = []
    for key, name in (('original', 'Original'), (cfg_key, dict(CONFIGS)[cfg_key])):
        base, nt, pooled, _, pre = config_setup(name)
        fmap = map_files(pooled, P)
        fpath = next(f for f, q in fmap.items() if q == pid)
        data = torch.load(fpath, weights_only=False)
        seq, pos = A.tokenise(data, nt, pre)
        model, head = load_model(name, fold, seed, nt)
        pred, v = rollout_and_pred(model, head, seq[:-1], pos[:-1])
        hm = heatmap(v, data[:-1])[:, :, z, k]
        # display only: rescale the shown slice to [0, 1], as Derma/Breast do for their
        # 2D map (overlap values are computed on the whole volume and are unchanged)
        hm = hm - hm.min(); hm = hm / hm.max() if hm.max() > 0 else hm
        hm = np.where(hm >= np.percentile(hm, 85), hm, 0)  # same display threshold as Derma/Breast
        panels.append((name, pred, hm))
    img = p['vol'][:, :, z, k]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original Cardiac MRI (ED, slice {z})\nTrue: {CLASS_NAMES[p['label']]}")
    for ax, (name, pred, hm) in zip(axes[1:], panels):
        ax.imshow(img, cmap='gray', extent=[0, W, H, 0])
        ax.imshow(hm, cmap='inferno', alpha=0.45, extent=[0, W, H, 0], interpolation='bilinear')
        ok = 'correct' if pred == p['label'] else 'wrong'
        ax.set_title(f"{name} Rollout Overlay (Layer {A.NUM_LAYERS})\nPred: {CLASS_NAMES[pred]} ({ok})")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"final_layer_overlay_acdc_patient{pid:03d}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print("Saved:", out)
    return out

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    print("Device:", DEVICE, flush=True)
    if arg == 'summary':
        summary()
    else:
        run_folds(list(range(N_FOLDS)) if arg == 'all' else [int(arg)], limit)
