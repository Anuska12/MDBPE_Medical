# -*- coding: utf-8 -*-
# organmnist3d_attention_rollout.py
#
# Attention rollout + foreground-mask overlap on OrganMNIST3D, same method as
# the DermaMNIST / BreastMNIST scripts:
#   - hook-based extraction on the real nn.TransformerEncoderLayer forward
#   - MAX over heads, 0.5*A + 0.5*I, row-normalised, multiplied through the layers
#   - CLS row, skipping CLS (0) and SOS (1), projected onto each token's anchor voxel
#   - Gaussian smoothing (sigma=1) + min-max normalisation, overlap = share of the
#     attention mass inside the Otsu mask
#
# What is adapted to 3D (state these in the thesis):
#   1. Volumes are 28x28x28; smoothing is 3D (sigma=1 voxel in x, y, z) and the
#      Otsu mask is computed on the whole volume.
#   2. OrganMNIST3D has no lesion. The mask is the Otsu FOREGROUND of the CT
#      volume (MASK_SIDE below). Check it visually first:
#          python3 -u organmnist3d_attention_rollout.py masks
#   3. The volume is rebuilt from the Original config's test token file (every
#      voxel is a base token = its 0-255 intensity), so the mask is computed on
#      exactly what the model sees. Every MDBPE test file is matched to its
#      Original file by its base-token voxels, so configs are paired by volume,
#      not by file number. Image id j = Original test file index.
#   4. Only the CLS row of the rollout is needed, so it is computed as a row
#      vector (identical to row 0 of the full product, without the L^3 matmul),
#      and each layer's attention matrix is formed 256 rows at a time from that
#      layer's own query/key projections (identical values; the full 21,952 x
#      21,952 x 8-head matrix would not fit on a 20 GB GPU slice).
#
# Forward passes use bf16 autocast on GPU, exactly like the classifier's evaluation
# loop, so predictions match the reported results (checked against the checkpoints).
#
#
# usage (GPU job; Original attention maps are ~2 GB per layer):
#   python3 -u organmnist3d_attention_rollout.py masks          # save mask check figure, then stop
#   python3 -u organmnist3d_attention_rollout.py 0              # seed 0 (skips seeds already saved)
#   python3 -u organmnist3d_attention_rollout.py summary        # overlap table + t-tests over ALL saved seeds
#   add --force to recompute an existing seed, --limit N for a quick test on N volumes
# Rescue / failure cases: rescue_failure_organ.py all 1024

import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from scipy.stats import ttest_ind

# ── EDIT THESE ──────────────────────────────────────────────────────────────────
OUT = "your/output/path"
BASE_PATH_ORIGINAL = f"{OUT}/mdbpe_output_organmnist3d_original"
BASE_PATH_512      = f"{OUT}/mdbpe_output_organmnist3d_512"
BASE_PATH_1024     = f"{OUT}/mdbpe_output_organmnist3d_1024"
SAVE_PATH_ORIGINAL = SAVE_PATH_512 = SAVE_PATH_1024 = \
    "your/path/to/mdbpe_results/classification_organmnist3d"   # folder with the best_ckpt_*.pt you reported
OUTPUT_DIR = os.path.join(SAVE_PATH_1024, "attention_rollout")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_NAME_ORIGINAL = "Original Sequence"
RUN_NAME_512      = "512 Extra Tokens"
RUN_NAME_1024     = "1024 Extra Tokens"
NUM_TOKENS_ORIGINAL = 256
NUM_TOKENS_512      = 768
NUM_TOKENS_1024     = 1280
BASE_TOKENS = 256
FILE_MODULO = 128
TARGET      = 28
NUM_HEADS   = 8
MASK_SIDE   = 'bright'   # foreground = voxels ABOVE the Otsu threshold (tissue); set after the mask check
USE_NPE, USE_IPE = True, True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["liver", "kidney-right", "kidney-left", "femur-right", "femur-left", "bladder",
               "heart", "lung-right", "lung-left", "spleen", "pancreas"]
CONFIGS = [
    ('original', BASE_PATH_ORIGINAL, SAVE_PATH_ORIGINAL, RUN_NAME_ORIGINAL, NUM_TOKENS_ORIGINAL),
    ('512',      BASE_PATH_512,      SAVE_PATH_512,      RUN_NAME_512,      NUM_TOKENS_512),
    ('1024',     BASE_PATH_1024,     SAVE_PATH_1024,     RUN_NAME_1024,     NUM_TOKENS_1024),
]

# ── architecture (identical to the OrganMNIST3D classifier) ────────────────────
DIMENSIONS = 512
NUM_LAYERS = 4

def ckpt_path(save_path, run_name, seed):
    return os.path.join(save_path, f"best_ckpt_{run_name.replace(' ', '_')}_seed{seed}.pt")

# ── positional encoding (copied from the OrganMNIST3D classifier) ──────────────
def grid_positional_encoding_3d(embed_dims):
    grid = torch.ones(TARGET, TARGET, TARGET, 3)
    for x in range(TARGET): grid[x, :, :, 0] = x / TARGET
    for y in range(TARGET): grid[:, y, :, 1] = y / TARGET
    for z in range(TARGET): grid[:, :, z, 2] = z / TARGET
    rets = []
    for i in range(embed_dims):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * grid))
    return torch.cat(rets, -1)

import math
# Compute embed_dims via ceiling to guarantee enough elements, then slice exactly
half_embed_dims    = math.ceil((DIMENSIONS / 2) / 6)   # enough for half slot
quarter_embed_dims = math.ceil((DIMENSIONS / 4) / 6)   # enough for quarter slot

positional_encodings_half_full    = grid_positional_encoding_3d(half_embed_dims)
positional_encodings_quarter_full = grid_positional_encoding_3d(quarter_embed_dims)

# Slice down to exact required size
positional_encodings_half    = positional_encodings_half_full[..., :int(DIMENSIONS/2)]
positional_encodings_quarter = positional_encodings_quarter_full[..., :int(DIMENSIONS/4)]

def get_pre_loaded_tokens(base_path, num_tokens):
    pre = []
    for i in range(BASE_TOKENS, num_tokens):
        p = os.path.join(base_path, "tokenshapes", f"{i}.dat")
        if os.path.exists(p):
            pre.append(torch.load(p, weights_only=False))
    return pre

def compute_positional_encoding_3d(x, y, z, tokenClass, n_x, n_y, n_z,
                                    pre_loaded_tokens, num_tokens):
    pe = torch.zeros(DIMENSIONS)
    pe[0:int(DIMENSIONS/4)] = positional_encodings_quarter[x][y][z]
    if n_x is not None and USE_NPE:
        pe[int(DIMENSIONS/4):int(DIMENSIONS/2)] = positional_encodings_quarter[n_x][n_y][n_z]
    if tokenClass < BASE_TOKENS and USE_IPE:
        pe[int(DIMENSIONS/2):] = positional_encodings_half[x][y][z]
    elif USE_IPE and tokenClass - BASE_TOKENS < len(pre_loaded_tokens):
        # tokenshape stored as [crop, offset_x, offset_y, offset_z] for 3D
        tokenshape = pre_loaded_tokens[tokenClass - BASE_TOKENS][0]
        offset_x   = pre_loaded_tokens[tokenClass - BASE_TOKENS][1]
        offset_y   = pre_loaded_tokens[tokenClass - BASE_TOKENS][2] if len(pre_loaded_tokens[tokenClass - BASE_TOKENS]) > 2 else 0
        offset_z   = pre_loaded_tokens[tokenClass - BASE_TOKENS][3] if len(pre_loaded_tokens[tokenClass - BASE_TOKENS]) > 3 else 0
        indices    = torch.nonzero(tokenshape != -1)
        indices[:, 0] += x + offset_x
        indices[:, 1] += y + offset_y
        indices[:, 2] += z + offset_z
        valid = ((indices[:,0]>=0)&(indices[:,0]<TARGET)&
                 (indices[:,1]>=0)&(indices[:,1]<TARGET)&
                 (indices[:,2]>=0)&(indices[:,2]<TARGET))
        indices = indices[valid]
        if indices.size(0) > 0:
            x_idx, y_idx, z_idx = indices[:,0].long(), indices[:,1].long(), indices[:,2].long()
            full_area = positional_encodings_half[x_idx, y_idx, z_idx, :]
            pe[int(DIMENSIONS/2):] = full_area.sum(0) #changed from sum
    return pe


def tokenise(data, num_tokens, pre_loaded_tokens):
    """Body copied from TokenSequenceDataset.tokenise in the classifier
    (self.num_tokens -> num_tokens, self.pre_loaded_tokens -> pre_loaded_tokens)."""
    SOS_token = num_tokens
    EOS_token = num_tokens + 1
    actual_len = len(data)
    seq_len    = actual_len + 1

    sequence = torch.ones(seq_len) * EOS_token
    sequence[0] = SOS_token
    positional_encoding = torch.zeros(seq_len, DIMENSIONS)

    for index in range(actual_len):
        tokenClass = data[index][0]
        x, y, z    = data[index][1], data[index][2], data[index][3]
        n_x, n_y, n_z = None, None, None
        if index + 1 < actual_len:
            n_x, n_y, n_z = data[index+1][1], data[index+1][2], data[index+1][3]
        positional_encoding[index+1] = compute_positional_encoding_3d(
            x, y, z, tokenClass, n_x, n_y, n_z,
            pre_loaded_tokens, num_tokens)
        sequence[index+1] = tokenClass

    return sequence.long(), positional_encoding


# ── Model (copied from the OrganMNIST3D classifier) ────────────────────────────
class TransformerDecoderModel(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.layer = nn.TransformerEncoderLayer(
            d_model=DIMENSIONS, nhead=NUM_HEADS, dim_feedforward=1024, #was 512,
            batch_first=True, norm_first=True, dropout=0.3) #was 0.1
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=NUM_LAYERS)
        self.embedding = nn.Embedding(num_tokens + 2, DIMENSIONS)
        self.linear = nn.Linear(DIMENSIONS, num_tokens + 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIMENSIONS))

    def get_features(self, tokens, pos_emb, padding_mask=None):
        x   = self.embedding(tokens) + pos_emb
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        if padding_mask is not None:
            cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)


class ClassificationHead(nn.Module):
    def __init__(self, dimensions=DIMENSIONS, num_classes=11):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dimensions), nn.Linear(dimensions, 128),
            nn.ReLU(), nn.Linear(128, num_classes))
    def forward(self, features): return self.classifier(features[:, 0])

def load_model_and_data(base_path, save_path, run_name, seed, num_tokens):
    ck = torch.load(ckpt_path(save_path, run_name, seed), weights_only=False, map_location='cpu')
    model = TransformerDecoderModel(num_tokens); head = ClassificationHead()
    model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
    return model.to(DEVICE).eval(), head.to(DEVICE).eval(), get_pre_loaded_tokens(base_path, num_tokens), ck

# ── rollout (hooks as in the verified Derma script; CLS row only) ──────────────
ROW_CHUNK = 256   # attention rows computed at a time (memory ~ ROW_CHUNK x L x heads)

def register_input_hooks(model):
    """Capture the input of every self-attention block (query = key = value =
    norm1(x), since norm_first=True). The model itself runs unchanged."""
    inputs, handles = [], []
    for layer in model.transformer.layers:
        def pre_hook(module, args, kwargs):
            inputs.append(args[0].detach())
        handles.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
    return inputs, handles

@torch.no_grad()
def attention_rows(mha, x, rows):
    """Rows `rows` of this layer's head-wise attention weights, MAX over heads.
    Same computation as nn.MultiheadAttention (in_proj, 1/sqrt(head_dim),
    softmax); only these rows are formed, so memory stays small."""
    x = x[0].float()                                         # (L, D)
    D = x.size(-1); H = mha.num_heads; hd = D // H
    W, b = mha.in_proj_weight.float(), mha.in_proj_bias.float()
    q = (x[rows] @ W[:D].T + b[:D]).view(-1, H, hd).transpose(0, 1)       # (H, r, hd)
    k = (x @ W[D:2*D].T + b[D:2*D]).view(-1, H, hd).transpose(0, 1)        # (H, L, hd)
    a = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(hd), dim=-1)     # (H, r, L)
    return a.max(dim=0).values                                             # MAX across heads -> (r, L)

@torch.no_grad()
def rollout_and_pred(model, head, seq, pos):
    inputs, handles = register_input_hooks(model)
    # same precision as the classifier's evaluation loop (bf16 autocast on GPU),
    # so predictions match the reported results exactly
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
        feats = model.get_features(seq.unsqueeze(0).to(DEVICE), pos.unsqueeze(0).to(DEVICE))
        pred = int(head(feats).argmax(dim=1).item())
    for h in handles: h.remove()
    if len(inputs) != len(model.transformer.layers):
        raise RuntimeError("Attention inputs not captured for every layer.")
    L = inputs[0].size(1)
    v = torch.zeros(L, device=inputs[0].device); v[0] = 1.0
    # e0^T A_L ... A_1  ==  row 0 of the full rollout (0.5*A + 0.5*I, row-normalised),
    # with each A_l built row-chunk by row-chunk
    for layer, x in zip(reversed(model.transformer.layers), reversed(inputs)):
        v_new = torch.zeros_like(v)
        for start in range(0, L, ROW_CHUNK):
            rows = torch.arange(start, min(start + ROW_CHUNK, L), device=v.device)
            A = attention_rows(layer.self_attn, x, rows)                    # (r, L)
            A.mul_(0.5); A[torch.arange(len(rows)), rows] += 0.5
            A.div_(A.sum(dim=-1, keepdim=True))
            v_new += v[rows] @ A
        v = v_new
    return pred, v[2:].cpu().numpy()            # skip CLS (0) and SOS (1)

def heatmap(cls_vec, data):
    """mean attention per anchor voxel, 3D Gaussian sigma=1, min-max."""
    tok = np.asarray([d[1:4] for d in data[:len(cls_vec)]], dtype=int)
    hm = np.zeros((TARGET,) * 3); cnt = np.zeros((TARGET,) * 3)
    np.add.at(hm,  (tok[:, 0], tok[:, 1], tok[:, 2]), cls_vec[:len(tok)])
    np.add.at(cnt, (tok[:, 0], tok[:, 1], tok[:, 2]), 1)
    hm = gaussian_filter(hm / np.maximum(cnt, 1), sigma=1.0)
    hm = hm - hm.min()
    return hm / hm.max() if hm.max() > 0 else hm

# ── volumes + masks + config pairing ───────────────────────────────────────────
def test_files(base_path):
    out = {}
    for i in range(FILE_MODULO):
        d = os.path.join(base_path, "tokensequences_test", str(i))
        if os.path.isdir(d):
            for f in os.listdir(d):
                out[int(f[:-4])] = os.path.join(d, f)
    return out

def volume_from_original(data):
    vol = np.zeros((TARGET,) * 3, dtype=np.int16)
    for c, x, y, z in data:
        vol[x, y, z] = c
    return vol

def get_mask(vol):
    th = threshold_otsu(vol)
    return vol > th if MASK_SIDE == 'bright' else vol < th

_VOLS = None
def original_volumes():
    """{j: volume} from the Original test files (j = Original test file index)."""
    global _VOLS
    if _VOLS is None:
        _VOLS = {j: volume_from_original(torch.load(f, weights_only=False))
                 for j, f in sorted(test_files(BASE_PATH_ORIGINAL).items())}
    return _VOLS

_MAPS = {}
def file_to_j(base_path):
    """{file idx in this config: j} matched on base-token voxels."""
    if base_path in _MAPS:
        return _MAPS[base_path]
    vols = original_volumes()
    if base_path == BASE_PATH_ORIGINAL:
        _MAPS[base_path] = {j: j for j in vols}; return _MAPS[base_path]
    js = np.array(sorted(vols)); V = np.stack([vols[j] for j in js])
    m, moved = {}, 0
    for k, f in sorted(test_files(base_path).items()):
        data = torch.load(f, weights_only=False)
        b = np.asarray([d for d in data if d[0] < BASE_TOKENS], dtype=int)
        if k in vols and (vols[k][b[:, 1], b[:, 2], b[:, 3]] == b[:, 0]).mean() > 0.99:
            m[k] = k; continue
        score = (V[:, b[:, 1], b[:, 2], b[:, 3]] == b[:, 0]).mean(1)
        if score.max() < 0.99:
            raise RuntimeError(f"{f}: no Original volume matches (best {score.max():.3f})")
        m[k] = int(js[score.argmax()]); moved += 1
    if len(set(m.values())) != len(m):
        raise RuntimeError(f"{base_path}: two files matched the same volume")
    print(f"  {os.path.basename(base_path)}: {len(m)} test files matched to volumes "
          f"({moved} not at the same file number)", flush=True)
    _MAPS[base_path] = m
    return m

def recovered_labels(base_path):
    return torch.load(os.path.join(base_path, "recovered_labels_test.pt"), weights_only=False)

# ── per-seed computation ───────────────────────────────────────────────────────
def compute_overlap_and_preds(base_path, save_path, run_name, seed, num_tokens, tag, items, limit=None):
    model, head, pre, ck = load_model_and_data(base_path, save_path, run_name, seed, num_tokens)
    vols = original_volumes(); labels = recovered_labels(base_path)
    J, OV, PR, TR, K = [], [], [], [], []
    print(f"\nComputing overlap + preds across {len(items)} volumes for {tag}...", flush=True)
    for n, (k, j, data, seq, pos) in enumerate(items[:limit] if limit else items):
        pred, v = rollout_and_pred(model, head, seq, pos)
        try:
            mask = get_mask(vols[j])
        except ValueError:
            continue
        hm = heatmap(v, data)
        if hm.sum() <= 0:
            continue
        J.append(j); OV.append(float(hm[mask].sum() / hm.sum())); PR.append(pred); TR.append(int(labels[k])); K.append(k)
        if (n + 1) % 100 == 0:
            print(f"  Processed {n+1}/{len(items)}", flush=True)
    J, OV, PR, TR = map(np.array, (J, OV, PR, TR))
    if 'all_probs' in ck:   # same order as the classifier's test DataLoader
        tdir = os.path.join(base_path, "tokensequences_test")
        order = [int(f[:-4]) for i in range(FILE_MODULO) if os.path.isdir(os.path.join(tdir, str(i)))
                 for f in sorted(os.listdir(os.path.join(tdir, str(i))))]
        ck_pred = dict(zip(order, np.asarray(ck['all_probs']).argmax(1)))
        agree = np.mean([ck_pred[k] == p for k, p in zip(K, PR)])
        print(f"  check: predictions agree with the checkpoint's saved test predictions for {100*agree:.1f}% of volumes"
              + ("" if agree > 0.97 else "   <-- LOW: model/tokenisation may not match training"), flush=True)
    print(f"{tag}: mean overlap = {OV.mean():.4f} +/- {OV.std():.4f}, N={len(OV)}; "
          f"accuracy = {(PR == TR).mean():.4f}", flush=True)
    del model, head
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return J, OV, PR, TR

def prepare_items(base_path, num_tokens):
    """tokenise every test file once per config (CPU), shared across seeds."""
    pre = get_pre_loaded_tokens(base_path, num_tokens); fmap = file_to_j(base_path)
    items = []
    for k, f in sorted(test_files(base_path).items()):
        data = torch.load(f, weights_only=False)
        seq, pos = tokenise(data, num_tokens, pre)
        items.append((k, fmap[k], data[:-1], seq[:-1], pos[:-1]))   # training uses [:, :-1]
    return items

# ── figure (used by rescue_failure_organ.py) ───────────────────────────────────
def plot_final_layer_comparison(j,
                                base_path_m, save_path_m, run_name_m, seed_m, num_tokens_m,
                                base_path_orig, save_path_orig, run_name_orig, seed_orig, num_tokens_orig,
                                label="MDBPE 1024"):
    vol = original_volumes()[j]; mask = get_mask(vol)
    z = int(np.argmax(mask.sum(axis=(0, 1))))         # axial slice with the largest foreground area
    panels = []
    for bp, sp, rn, sd, nt in ((base_path_orig, save_path_orig, run_name_orig, seed_orig, num_tokens_orig),
                               (base_path_m, save_path_m, run_name_m, seed_m, num_tokens_m)):
        k = {jj: kk for kk, jj in file_to_j(bp).items()}[j]
        data = torch.load(test_files(bp)[k], weights_only=False)
        model, head, pre, _ = load_model_and_data(bp, sp, rn, sd, nt)
        seq, pos = tokenise(data, nt, pre)
        pred, v = rollout_and_pred(model, head, seq[:-1], pos[:-1])
        hm = heatmap(v, data[:-1])[:, :, z]
        # display only: rescale the shown slice to [0, 1], as Derma/Breast do for their
        # 2D map (overlap values are computed on the whole volume and are unchanged)
        hm = hm - hm.min(); hm = hm / hm.max() if hm.max() > 0 else hm
        hm = np.where(hm >= np.percentile(hm, 85), hm, 0)    # same display threshold as Derma/Breast
        panels.append((pred, hm, int(recovered_labels(bp)[k])))
    true = panels[0][2]
    img = vol[:, :, z]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f"Original CT Volume (axial slice {z})\nTrue: {CLASS_NAMES[true]}")
    for ax, name, (pred, hm, _) in zip(axes[1:], ("Original", label), panels):
        ax.imshow(img, cmap='gray', extent=[0, TARGET, TARGET, 0])
        ax.imshow(hm, cmap='inferno', alpha=0.45, extent=[0, TARGET, TARGET, 0], interpolation='bilinear')
        ax.set_title(f"{name} Rollout Overlay (Layer {NUM_LAYERS})\n"
                     f"Pred: {CLASS_NAMES[pred]} ({'correct' if pred == true else 'wrong'})")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"final_layer_overlay_organ_img{j}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {out}")

def mask_check():
    vols = original_volumes(); labels = recovered_labels(BASE_PATH_ORIGINAL)
    js = sorted(vols)[::max(1, len(vols) // 8)][:8]
    fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
    for c, j in enumerate(js):
        v = vols[j]; th = threshold_otsu(v); z = TARGET // 2
        for r, (m, nm) in enumerate(((v > th, "bright (> Otsu)"), (v < th, "dark (< Otsu)"))):
            axes[r, c].imshow(v[:, :, z], cmap='gray')
            axes[r, c].contour(m[:, :, z], levels=[0.5], colors='cyan', linewidths=1)
            axes[r, c].set_title(f"{CLASS_NAMES[int(labels[j])]}\n{nm}", fontsize=9); axes[r, c].axis('off')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "mask_check_organ.png")
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print("Saved:", out, "-- set MASK_SIDE to the row whose cyan outline covers the organ/tissue")

def summary():
    npz = lambda s: os.path.join(OUTPUT_DIR, f"overlap_preds_organ_seed{s}.npz")
    seeds = sorted(int(os.path.basename(f)[len("overlap_preds_organ_seed"):-4])
                   for f in glob.glob(os.path.join(OUTPUT_DIR, "overlap_preds_organ_seed*.npz")))
    if not seeds:
        print(f"\nNo overlap_preds_organ_seed*.npz in {OUTPUT_DIR} yet -- run a seed first, e.g.\n"
              f"  sbatch run_organ_rollout.sh 0"); return
    per = {k: [] for k, *_ in CONFIGS}
    for s in seeds:
        z = np.load(npz(s))
        for k, *_ in CONFIGS:
            per[k].append(z[f"{k}_ov"])
    means = {k: np.array([v.mean() for v in per[k]]) for k in per}
    pooled = {k: np.concatenate(per[k]) for k in per}
    print(f"\n{'='*66}\nFOREGROUND-MASK OVERLAP (seeds {seeds}, n={len(pooled['original'])} pooled)\n{'='*66}")
    for k, _, _, rn, _ in CONFIGS:
        print(f"{rn:<20} per-seed {means[k].mean():.4f} +/- {means[k].std():.4f}   "
              f"pooled {pooled[k].mean():.4f} +/- {pooled[k].std():.4f}")
    print("\nPer-seed means:\n  seed  " + "".join(f"{rn:>22}" for _, _, _, rn, _ in CONFIGS))
    for i, s in enumerate(seeds):
        print(f"  {s:<6}" + "".join(f"{means[k][i]:>22.4f}" for k, *_ in CONFIGS))
    print("\nSignificance (independent t-test):")
    for a, b in [('original', '512'), ('original', '1024'), ('512', '1024')]:
        t1, p1 = ttest_ind(means[a], means[b]); t2, p2 = ttest_ind(pooled[a], pooled[b])
        print(f"  {a} vs {b}: per-seed means t={t1:.4f}, p={p1:.4g} | pooled t={t2:.4f}, p={p2:.4g}")
    np.savez(os.path.join(OUTPUT_DIR, "overlap_summary_organ_allseeds.npz"),
             **{f"means_{k}": means[k] for k in means}, **{f"pooled_{k}": pooled[k] for k in pooled})
    print("\nRescue / failure cases: python3 -u rescue_failure_organ.py all 1024")

if __name__ == "__main__":
    argv = sys.argv[1:]
    LIMIT = int(argv[argv.index('--limit') + 1]) if '--limit' in argv else None
    if '--limit' in argv:
        del argv[argv.index('--limit'):argv.index('--limit') + 2]
    FORCE = '--force' in argv
    args = [a for a in argv if not a.startswith('--')]
    print("Device:", DEVICE, flush=True)
    if not args:
        print("usage: organmnist3d_attention_rollout.py masks | <seed> [<seed> ...] | summary  [--force] [--limit N]")
        sys.exit(1)
    if args[0] == 'masks':
        mask_check(); sys.exit(0)
    if args and args[0] != 'summary':
        seeds = [int(a) for a in args]
        todo = [s for s in seeds if FORCE or not os.path.exists(
            os.path.join(OUTPUT_DIR, f"overlap_preds_organ_seed{s}.npz"))]
        for s in sorted(set(seeds) - set(todo)):
            print(f"seed {s}: already saved -- skipping (use --force to recompute)")
        if todo:
            res = {s: {} for s in todo}
            for key, bp, sp, rn, nt in CONFIGS:
                print(f"\n{rn}: tokenising test volumes (once for all seeds)...", flush=True)
                items = prepare_items(bp, nt)
                for s in todo:
                    J, OV, PR, TR = compute_overlap_and_preds(bp, sp, rn, s, nt, f"{rn} (seed {s})", items, LIMIT)
                    res[s][key] = dict(j=J, ov=OV, pred=PR, true=TR)
                del items
            for s in todo:
                out = os.path.join(OUTPUT_DIR, f"overlap_preds_organ_seed{s}.npz")
                np.savez(out, **{f"{k}_{f}": res[s][k][f] for k in res[s] for f in ('j', 'ov', 'pred', 'true')})
                print("saved", out)
    summary()
