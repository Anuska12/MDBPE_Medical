# -*- coding: utf-8 -*-
# attention_rollout_full_breast.py
# methodology (attention_rollout_full_derma.py): hook-based extraction on
# the real nn.TransformerEncoderLayer forward (not a manual reimplementation),
# MAX pooling across heads (not mean), correct CLS/SOS slicing ([0,0,2:]),
# and Gaussian-smoothed, min-max normalized final-layer rollout before
# computing lesion mask overlap.
# Compares Original Sequence vs +256 and +512 Extra Tokens on BreastMNIST.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import medmnist
from medmnist import INFO
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from scipy.stats import ttest_ind

# ── EDIT THESE ──────────────────────────────────────────────────────────────────
BASE_PATH_256      = "your/path/256"
BASE_PATH_512      = "your/path/512"
BASE_PATH_ORIGINAL = "your/path/original"
SAVE_PATH_256      = "your/path/results/256"
SAVE_PATH_512      = "your/path/results/512"
SAVE_PATH_ORIGINAL = "your/path/results/original"
OUTPUT_DIR          = "your/path/attention_rollout"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_NAME_256      = "256 Extra Tokens"
RUN_NAME_512      = "512 Extra Tokens"
RUN_NAME_ORIGINAL = "Original Sequence"

SEEDS = [0, 1, 2, 3]  

NUM_TOKENS_256      = 512
NUM_TOKENS_512      = 768
NUM_TOKENS_ORIGINAL = 256
BASE_TOKENS = 256

DEVICE = torch.device('cpu')
DIMENSIONS  = 512
NUM_HEADS   = 8
NUM_LAYERS  = 4
FILE_MODULO = 128
USE_NPE, USE_IPE = True, True

# ── Data / metadata ──────────────────────────────────────────────────────────────
def load_metadata(base_path):
    return torch.load(os.path.join(base_path, "metadata.pt"), weights_only=False)

meta = load_metadata(BASE_PATH_512)
WIDTH, HEIGHT = meta["WIDTH"], meta["HEIGHT"]
print(f"Image size: {WIDTH}x{HEIGHT}")

data_flag = 'breastmnist'
info      = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
transform = transforms.Compose([transforms.Resize((WIDTH, HEIGHT)), transforms.ToTensor()])
test_dataset = DataClass(split='test', download=True, transform=transform)
test_labels  = [int(test_dataset[i][1][0]) for i in range(len(test_dataset))]
CLASS_NAMES  = list(info['label'].values())

# ── Recovered file->dataset-image maps. File k is NOT dataset image k: each
#    config was tokenized in its own order, so labels, lesion masks and
#    cross-config pairing must all go through these maps.
IDX_MAP = {}      # base_path -> {file_idx: dataset_j}
REV_MAP = {}      # base_path -> {dataset_j: file_idx}
for _bp in (BASE_PATH_ORIGINAL, BASE_PATH_256, BASE_PATH_512):
    _m = torch.load(os.path.join(_bp, "recovered_index_test.pt"), weights_only=False)
    IDX_MAP[_bp] = _m
    REV_MAP[_bp] = {j: k for k, j in _m.items()}
    print(f"Loaded index map: {os.path.basename(_bp)} ({len(_m)} files)")  # e.g. ['malignant', 'normal, benign']

# ── Positional encoding (identical to Derma / your classifier) ──────────────────
def grid_positional_encoding(embed_dims):
    grid = torch.ones(WIDTH, HEIGHT, 2)
    for x in range(WIDTH): grid[x, :, 0] = x / WIDTH
    for y in range(HEIGHT): grid[:, y, 1] = y / HEIGHT
    rets = []
    for i in range(embed_dims):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * grid))
    return torch.cat(rets, -1)

positional_encodings_half    = grid_positional_encoding(int(DIMENSIONS / 8))
positional_encodings_quarter = grid_positional_encoding(int(DIMENSIONS / 16))

def get_pre_loaded_tokens(base_path, num_tokens):
    pre_loaded = []
    for i in range(BASE_TOKENS, num_tokens):
        path = os.path.join(base_path, "tokenshapes", str(i) + ".dat")
        if os.path.exists(path):
            pre_loaded.append(torch.load(path, weights_only=False))
    return pre_loaded

def compute_positional_encoding(x, y, tokenClass, n_x, n_y, pre_loaded_tokens, num_tokens):
    pe = torch.zeros(DIMENSIONS)
    pe[0:int(DIMENSIONS/4)] = positional_encodings_quarter[x][y]
    if n_x is not None and USE_NPE:
        pe[int(DIMENSIONS/4):int(DIMENSIONS/2)] = positional_encodings_quarter[n_x][n_y]
    if tokenClass < BASE_TOKENS and USE_IPE:
        pe[int(DIMENSIONS/2):] = positional_encodings_half[x][y]
    elif USE_IPE and tokenClass - BASE_TOKENS < len(pre_loaded_tokens):
        tokenshape = pre_loaded_tokens[tokenClass - BASE_TOKENS][0]
        offset_x   = pre_loaded_tokens[tokenClass - BASE_TOKENS][1]
        indices    = torch.nonzero(tokenshape != -1)
        indices[:, 0] += x + offset_x
        indices[:, 1] += y
        valid = ((indices[:,0]>=0)&(indices[:,0]<WIDTH)&(indices[:,1]>=0)&(indices[:,1]<HEIGHT))
        indices = indices[valid]
        if indices.size(0) > 0:
            full_area = positional_encodings_half[indices[:,0].long(), indices[:,1].long(), :]
            pe[int(DIMENSIONS/2):] = full_area.sum(0)
    return pe

def tokenise(data, num_tokens, pre_loaded_tokens):
    SOS_token = num_tokens
    EOS_token = num_tokens + 1
    actual_len = len(data)
    seq_len = actual_len + 1
    sequence = torch.ones(seq_len) * EOS_token
    sequence[0] = SOS_token
    pos_enc = torch.zeros(seq_len, DIMENSIONS)
    positions = [(None, None)]  # SOS has no spatial position
    for index in range(actual_len):
        tokenClass = data[index][0]
        x, y = data[index][1], data[index][2]
        n_x, n_y = None, None
        if index + 1 < actual_len:
            n_x, n_y = data[index+1][1], data[index+1][2]
        pos_enc[index+1] = compute_positional_encoding(
            x, y, tokenClass, n_x, n_y, pre_loaded_tokens, num_tokens)
        sequence[index+1] = tokenClass
        positions.append((x, y))
    return sequence.long(), pos_enc, positions

# ── Model (identical architecture to your BreastMNIST classifier) ───────────────
class TransformerDecoderModel(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=DIMENSIONS, nhead=NUM_HEADS, dim_feedforward=1024,
            batch_first=True, norm_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=NUM_LAYERS)
        self.embedding = nn.Embedding(num_tokens + 2, DIMENSIONS)
        self.linear = nn.Linear(DIMENSIONS, num_tokens + 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIMENSIONS))

    def get_features(self, tokens, pos_emb, padding_mask=None):
        x = self.embedding(tokens) + pos_emb
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        if padding_mask is not None:
            cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)


class ClassificationHead(nn.Module):
    def __init__(self, dimensions=DIMENSIONS, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dimensions), nn.Linear(dimensions, 128),
            nn.ReLU(), nn.Linear(128, num_classes))
    def forward(self, features): return self.classifier(features[:, 0])


# ── Hook-based real attention extraction (verbatim from verified Derma script) ──
def register_attention_hooks(model):
    attn_maps = []
    original_forwards = []

    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple) and output[1] is not None:
                attn_weights = output[1].detach()
                attn_max = attn_weights.max(dim=1).values  # MAX across heads
                attn_maps.append(attn_max)
        return hook

    handles = []
    for layer in model.transformer.layers:
        orig_forward = layer.self_attn.forward
        original_forwards.append((layer.self_attn, orig_forward))
        def patched_forward(*args, orig_forward=orig_forward, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return orig_forward(*args, **kwargs)
        layer.self_attn.forward = patched_forward
        h = layer.self_attn.register_forward_hook(make_hook())
        handles.append(h)

    return attn_maps, handles, original_forwards


def project_to_grid(cls_attn, positions, width=WIDTH, height=HEIGHT):
    heatmap = torch.zeros(width, height)
    counts  = torch.zeros(width, height)
    for i, (x, y) in enumerate(positions[1:]):
        if x is None: continue
        if i < cls_attn.size(0):
            heatmap[x, y] += cls_attn[i].item()
            counts[x, y] += 1
    counts[counts == 0] = 1
    return (heatmap / counts).numpy()


def get_all_layer_data(model, sequence, pos_enc, positions):
    """Same as verified Derma script: raw per-layer attn matrices, raw
    (non-cumulative) CLS heatmaps, and cumulative rollout CLS heatmaps."""
    attn_maps, handles, original_forwards = register_attention_hooks(model)
    model.eval()
    with torch.no_grad():
        _ = model.get_features(sequence.unsqueeze(0), pos_enc.unsqueeze(0))
    for h in handles:
        h.remove()
    for module, orig_fwd in original_forwards:
        module.forward = orig_fwd

    if len(attn_maps) == 0:
        raise RuntimeError("No attention maps captured.")

    raw_attn_matrices = [a[0].numpy() for a in attn_maps]

    raw_cls_heatmaps = []
    for attn in attn_maps:
        cls_attn = attn[0, 0, 2:]  # skip CLS (0) and SOS (1) -- same as Derma
        raw_cls_heatmaps.append(project_to_grid(cls_attn, positions))

    result = torch.eye(attn_maps[0].size(-1)).unsqueeze(0)
    rollout_cls_heatmaps = []
    for attn in attn_maps:
        identity = torch.eye(attn.size(-1)).unsqueeze(0)
        attn_aug = 0.5 * attn + 0.5 * identity
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_aug, result)
        cls_attn = result[0, 0, 2:]
        rollout_cls_heatmaps.append(project_to_grid(cls_attn, positions))

    return raw_attn_matrices, raw_cls_heatmaps, rollout_cls_heatmaps


def load_model_and_data(base_path, save_path, run_name, seed, num_tokens):
    safe_name = run_name.replace(' ', '_')
    ckpt_path = os.path.join(save_path, f"best_ckpt_{safe_name}_seed{seed}.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = TransformerDecoderModel(num_tokens)
    head  = ClassificationHead(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt['model'])
    head.load_state_dict(ckpt['head'])
    model.eval(); head.eval()
    pre_loaded = get_pre_loaded_tokens(base_path, num_tokens)
    return model, head, pre_loaded


# ── Lesion mask -- flagged for visual verification on ultrasound images ─────────
def get_lesion_mask(img_np):
    """Boolean mask via Otsu thresholding. Same convention as verified Derma
    script (lesion = darker region). BreastMNIST is grayscale ultrasound;
    malignant masses are typically hypoechoic (darker), but VERIFY this
    visually on a few images before trusting full-test-set numbers --
    ultrasound has more texture/shadow variability than dermoscopy."""
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np
    thresh = threshold_otsu(gray)
    return gray < thresh


# ── Full-test-set overlap + predictions (adapted from verified Derma script) ────
def compute_overlap_and_preds(base_path, save_path, run_name, seed, num_tokens, tag):
    model, head, pre_loaded = load_model_and_data(base_path, save_path, run_name, seed, num_tokens)

    seq_test_dir = os.path.join(base_path, "tokensequences_test")
    files = sorted(
        [os.path.join(seq_test_dir, sub, f)
         for sub in os.listdir(seq_test_dir)
         for f in os.listdir(os.path.join(seq_test_dir, sub))],
        key=lambda p: int(os.path.basename(p).replace(".dat", "")))

    overlaps, preds, trues, indices = [], [], [], []
    print(f"\nComputing overlap + preds across {len(files)} images for {tag}...")
    for i, f in enumerate(files):
        idx = int(os.path.basename(f).replace(".dat", ""))
        j = IDX_MAP[base_path][idx]          # actual dataset image for this file
        data = torch.load(f, weights_only=False)
        seq, pos, positions = tokenise(data, num_tokens, pre_loaded)
        seq, pos, positions = seq[:-1], pos[:-1], positions[:-1]   # match training's [:, :-1]

        with torch.no_grad():
            features = model.get_features(seq.unsqueeze(0), pos.unsqueeze(0))
            pred = head(features).argmax(dim=1).item()
        true = test_labels[j]

        _, _, rollout_cls = get_all_layer_data(model, seq, pos, positions)
        final = gaussian_filter(rollout_cls[-1], sigma=1.0)
        final = final - final.min()
        if final.max() > 0:
            final = final / final.max()

        img, _ = test_dataset[j]
        img_np = img.permute(1, 2, 0).numpy()
        try:
            mask = get_lesion_mask(img_np)
        except ValueError:
            continue

        attn = final.T
        total = attn.sum()
        if total <= 0:
            continue

        overlaps.append(float(attn[mask].sum() / total))
        preds.append(int(pred))
        trues.append(int(true))
        indices.append(j)          # dataset index, so configs can be compared

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)}")

    overlaps = np.array(overlaps); indices = np.array(indices)
    preds = np.array(preds); trues = np.array(trues)
    print(f"\n{tag}: mean overlap = {overlaps.mean():.4f} +/- {overlaps.std():.4f}, N={len(overlaps)}")
    print(f"{tag}: accuracy = {(preds == trues).mean():.4f}")
    return indices, overlaps, preds, trues


# ── Final-layer rescue/failure comparison figure (adapted from verified
#    Derma script's plot_final_layer_comparison) ─────────────────────────────
def plot_final_layer_comparison(j,
                                  base_path_512, save_path_512, run_name_512, seed_512, num_tokens_512,
                                  base_path_orig, save_path_orig, run_name_orig, seed_orig, num_tokens_orig):
    """j is a DATASET image index; the corresponding file index is looked up
    separately per config, since their file orderings differ."""
    idx_512  = REV_MAP[base_path_512][j]
    idx_orig = REV_MAP[base_path_orig][j]

    model_512, head_512, pre_512 = load_model_and_data(
        base_path_512, save_path_512, run_name_512, seed_512, num_tokens_512)
    model_orig, head_orig, pre_orig = load_model_and_data(
        base_path_orig, save_path_orig, run_name_orig, seed_orig, num_tokens_orig)

    seq_512_path  = os.path.join(base_path_512,  "tokensequences_test", str(idx_512 % FILE_MODULO), f"{idx_512}.dat")
    seq_orig_path = os.path.join(base_path_orig, "tokensequences_test", str(idx_orig % FILE_MODULO), f"{idx_orig}.dat")

    data_512 = torch.load(seq_512_path, weights_only=False)
    seq_512, pos_512, positions_512 = tokenise(data_512, num_tokens_512, pre_512)
    seq_512, pos_512, positions_512 = seq_512[:-1], pos_512[:-1], positions_512[:-1]

    data_orig = torch.load(seq_orig_path, weights_only=False)
    seq_orig, pos_orig, positions_orig = tokenise(data_orig, num_tokens_orig, pre_orig)
    seq_orig, pos_orig, positions_orig = seq_orig[:-1], pos_orig[:-1], positions_orig[:-1]

    true_label = test_labels[j]

    with torch.no_grad():
        feat_512 = model_512.get_features(seq_512.unsqueeze(0), pos_512.unsqueeze(0))
        pred_512 = head_512(feat_512).argmax(dim=1).item()
        feat_orig = model_orig.get_features(seq_orig.unsqueeze(0), pos_orig.unsqueeze(0))
        pred_orig = head_orig(feat_orig).argmax(dim=1).item()

    _, _, rollout_512  = get_all_layer_data(model_512,  seq_512,  pos_512,  positions_512)
    _, _, rollout_orig = get_all_layer_data(model_orig, seq_orig, pos_orig, positions_orig)

    final_512  = gaussian_filter(rollout_512[-1],  sigma=1.0)
    final_orig = gaussian_filter(rollout_orig[-1], sigma=1.0)

    def normalize(h):
        h = h - h.min()
        if h.max() > 0:
            h = h / h.max()
        return h

    final_512_norm  = normalize(final_512)
    final_orig_norm = normalize(final_orig)

    threshold = np.percentile(final_orig_norm, 85)
    final_orig_norm = np.where(final_orig_norm >= threshold, final_orig_norm, 0)

    threshold = np.percentile(final_512_norm, 85)
    final_512_norm = np.where(final_512_norm >= threshold, final_512_norm, 0)

    img, _ = test_dataset[j]
    img_np = img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img_np, cmap='gray' if img_np.shape[-1] == 1 else None)
    axes[0].set_title(f"Original Ultrasound Image\nTrue: {CLASS_NAMES[true_label]}")
    axes[0].axis('off')

    axes[1].imshow(img_np, extent=[0, WIDTH, HEIGHT, 0], cmap='gray' if img_np.shape[-1] == 1 else None)
    axes[1].imshow(final_orig_norm.T, cmap='inferno', alpha=0.45,
               extent=[0, WIDTH, HEIGHT, 0], interpolation='bilinear')
    correct_orig = 'correct' if pred_orig == true_label else 'wrong'
    axes[1].set_title(f"Original Rollout Overlay (Layer 4)\nPred: {CLASS_NAMES[pred_orig]} ({correct_orig})")
    axes[1].axis('off')

    axes[2].imshow(img_np, extent=[0, WIDTH, HEIGHT, 0], cmap='gray' if img_np.shape[-1] == 1 else None)
    axes[2].imshow(final_512_norm.T, cmap='inferno', alpha=0.45,
               extent=[0, WIDTH, HEIGHT, 0], interpolation='bilinear')
    correct_512 = 'correct' if pred_512 == true_label else 'wrong'
    axes[2].set_title(f"MDBPE 512 Rollout Overlay (Layer 4)\nPred: {CLASS_NAMES[pred_512]} ({correct_512})")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"final_layer_overlay_breast_img{j}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    CONFIGS = [
        ('original', BASE_PATH_ORIGINAL, SAVE_PATH_ORIGINAL, RUN_NAME_ORIGINAL, NUM_TOKENS_ORIGINAL),
        ('256',      BASE_PATH_256,      SAVE_PATH_256,      RUN_NAME_256,      NUM_TOKENS_256),
        ('512',      BASE_PATH_512,      SAVE_PATH_512,      RUN_NAME_512,      NUM_TOKENS_512),
    ]

    # ── Full-test-set overlap: every config, every seed ───────────────────────
    per_seed = {k: [] for k, *_ in CONFIGS}
    for seed in SEEDS:
        for key, bp, sp, rn, nt in CONFIGS:
            j_, ov_, pred_, true_ = compute_overlap_and_preds(bp, sp, rn, seed, nt,
                                                              f"{rn} (seed {seed})")
            per_seed[key].append(dict(seed=seed, j=j_, ov=ov_, pred=pred_, true=true_,
                                      mean_overlap=ov_.mean()))
        np.savez(os.path.join(OUTPUT_DIR, f"overlap_preds_breast_seed{seed}.npz"),
                 **{f"{k}_{f}": per_seed[k][-1][f] for k, *_ in CONFIGS
                    for f in ('j', 'ov', 'pred', 'true')})

    means  = {k: np.array([r['mean_overlap'] for r in per_seed[k]]) for k, *_ in CONFIGS}
    pooled = {k: np.concatenate([r['ov'] for r in per_seed[k]])     for k, *_ in CONFIGS}

    print(f"\n{'='*66}\nLESION-MASK OVERLAP ({len(SEEDS)} seeds, n={len(pooled['original'])} pooled)\n{'='*66}")
    for key, _, _, rn, _ in CONFIGS:
        print(f"{rn:<20} per-seed {means[key].mean():.4f} +/- {means[key].std():.4f}   "
              f"pooled {pooled[key].mean():.4f} +/- {pooled[key].std():.4f}")

    print("\nPer-seed means:")
    hdr = "  seed  " + "".join(f"{rn:>22}" for _, _, _, rn, _ in CONFIGS)
    print(hdr)
    for i, seed in enumerate(SEEDS):
        row = f"  {seed:<6}" + "".join(f"{per_seed[k][i]['mean_overlap']:>22.4f}" for k, *_ in CONFIGS)
        print(row)
    print("\nPer-seed deltas vs Original:")
    for key in ('256', '512'):
        ds = [per_seed[key][i]['mean_overlap'] - per_seed['original'][i]['mean_overlap']
              for i in range(len(SEEDS))]
        print(f"  {key:<5} " + "  ".join(f"seed{SEEDS[i]}={d:+.4f}" for i, d in enumerate(ds)))

    print("\nSignificance (independent t-test):")
    for a, b in [('original', '256'), ('original', '512'), ('256', '512')]:
        t1, p1 = ttest_ind(means[a], means[b])
        t2, p2 = ttest_ind(pooled[a], pooled[b])
        print(f"  {a} vs {b}: per-seed means t={t1:.4f}, p={p1:.4g} | "
              f"pooled t={t2:.4f}, p={p2:.4g}")

    np.savez(os.path.join(OUTPUT_DIR, "overlap_summary_breast_allseeds.npz"),
             **{f"means_{k}": means[k] for k, *_ in CONFIGS},
             **{f"pooled_{k}": pooled[k] for k, *_ in CONFIGS})

    # ── Rescue / failure analysis on the representative seed ──────────────────
    REPRESENTATIVE_SEED = 1
    si = SEEDS.index(REPRESENTATIVE_SEED)
    ro = per_seed['original'][si]
    ovo = dict(zip(ro['j'], ro['ov'])); po = dict(zip(ro['j'], ro['pred']))
    tru = dict(zip(ro['j'], ro['true']))

    for key in ('256', '512'):
        rk = per_seed[key][si]
        ovk = dict(zip(rk['j'], rk['ov'])); pk = dict(zip(rk['j'], rk['pred']))
        common = sorted(set(ovk) & set(ovo))
        rescue  = [j for j in common if pk[j] == tru[j] and po[j] != tru[j]]
        failure = [j for j in common if pk[j] != tru[j] and po[j] == tru[j]]

        print(f"\n{'='*66}\nSeed {REPRESENTATIVE_SEED}: {key} vs Original -- "
              f"{len(rescue)} rescue, {len(failure)} failure (of {len(common)} images)")
        # distribution-level comparison: does overlap gain differ between the two groups?
        for nm, grp in (('rescue', rescue), ('failure', failure)):
            if grp:
                d = np.array([ovk[j] - ovo[j] for j in grp])
                print(f"  {nm}: mean overlap delta {d.mean():+.4f} +/- {d.std():.4f} (n={len(grp)})")
        if rescue and failure:
            t3, p3 = ttest_ind([ovk[j] - ovo[j] for j in rescue],
                               [ovk[j] - ovo[j] for j in failure])
            print(f"  rescue vs failure delta: t={t3:.4f}, p={p3:.4g}")

    # ── Comparison figures: Original vs +512, clearest rescue and failure ─────
    rk = per_seed['512'][si]
    ov5 = dict(zip(rk['j'], rk['ov'])); p5 = dict(zip(rk['j'], rk['pred']))
    common = sorted(set(ov5) & set(ovo))
    rescue  = [j for j in common if p5[j] == tru[j] and po[j] != tru[j]]
    failure = [j for j in common if p5[j] != tru[j] and po[j] == tru[j]]

    picks = []
    if rescue:
        picks.append(("rescue",  sorted(rescue,  key=lambda j: ov5[j] - ovo[j])[len(rescue) // 2]))
    if failure:
        picks.append(("failure", sorted(failure, key=lambda j: ov5[j] - ovo[j])[len(failure) // 2]))

    for kind, j in picks:
        print(f"\n{kind.upper()} (512 vs Original): dataset image {j} | true={CLASS_NAMES[tru[j]]} | "
              f"overlap 512={ov5[j]:.3f} orig={ovo[j]:.3f} (delta {ov5[j]-ovo[j]:+.3f}) | "
              f"pred 512={CLASS_NAMES[p5[j]]} orig={CLASS_NAMES[po[j]]}")
        plot_final_layer_comparison(
            j,
            BASE_PATH_512, SAVE_PATH_512, RUN_NAME_512, REPRESENTATIVE_SEED, NUM_TOKENS_512,
            BASE_PATH_ORIGINAL, SAVE_PATH_ORIGINAL, RUN_NAME_ORIGINAL, REPRESENTATIVE_SEED, NUM_TOKENS_ORIGINAL)