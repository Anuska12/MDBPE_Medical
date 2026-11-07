# -*- coding: utf-8 -*-
"""
ACDC k-fold: inference time per patient, measured AFTER training (no retraining).

For every config (Original / MDBPE 1024 / MDBPE 2048), fold (5) and seed (4):
  - rebuilds that fold's test set exactly as in the k-fold training script
    (same pooled file list, StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
  - loads the saved model  model_<run>_fold<f>_seed<s>.pt  from SAVE_PATH
  - times the forward pass (embedding -> encoder -> CLS head) for each test
    patient at batch size 1, fp32, on the GPU, with warm-up and cuda.synchronize

All configs are timed in the SAME job on the SAME GPU, one after another,
so the comparison is not affected by which node a training job ran on.
Tokenisation / positional-encoding construction happens beforehand on the
CPU and is NOT included (same as measure_inference_time in the other scripts).

Output: per-run mean ms/patient, then per config mean ± std over the 20 runs
(np.std, population std, as in your other tables), plus mean sequence length.

usage: python3 -u acdc_kfold_inference_time.py
"""
import os, glob, json, time
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold

# ── must match the k-fold training script ─────────────────────────────────────
BASE_PATHS = {
    'Original':   "/home/anushka1/scratch/mdbpe_output_acdc4d_original",
    'MDBPE 1024': "/home/anushka1/scratch/mdbpe_output_acdc4d_1024",
    'MDBPE 2048': "/home/anushka1/scratch/mdbpe_output_acdc4d_2048",
}
FALLBACK_TOKENS = {'Original': 256, 'MDBPE 1024': 1280, 'MDBPE 2048': 2304}
ACDC_ROOT_TRAIN = "/home/anushka1/scratch/acdc/training"
ACDC_ROOT_TEST  = "/home/anushka1/scratch/acdc/testing"
SAVE_PATH       = "/home/anushka1/scratch/codes/4d/k_fold"
DIMENSIONS, NUM_HEADS, NUM_LAYERS = 128, 4, 2
BASE_TOKENS, FILE_MODULO = 256, 128
SEEDS   = [0, 1, 2, 3]
N_FOLDS = 5
TARGET_H, TARGET_W, TARGET_D, TARGET_T = 32, 27, 2, 8
USE_NPE = USE_IPE = True
N_WARMUP = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── labels (identical logic to the training script) ───────────────────────────
def load_acdc_labels_dynamic(train_root, test_root):
    train_dirs = sorted(glob.glob(os.path.join(train_root, "patient*")))
    test_dirs  = sorted(glob.glob(os.path.join(test_root,  "patient*")))
    groups, dir_groups = set(), {}
    for d in train_dirs + test_dirs:
        cfg = os.path.join(d, "Info.cfg")
        if not os.path.exists(cfg):
            continue
        with open(cfg) as f:
            for line in f:
                if line.startswith("Group:"):
                    g = line.strip().split(": ")[1].strip()
                    groups.add(g); dir_groups[d] = g
                    break
    class_map = {g: i for i, g in enumerate(sorted(groups))}
    lab = lambda dirs: [class_map[dir_groups[d]] if d in dir_groups else 0 for d in dirs]
    return lab(train_dirs), lab(test_dirs), class_map, len(class_map)

train_labels, test_labels, CLASS_MAP, NUM_CLASSES = \
    load_acdc_labels_dynamic(ACDC_ROOT_TRAIN, ACDC_ROOT_TEST)

def build_pooled_file_label_list(base_path):
    entries = []
    for folder_name, labels in [('tokensequences', train_labels),
                                ('tokensequences_test', test_labels)]:
        for i in range(FILE_MODULO):
            folder = os.path.join(base_path, folder_name, str(i))
            if not os.path.exists(folder):
                continue
            for f in sorted(os.listdir(folder)):
                orig_idx = int(f.replace(".dat", ""))
                label = labels[orig_idx] if orig_idx < len(labels) else 0
                entries.append((os.path.join(folder, f), label))
    return entries

# ── positional encoding + tokenise (copied from the training script) ─────────
def grid_positional_encoding_4d(embed_dims):
    grid = torch.zeros(TARGET_H, TARGET_W, TARGET_D, TARGET_T, 4)
    for x in range(TARGET_H): grid[x, :, :, :, 0] = x / TARGET_H
    for y in range(TARGET_W): grid[:, y, :, :, 1] = y / TARGET_W
    for z in range(TARGET_D): grid[:, :, z, :, 2] = z / TARGET_D
    for t in range(TARGET_T): grid[:, :, :, t, 3] = t / TARGET_T
    rets = []
    for i in range(embed_dims):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * grid))
    return torch.cat(rets, -1)

PE_HALF    = grid_positional_encoding_4d(int(DIMENSIONS / 16))
PE_QUARTER = grid_positional_encoding_4d(int(DIMENSIONS / 32))

def get_pre_loaded_tokens(base_path, num_tokens):
    out = []
    for i in range(BASE_TOKENS, num_tokens):
        p = os.path.join(base_path, "tokenshapes", f"{i}.dat")
        if os.path.exists(p):
            out.append(torch.load(p, weights_only=False))
    return out

def pos_enc_4d(x, y, z, t, cls, n, pre):
    pe = torch.zeros(DIMENSIONS)
    pe[0:DIMENSIONS // 4] = PE_QUARTER[x][y][z][t]
    if n is not None and USE_NPE:
        pe[DIMENSIONS // 4:DIMENSIONS // 2] = PE_QUARTER[n[0]][n[1]][n[2]][n[3]]
    if cls < BASE_TOKENS and USE_IPE:
        pe[DIMENSIONS // 2:] = PE_HALF[x][y][z][t]
    elif USE_IPE and cls - BASE_TOKENS < len(pre):
        td = pre[cls - BASE_TOKENS]
        idx = torch.nonzero(td[0] != -1)
        idx[:, 0] += x + td[1]; idx[:, 1] += y + td[2]
        idx[:, 2] += z + td[3]; idx[:, 3] += t + td[4]
        ok = ((idx[:, 0] >= 0) & (idx[:, 0] < TARGET_H) & (idx[:, 1] >= 0) & (idx[:, 1] < TARGET_W) &
              (idx[:, 2] >= 0) & (idx[:, 2] < TARGET_D) & (idx[:, 3] >= 0) & (idx[:, 3] < TARGET_T))
        idx = idx[ok]
        if idx.size(0) > 0:
            pe[DIMENSIONS // 2:] = PE_HALF[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], :].sum(0)
    return pe

def tokenise(data, num_tokens, pre):
    L = len(data)
    seq = torch.ones(L + 1).long() * (num_tokens + 1)
    seq[0] = num_tokens
    pos = torch.zeros(L + 1, DIMENSIONS)
    for i in range(L):
        c, x, y, z, t = data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]
        n = (data[i+1][1], data[i+1][2], data[i+1][3], data[i+1][4]) if i + 1 < L else None
        pos[i + 1] = pos_enc_4d(x, y, z, t, c, n, pre)
        seq[i + 1] = c
    return seq, pos

# ── model (copied from the training script) ───────────────────────────────────
class TransformerDecoderModel(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.embedding   = nn.Embedding(num_tokens + 2, DIMENSIONS)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, DIMENSIONS))
        self.layer       = nn.TransformerEncoderLayer(
            d_model=DIMENSIONS, nhead=NUM_HEADS, dim_feedforward=512,
            batch_first=True, norm_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=NUM_LAYERS)
        self.linear      = nn.Linear(DIMENSIONS, num_tokens + 2)
    def get_features(self, tokens, pos_emb, padding_mask=None):
        x = self.embedding(tokens) + pos_emb
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat([torch.zeros(x.size(0), 1, dtype=torch.bool,
                                                  device=x.device), padding_mask], dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)

class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(DIMENSIONS), nn.Linear(DIMENSIONS, 128),
                                        nn.ReLU(), nn.Linear(128, num_classes))
    def forward(self, f): return self.classifier(f[:, 0])

def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

@torch.no_grad()
def time_model(model, head, items):
    """items: list of (seq, pos) CPU tensors. Returns per-patient seconds."""
    model.eval(); head.eval()
    gpu = [(s.unsqueeze(0).to(DEVICE), p.unsqueeze(0).to(DEVICE)) for s, p in items]
    for k in range(min(N_WARMUP, len(gpu))):                  # warm-up
        s, p = gpu[k]
        head(model.get_features(s[:, :-1], p[:, :-1]))
    sync()
    times = []
    for s, p in gpu:                                          # batch size 1, every test patient
        sync(); t0 = time.perf_counter()
        head(model.get_features(s[:, :-1], p[:, :-1]))
        sync(); times.append(time.perf_counter() - t0)
    return np.array(times)

if __name__ == "__main__":
    print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else "")
    print(f"Classes: {CLASS_MAP}")
    summary, per_run = {}, []
    for name, base in BASE_PATHS.items():
        try:
            num_tokens = torch.load(os.path.join(base, "metadata.pt"),
                                    weights_only=False).get('MAXIMUM_NUMBER_OF_TOKENS', FALLBACK_TOKENS[name])
        except Exception:
            num_tokens = FALLBACK_TOKENS[name]
        pooled = build_pooled_file_label_list(base)
        labels = np.array([l for _, l in pooled])
        pre = get_pre_loaded_tokens(base, num_tokens)
        print(f"\n{name}: num_tokens={num_tokens}, pooled {len(pooled)} patients, "
              f"{len(pre)} token shapes", flush=True)

        # tokenise every pooled patient once (CPU, not timed)
        cache = {}
        for fpath, _ in pooled:
            cache[fpath] = tokenise(torch.load(fpath, weights_only=False), num_tokens, pre)
        lens = np.array([cache[f][0].numel() - 1 for f, _ in pooled])

        run_means = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(pooled)), labels)):
            items = [cache[pooled[i][0]] for i in test_idx]
            for seed in SEEDS:
                mpath = os.path.join(SAVE_PATH, f"model_{name.replace(' ', '_')}_fold{fold}_seed{seed}.pt")
                if not os.path.exists(mpath):
                    print(f"  missing {os.path.basename(mpath)}"); continue
                ck = torch.load(mpath, weights_only=False, map_location='cpu')
                model = TransformerDecoderModel(num_tokens).to(DEVICE)
                head = ClassificationHead(NUM_CLASSES).to(DEVICE)
                model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
                t = time_model(model, head, items) * 1000.0           # ms
                run_means.append(t.mean())
                per_run.append(dict(config=name, fold=fold, seed=seed, n_test=len(items),
                                    ms_mean=float(t.mean()), ms_std=float(t.std())))
                print(f"  fold {fold} seed {seed}: {t.mean():.3f} ± {t.std():.3f} ms/patient "
                      f"(n={len(items)})", flush=True)
                del model, head
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
        run_means = np.array(run_means)
        summary[name] = dict(n_runs=int(len(run_means)), ms_mean=float(run_means.mean()),
                             ms_std=float(run_means.std()), mean_len=float(lens.mean()),
                             max_len=int(lens.max()))

    print("\n── ACDC k-fold inference time (ms/patient, batch 1, forward pass only) ──")
    print(f"{'config':12s} {'runs':>5} {'ms/patient':>18} {'mean len':>9} {'max len':>8}")
    for name, r in summary.items():
        print(f"{name:12s} {r['n_runs']:>5} {r['ms_mean']:>9.3f} ± {r['ms_std']:<6.3f} "
              f"{r['mean_len']:>9.1f} {r['max_len']:>8}")
    orig = summary.get('Original', {}).get('ms_mean')
    if orig:
        for name, r in summary.items():
            if name != 'Original':
                print(f"{name}: {100 * (1 - r['ms_mean'] / orig):+.1f}% faster than Original")
    out = os.path.join(SAVE_PATH, "inference_time_kfold.json")
    json.dump(dict(summary=summary, runs=per_run), open(out, "w"), indent=2)
    print("saved", out)
