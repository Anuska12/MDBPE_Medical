# -*- coding: utf-8 -*-
"""
DermaMNIST (final focal gamma=2 runs): inference time per image, measured
AFTER training from the saved best checkpoints -- no retraining.

For each config (Original / 512 / 1024) and seed (0-3):
  - loads  best_ckpt_<run>_seed<s>.pt  from the gamma2 results folder
  - times the forward pass (embedding -> encoder -> CLS head) on EVERY test
    image (2005), batch size 1, fp32 (as in training), on the GPU,
    after warm-up, with cuda.synchronize around each forward pass
All configs are timed in the SAME job on the SAME GPU, one after another.
Tokenisation / positional-encoding construction is done beforehand on the
CPU and is NOT timed (same convention as measure_inference_time).

Model, positional encoding and tokenise() are copied unchanged from the
final gamma2 classifier script.

usage: python3 -u derma_inference_time.py
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn

# ── must match the final gamma2 classifier script ─────────────────────────────
CONFIGS = [   # (run name, base path, num_tokens)
    ('Original Sequence', "/your/path",   1000),
    ('512 Extra Tokens',  "/your/path",  1512),
    ('1024 Extra Tokens', "/your/path", 2024),
]
SAVE_PATH   = "/your/path"
DIMENSIONS, NUM_HEADS, NUM_LAYERS = 512, 8, 4
BASE_TOKENS, FILE_MODULO = 1000, 128
NUM_CLASSES = 7
USE_NPE = USE_IPE = True
WIDTH = HEIGHT = 28
SEEDS    = [0, 1, 2, 3]
N_WARMUP = 5                      # same as n_warmup=5 in measure_inference_time
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── positional encoding (copied from the classifier) ──────────────────────────
def grid_positional_encoding(embed_dims):
    grid = torch.ones(WIDTH, HEIGHT, 2)
    for x in range(WIDTH):
        grid[x, :, 0] = x / WIDTH
    for y in range(HEIGHT):
        grid[:, y, 1] = y / HEIGHT
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
    positional_encoding = torch.zeros(DIMENSIONS)
    positional_encoding[0:int(DIMENSIONS / 4)] = positional_encodings_quarter[x][y]
    if n_x is not None and USE_NPE:
        positional_encoding[int(DIMENSIONS/4):int(DIMENSIONS/4*2)] = positional_encodings_quarter[n_x][n_y]
    if tokenClass < BASE_TOKENS and USE_IPE:
        positional_encoding[int(DIMENSIONS / 2):] = positional_encodings_half[x][y]
    elif USE_IPE and tokenClass - BASE_TOKENS < len(pre_loaded_tokens):
        tokenshape = pre_loaded_tokens[tokenClass - BASE_TOKENS][0]
        offset_x   = pre_loaded_tokens[tokenClass - BASE_TOKENS][1]
        indices    = torch.nonzero(tokenshape != -1)
        indices[:, 0] += x + offset_x
        indices[:, 1] += y
        valid = ((indices[:, 0] >= 0) & (indices[:, 0] < WIDTH) &
                 (indices[:, 1] >= 0) & (indices[:, 1] < HEIGHT))
        indices = indices[valid]
        if indices.size(0) > 0:
            x_idx = indices[:, 0].long()
            y_idx = indices[:, 1].long()
            full_area = positional_encodings_half[x_idx, y_idx, :]
            positional_encoding[int(DIMENSIONS / 2):] = full_area.sum(0)
    return positional_encoding

def tokenise(data, num_tokens, pre_loaded_tokens):
    SOS_token, EOS_token = num_tokens, num_tokens + 1
    actual_len = len(data)
    seq_len    = actual_len + 1
    sequence = torch.ones(seq_len) * EOS_token
    sequence[0] = SOS_token
    positional_encoding = torch.zeros(seq_len, DIMENSIONS)
    for index in range(actual_len):
        tokenClass = data[index][0]
        x, y       = data[index][1], data[index][2]
        n_x, n_y   = None, None
        if index + 1 < actual_len:
            n_x = data[index+1][1]
            n_y = data[index+1][2]
        positional_encoding[index+1] = compute_positional_encoding(
            x, y, tokenClass, n_x, n_y, pre_loaded_tokens, num_tokens)
        sequence[index+1] = tokenClass
    return sequence.long(), positional_encoding

def test_files(base_path):
    files = []
    for i in range(FILE_MODULO):
        folder = os.path.join(base_path, 'tokensequences_test', str(i))
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            files.append(os.path.join(folder, f))
    return files

# ── model (copied from the classifier) ────────────────────────────────────────
class TransformerDecoderModel(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.layer = nn.TransformerEncoderLayer(
            d_model=DIMENSIONS, nhead=NUM_HEADS, dim_feedforward=1024,
            batch_first=True, norm_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=NUM_LAYERS)
        self.embedding = nn.Embedding(num_tokens + 2, DIMENSIONS)
        self.linear = nn.Linear(DIMENSIONS, num_tokens + 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIMENSIONS))
    def get_features(self, tokens, pos_emb, padding_mask=None):
        x   = self.embedding(tokens)
        x   = x + pos_emb
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
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
    def forward(self, features):
        return self.classifier(features[:, 0])

def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

@torch.no_grad()
def time_model(model, head, items):
    model.eval(); head.eval()
    for k in range(min(N_WARMUP, len(items))):               # warm-up, not timed
        s, p = items[k]
        head(model.get_features(s[:, :-1], p[:, :-1]))
    sync()
    times = []
    for s, p in items:                                        # batch size 1, every test image
        sync(); t0 = time.perf_counter()
        head(model.get_features(s[:, :-1], p[:, :-1]))
        sync(); times.append(time.perf_counter() - t0)
    return np.array(times)

if __name__ == "__main__":
    print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else "", flush=True)
    summary, per_run = {}, []
    for run_name, base, num_tokens in CONFIGS:
        pre = get_pre_loaded_tokens(base, num_tokens)
        files = test_files(base)
        print(f"\n{run_name}: num_tokens={num_tokens}, {len(pre)} token shapes, "
              f"{len(files)} test images -- tokenising (CPU, not timed)...", flush=True)
        items, lens = [], []
        for f in files:
            s, p = tokenise(torch.load(f, weights_only=False), num_tokens, pre)
            lens.append(s.numel() - 1)
            items.append((s.unsqueeze(0).to(DEVICE), p.unsqueeze(0).to(DEVICE)))
        lens = np.array(lens)

        run_means = []
        for seed in SEEDS:
            path = os.path.join(SAVE_PATH, f"best_ckpt_{run_name.replace(' ', '_')}_seed{seed}.pt")
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)}"); continue
            ck = torch.load(path, weights_only=False, map_location='cpu')
            model = TransformerDecoderModel(num_tokens).to(DEVICE)
            head  = ClassificationHead(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(ck['model']); head.load_state_dict(ck['head'])
            t = time_model(model, head, items) * 1000.0                # ms
            run_means.append(t.mean())
            per_run.append(dict(config=run_name, seed=seed, n_test=len(items),
                                ms_mean=float(t.mean()), ms_std=float(t.std())))
            print(f"  seed {seed}: {t.mean():.3f} ± {t.std():.3f} ms/image (n={len(items)})", flush=True)
            del model, head
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        run_means = np.array(run_means)
        summary[run_name] = dict(n_runs=int(len(run_means)), ms_mean=float(run_means.mean()),
                                 ms_std=float(run_means.std()), mean_len=float(lens.mean()),
                                 max_len=int(lens.max()))
        del items

    print("\n── DermaMNIST inference time (ms/image, batch 1, fp32, forward pass only) ──")
    print(f"{'config':20s} {'seeds':>5} {'ms/image':>18} {'mean len':>9} {'max len':>8}")
    for name, r in summary.items():
        print(f"{name:20s} {r['n_runs']:>5} {r['ms_mean']:>9.3f} ± {r['ms_std']:<6.3f} "
              f"{r['mean_len']:>9.1f} {r['max_len']:>8}")
    orig = summary.get('Original Sequence', {}).get('ms_mean')
    if orig:
        for name, r in summary.items():
            if name != 'Original Sequence':
                print(f"{name}: {100 * (1 - r['ms_mean'] / orig):+.1f}% faster than Original")
    out = os.path.join(SAVE_PATH, "inference_time_derma.json")
    json.dump(dict(summary=summary, runs=per_run), open(out, "w"), indent=2)
    print("saved", out)