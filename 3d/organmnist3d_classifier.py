# -*- coding: utf-8 -*-
# OrganMNIST3D Classification using MDBPE Tokens + ViT
import torch
from opacus import PrivacyEngine
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import math
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, classification_report)
from scipy.stats import ttest_ind
import copy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH_ORIGINAL = os.path.expanduser(
    "/your/path")
BASE_PATH_512 = os.path.expanduser(
    "/your/path")
BASE_PATH_1024 = os.path.expanduser(
    "/your/path")
SAVE_PATH = os.path.expanduser(
    "/your/path")
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DIMENSIONS  = 512 #128
NUM_HEADS   = 8 #4
NUM_LAYERS  = 4 #2
BASE_TOKENS = 256
FILE_MODULO = 128
EPOCHS      = 40
BATCH_SIZE  = 1
USE_NPE     = True
USE_IPE     = True
SEEDS       = [0, 1, 2, 3]
TARGET      = 28   # 16x16x16 downsampled volume

USE_DP      = False
DP_NOISE    = 0.5
DP_MAX_GRAD = 1.0
DP_DELTA    = 1e-5

# ── MedMNIST labels ────────────────────────────────────────────────────────────
data_flag = 'organmnist3d'
info      = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True, transform=None)
test_dataset  = DataClass(split='test',  download=True, transform=None)

train_labels = [int(train_dataset[i][1][0]) for i in range(len(train_dataset))]
test_labels  = [int(test_dataset[i][1][0])  for i in range(len(test_dataset))]

print(f"Train: {len(train_labels)} labels")
print(f"Test:  {len(test_labels)} labels")

# ── Positional encoding (3D) ────────────────────────────────────────────────────
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

# Compute embed_dims via ceiling to guarantee enough elements, then slice exactly
half_embed_dims    = math.ceil((DIMENSIONS / 2) / 6)   # enough for half slot
quarter_embed_dims = math.ceil((DIMENSIONS / 4) / 6)   # enough for quarter slot

positional_encodings_half_full    = grid_positional_encoding_3d(half_embed_dims)
positional_encodings_quarter_full = grid_positional_encoding_3d(quarter_embed_dims)

# Slice down to exact required size
positional_encodings_half    = positional_encodings_half_full[..., :int(DIMENSIONS/2)]
positional_encodings_quarter = positional_encodings_quarter_full[..., :int(DIMENSIONS/4)]

def get_pre_loaded_tokens(base_path, num_tokens):
    pre_loaded = []
    for i in range(BASE_TOKENS, num_tokens):
        path = os.path.join(base_path, "tokenshapes", str(i) + ".dat")
        if os.path.exists(path):
            pre_loaded.append(torch.load(path, weights_only=False))
    return pre_loaded

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


class TokenSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, labels, num_tokens, pre_loaded_tokens, split='train'):
        self.base_path         = base_path
        self.labels            = labels
        self.num_tokens        = num_tokens
        self.pre_loaded_tokens = pre_loaded_tokens
        self.files             = []
        ORIGINAL_PATH = "/home/anushka1/scratch/codes/3d/output/mdbpe_output_organmnist3d_original"
        split_tag = 'test' if split == 'test' else 'train'
        self.recovered = torch.load(
            os.path.join(base_path, f"recovered_labels_{split_tag}.pt"),
            weights_only=False)

        folder_name = 'tokensequences_test' if split == 'test' else 'tokensequences'
        for i in range(FILE_MODULO):
            folder = os.path.join(base_path, folder_name, str(i))
            if not os.path.exists(folder):
                continue
            for file in sorted(os.listdir(folder)):
                self.files.append(os.path.join(folder, file))

        print(f"Found {len(self.files)} sequence files ({split})")

    def __len__(self):
        return len(self.files)

    def tokenise(self, data):
        SOS_token = self.num_tokens
        EOS_token = self.num_tokens + 1
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
                self.pre_loaded_tokens, self.num_tokens)
            sequence[index+1] = tokenClass

        return sequence.long(), positional_encoding

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        sequence, pos_enc = self.tokenise(data)
        fname    = os.path.basename(self.files[idx])
        orig_idx = int(fname.replace(".dat", ""))
        label = self.recovered[orig_idx]
        return sequence, pos_enc, torch.tensor(label).long()


def make_collate_fn(num_tokens):
    EOS_token = num_tokens + 1
    def collate_fn(batch):
        seqs, pos_encs, labels = zip(*batch)
        max_len = max(s.size(0) for s in seqs)
        padded_seqs = torch.stack([
            F.pad(s, (0, max_len - s.size(0)), value=EOS_token) for s in seqs])
        padded_pos = torch.stack([
            F.pad(p, (0, 0, 0, max_len - p.size(0)), value=0.0) for p in pos_encs])
        masks = torch.stack([
            torch.cat([torch.zeros(s.size(0), dtype=torch.bool),
                       torch.ones(max_len - s.size(0), dtype=torch.bool)])
            for s in seqs])
        return padded_seqs, padded_pos, masks, torch.stack(labels)
    return collate_fn


def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == np.array(labels))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


# ── Model ──────────────────────────────────────────────────────────────────────
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

def _dataset_lengths(dataset):
    """Sequence length of every file, without running the expensive
    tokenise()/positional-encoding step -- just enough to rank by length."""
    lengths = []
    for f in dataset.files:
        data = torch.load(f, weights_only=False)
        lengths.append(len(data))
    return np.array(lengths)

def find_max_batch_size(model, head, criterion, dataset, collate_fn, optimizer=None, start=4, max_try=64):
    # Probe against the dataset's globally longest sequences (attention
    # memory is O(seq_len^2), so a batch drawn later under shuffle=True
    # with longer sequences can OOM even if a random probe batch passed).
    # ALSO force Adam's lazy per-parameter state (exp_avg, exp_avg_sq) to
    # allocate during probing by actually calling optimizer.step() -- Adam
    # never allocates that state on backward() alone, so a probe that only
    # calls backward() silently underestimates real training's memory use.
    lengths = _dataset_lengths(dataset)
    longest_order = np.argsort(lengths)[::-1]

    def _try_batch(bs):
        idx = longest_order[:bs].tolist()
        subset = torch.utils.data.Subset(dataset, idx)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(subset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
            seq, pos, mask, labels = next(iter(loader))
            seq, pos, mask, labels = (seq.to(DEVICE), pos.to(DEVICE),
                                    mask.to(DEVICE), labels.to(DEVICE))
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                feat = model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1])
                loss = criterion(head(feat), labels)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            del seq, pos, mask, labels, feat, loss
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    best = 0
    bs = start
    while bs <= max_try:
        if _try_batch(bs):
            best = bs
            bs *= 2
        else:
            break

    if best == 0:
        bs = max(1, start // 2)
        while bs >= 1:
            if _try_batch(bs):
                best = bs
                break
            bs //= 2

    if best == 0:
        raise RuntimeError(
            "Even batch_size=1 OOMs on the longest sequence in this "
            "dataset. Sequence length itself is the problem, not batch "
            "size -- needs gradient checkpointing, mixed precision, or a "
            "hard length cap for this config."
        )
    return best

def measure_inference_time(model, head, dataset, collate_fn, batch_size, n_warmup=5, n_measure=20):
    # batch_size was tuned against train_ds's longest sequences via
    # find_max_batch_size, not this dataset's. test_ds is a different,
    # unsubsetted set of files -- no guarantee its longest sequence is
    # <= train_ds's longest, so re-clamp against this dataset directly.
    dummy_criterion = nn.CrossEntropyLoss()
    safe_batch_size = min(batch_size, find_max_batch_size(
        model, head, dummy_criterion, dataset, collate_fn,
        start=max(1, batch_size), max_try=batch_size))

    loader = DataLoader(dataset, batch_size=safe_batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval(); head.eval()
    it = iter(loader)

    def _safe_forward(seq, pos, mask):
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return head(model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
    with torch.no_grad():
        for _ in range(min(n_warmup, max(1, len(dataset) // safe_batch_size))):
            try:
                seq, pos, mask, labels = next(it)
            except StopIteration:
                it = iter(loader); seq, pos, mask, labels = next(it)
            seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
            _safe_forward(seq, pos, mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    max_attempts = n_measure * 20   # generous cap to prevent infinite hang
    attempts = 0
    with torch.no_grad():
        while len(times) < n_measure and attempts < max_attempts:
            attempts += 1
            try:
                seq, pos, mask, labels = next(it)
            except StopIteration:
                it = iter(loader); seq, pos, mask, labels = next(it)
            seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            out = _safe_forward(seq, pos, mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if out is None:
                print(f"  ⚠️ Skipped inference-timing batch (OOM, seq_len={seq.size(1)})")
                continue
            times.append(time.time() - t0)

    if len(times) == 0:
        raise RuntimeError(
            "Every inference-timing batch OOM'd -- this sequence length "
            "cannot be measured at this batch size, even with mixed precision."
        )

    times = np.array(times)
    per_image = times / safe_batch_size
    return per_image.mean(), per_image.std()


# ── Train one seed ─────────────────────────────────────────────────────────────
def train_classifier(base_path, train_labels, test_labels, num_tokens, run_name, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'='*50}\nRun: {run_name} | Seed: {seed}\nvocab_size: {num_tokens}\n{'='*50}")

    pre_loaded = get_pre_loaded_tokens(base_path, num_tokens)
    print(f"Loaded {len(pre_loaded)} token shapes")

    train_ds = TokenSequenceDataset(base_path, train_labels, num_tokens, pre_loaded, split='train')
    test_ds  = TokenSequenceDataset(base_path, test_labels,  num_tokens, pre_loaded, split='test')

    collate = make_collate_fn(num_tokens)
    nw = 4 if os.name != 'nt' else 0

    model = TransformerDecoderModel(num_tokens).to(DEVICE)
    num_classes = len(set(train_labels))
    head = ClassificationHead(num_classes=num_classes).to(DEVICE)

    # weighted loss ✅
    used = [train_ds.recovered[int(os.path.basename(f)[:-4])] for f in train_ds.files]
    total = len(used)
    class_counts = [used.count(i) for i in range(num_classes)]
    weights = torch.tensor([total / (num_classes * c + 1e-8) for c in class_counts]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)


    N = sum(p.numel() for p in
        list(model.parameters()) +
        list(head.parameters()))
    base_lr   = 0.0002
    scaled_lr = base_lr * \
        (1.0 / math.sqrt(
            N / 54_000_000))
    # scaled_lr = scaled_lr / 5
    scaled_lr = 1e-4

    import copy

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                   lr=scaled_lr, weight_decay=0.01)

    # Probe with the optimizer so Adam's state is allocated and the memory
    # ceiling is measured accurately -- then undo the weight updates the
    # probe's optimizer.step() calls made, so training starts from the
    # seeded initialisation.
    m_state = copy.deepcopy(model.state_dict())
    h_state = copy.deepcopy(head.state_dict())

    max_bs = find_max_batch_size(model, head, criterion, train_ds, collate, optimizer=optimizer)

    model.load_state_dict(m_state)
    head.load_state_dict(h_state)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                   lr=scaled_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # actual_batch_size = min(BATCH_SIZE, max_bs)
    # print(f"  Max batch size found: {max_bs}  (using {actual_batch_size})")
    # torch.cuda.empty_cache() 

    # if run_name == "Original Sequence":   # match this to however your script identifies this config
    #     actual_batch_size = 1
    #     print(f"  Forcing batch_size=1 for Original Sequence given confirmed OOM at longer sequences")

    #     # Halve the probe-validated batch size as a safety margin: the probe
    # # cleans (empty_cache + reset_peak_memory_stats) before every single
    # # candidate test in isolation, but real training runs hundreds of
    # # batches back-to-back with no such per-batch cleanup, so memory can
    # # fragment over time even though the same nominal amount is free at
    # # the very start.
    # else:
    #     actual_batch_size = max(1, actual_batch_size // 2)
    #     actual_batch_size = max(1, actual_batch_size // 2)
    #     print(f"  Using safety-margined batch size: {actual_batch_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, collate_fn=collate)
 


    best_auc = 1.0 / num_classes
    best_epoch = 0
    best_test_acc = 0.0      # ← ADD
    best_bal_acc = 0.0 
    patience = 20
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(SAVE_PATH, f"best_ckpt_{run_name.replace(' ','_')}_seed{seed}.pt")

    if USE_DP:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=DP_NOISE, max_grad_norm=DP_MAX_GRAD)

    train_accs, test_accs, auc_scores = [], [], []
    bal_accs, macro_f1_scores, per_class_f1_scores, ece_scores = [], [], [], []
    epoch_times = []
    per_epoch_times = [] 
    time_start = time.time()

    ckpt_path = os.path.join(SAVE_PATH, f"ckpt_{run_name.replace(' ','_')}_seed{seed}.pt")
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accs = ckpt['train_accs']
        test_accs  = ckpt['test_accs']
        auc_scores = ckpt.get('auc_scores', [])
        bal_accs   = ckpt.get('bal_accs', [])
        macro_f1_scores     = ckpt.get('macro_f1_scores', [])
        per_class_f1_scores = ckpt.get('per_class_f1_scores', [])
        ece_scores          = ckpt.get('ece_scores', [])
        epoch_times         = ckpt.get('epoch_times', []) 
        start_epoch = ckpt['epoch'] + 1
        best_auc   = ckpt.get('best_auc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        best_test_acc = ckpt.get('best_test_acc', 0.0)   # ← ADD
        best_bal_acc  = ckpt.get('best_bal_acc', 0.0)    # ← ADD
        epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        print(f"Resumed from epoch {start_epoch}")
        scheduler.last_epoch = start_epoch
       

    peak_memory_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for seq, pos, mask, labels in train_loader:
        seq, pos = seq.to(DEVICE), pos.to(DEVICE)
        emb = model.embedding(seq[:, :-1])
        print("embedding abs max:", emb.abs().max().item())
        print("embedding abs mean:", emb.abs().mean().item())
        print("pos_emb  abs max:", pos[:, :-1].abs().max().item())
        print("pos_emb  abs mean:", pos[:, :-1].abs().mean().item())
        break
    for epoch in range(start_epoch, epochs):
        model.train(); head.train()
        correct = total_b = 0
        total_loss = 0.0
        ACCUM = 8 
        optimizer.zero_grad()
        for i, (seq, pos, mask, labels) in enumerate(train_loader):
            seq, pos, mask, labels = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    features = model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1])
                    logits = head(features)
                    loss = criterion(logits, labels)
                (loss / ACCUM).backward()
                if (i + 1) % ACCUM == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️ Skipped batch (OOM, seq_len={seq.size(1)})")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum()
            total_b += labels.size(0)
            total_loss += loss.item()
        scheduler.step()
        train_acc = (correct/total_b).item()
        train_accs.append(train_acc)

        torch.cuda.empty_cache()

        if epoch == start_epoch and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Peak GPU memory (epoch {epoch}): {peak_memory_mb:.1f} MB")

        model.eval(); head.eval()
        correct = total_b = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for seq, pos, mask, labels in test_loader:
                seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
                try:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        features = model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1])
                        logits = head(features)
                except torch.cuda.OutOfMemoryError:
                    print(f"  ⚠️ Skipped eval batch (OOM, seq_len={seq.size(1)})")
                    torch.cuda.empty_cache()
                    continue
                preds = logits.argmax(dim=1)
                correct += (preds == labels.to(DEVICE)).sum()
                total_b += labels.size(0)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        print("  pred_dist:", Counter(np.argmax(all_probs, axis=1).tolist()))

        test_acc = 0.0 if total_b == 0 else (correct/total_b).item()
        test_accs.append(test_acc)

        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"  ⚠️ AUC computation failed: {e}")
            auc = 1.0 / num_classes
        auc_scores.append(auc)

        bal_acc = balanced_accuracy_score(all_labels, np.argmax(all_probs, axis=1))
        bal_accs.append(bal_acc)

        preds_array = np.argmax(all_probs, axis=1)
        macro_f1 = f1_score(all_labels, preds_array, average='macro')
        per_class_f1 = f1_score(all_labels, preds_array, average=None, labels=list(range(num_classes)))
        macro_f1_scores.append(macro_f1)
        per_class_f1_scores.append(per_class_f1)

        ece = compute_ece(all_probs, all_labels)
        ece_scores.append(ece)

        elapsed = time.time() - time_start
        epoch_times.append(elapsed)

        # per-epoch delta — isolates pure compute cost per step from convergence
        # speed (the paper's actual FLOPs/sequence-length claim), independent of
        # how many epochs it takes to reach best AUC
        per_epoch_time = elapsed - (epoch_times[-2] if len(epoch_times) > 1 else 0.0)
        per_epoch_times.append(per_epoch_time)


        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_test_acc = test_acc      # ← ADD
            best_bal_acc = bal_acc        # ← ADD
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(), 'head': head.state_dict(),
                'optimizer': optimizer.state_dict(), 'test_acc': test_acc, 'auc': auc,
                'bal_acc': bal_acc, 'macro_f1_scores': macro_f1_scores,
                'per_class_f1_scores': per_class_f1_scores, 'ece_scores': ece_scores,
                'all_probs': all_probs, 'all_labels': all_labels
            }, best_ckpt_path)
        else:
            epochs_without_improvement += 1
            # if epochs_without_improvement >= patience:
            #     print(f"\nEarly stopping at epoch {epoch}. Best AUC={best_auc:.4f} at epoch {best_epoch}")
            #     break

        print(f"  Epoch {epoch}: train={train_acc:.3f} test={test_acc:.3f} auc={auc:.3f} "
              f"loss={total_loss/len(train_loader):.4f} time={elapsed/60:.1f}min")

        torch.save({
            'epoch': epoch, 'model': model.state_dict(), 'head': head.state_dict(),
            'optimizer': optimizer.state_dict(), 'train_accs': train_accs, 'test_accs': test_accs,
            'auc_scores': auc_scores, 'bal_accs': bal_accs, 'macro_f1_scores': macro_f1_scores,
            'per_class_f1_scores': per_class_f1_scores, 'ece_scores': ece_scores, 'best_auc': best_auc, 'best_epoch': best_epoch,  'best_test_acc': best_test_acc, 'best_bal_acc': best_bal_acc,                     # ← ADD
    'epochs_without_improvement': epochs_without_improvement,  'epoch_times': epoch_times 
        }, ckpt_path)

    if USE_DP:
        epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
        print(f"\n✅ ε={epsilon:.2f} δ={DP_DELTA}")
    else:
        epsilon = float('inf')

    print(f"\n{run_name} seed={seed} final test acc: {test_accs[-1]:.3f} auc: {auc_scores[-1]:.3f}")

    inf_mean, inf_std = measure_inference_time(model, head, test_ds, collate, batch_size=BATCH_SIZE)
    print(f"  Inference: {inf_mean*1000:.3f} ± {inf_std*1000:.3f} ms/image")


    print(f"\n{run_name} seed={seed} final test acc: {test_accs[-1]:.3f} auc: {auc_scores[-1]:.3f}")
    return train_accs, test_accs, auc_scores, epsilon, bal_accs, macro_f1_scores, \
           per_class_f1_scores, ece_scores, epoch_times, peak_memory_mb, best_auc, best_test_acc, best_bal_acc, per_epoch_times, inf_mean, inf_std, max_bs


# ── Run all experiments with 4 seeds ──────────────────────────────────────────
experiments = {
    '512':      (BASE_PATH_512,      768,  '512 Extra Tokens'),
    '1024':     (BASE_PATH_1024,     1280, '1024 Extra Tokens'),
    'original': (BASE_PATH_ORIGINAL, 256,  'Original Sequence')
}

for key, (path, _, name) in experiments.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
    else:
        print(f"✅ Found: {name}")

all_results = {k: [] for k in experiments}
for key, (base_path, num_tokens, run_name) in experiments.items():
    if not os.path.exists(base_path):
        print(f"⚠️ Skipping {run_name} — path missing")
        continue
    for seed in SEEDS:
        result = train_classifier(base_path, train_labels, test_labels, num_tokens, run_name, seed, EPOCHS)
        all_results[key].append(result)
    print(f"\n✅ {run_name} all {len(SEEDS)} seeds done")

print("\n── Last-10-epoch mean (report this) ──")
for key, runs in all_results.items():
    if not runs: continue
    acc = [np.mean(r[1][-10:]) for r in runs]
    auc = [np.mean(r[2][-10:]) for r in runs]
    bal = [np.mean(r[4][-10:]) for r in runs]
    print(f"{key}: acc {np.mean(acc):.3f}±{np.std(acc):.3f}  "
          f"auc {np.mean(auc):.4f}±{np.std(auc):.4f}  balAcc {np.mean(bal):.3f}±{np.std(bal):.3f}")


# ── Plot with mean±std ─────────────────────────────────────────────────────────
def ema_smooth(values, alpha=0.9):
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed

fig, ax = plt.subplots(figsize=(10, 6))
colors = {'original': '#1f77b4', '512': '#ff7f0e', '1024': '#2ca02c'}
names  = {'original': 'Original Sequence', '512': '512 Extra Tokens', '1024': '1024 Extra Tokens'}

for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    min_len = min(len(r[1]) for r in runs)
    all_accs = np.array([r[1][:min_len] for r in runs])
    mean = all_accs.mean(axis=0)
    std  = all_accs.std(axis=0)
    epochs_x = list(range(len(mean)))
    color, name = colors[key], names[key]

    smoothed_mean = ema_smooth(mean.tolist(), alpha=0.9)
    smoothed_std  = ema_smooth(std.tolist(), alpha=0.9)
    sm, ss = np.array(smoothed_mean), np.array(smoothed_std)

    final_accs_leg = np.array([r[11] for r in runs])
    acc_mean_leg, acc_std_leg = final_accs_leg.mean(), final_accs_leg.std()

    ax.plot(epochs_x, sm, color=color, linewidth=2,
            label=f"{name} ({acc_mean_leg:.3f}±{acc_std_leg:.3f})")
    ax.fill_between(epochs_x, sm - ss, sm + ss, color=color, alpha=0.2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Classification Test Accuracy, ViT on OrganMNIST3D 16×16×16\n'
             f'({len(SEEDS)} seeds, mean±std)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.0, 1.0])
plt.tight_layout()
plot_path = os.path.join(SAVE_PATH, "classification.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n✅ Plot saved: {plot_path}")


# ── Final table ────────────────────────────────────────────────────────────────
print(f"\n{'Run':<22} {'Acc':>8} {'±':>6} {'AUC':>8} {'±':>6} {'Bal. Acc':>8} {'±':>6}")
print("─" * 70)
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]
    times_to_best = []
    for r in runs:
        auc_list = r[2]
        epoch_times_list = r[8]
        best_idx = int(np.argmax(auc_list))
        times_to_best.append(epoch_times_list[best_idx] / 60)
    times_to_best = np.array(times_to_best)
    print(f"{name:<22} {times_to_best.mean():.1f} ± {times_to_best.std():.1f} min")

    final_accs = np.array([r[11] for r in runs]) 
    final_aucs = np.array([r[10] for r in runs])
    final_bal_accs = np.array([r[12] for r in runs])

    acc_mean, acc_std = final_accs.mean(), final_accs.std()
    auc_mean, auc_std = final_aucs.mean(), final_aucs.std()
    bal_acc_mean, bal_acc_std = final_bal_accs.mean(), final_bal_accs.std()

    print(f"{name:<22} {acc_mean:.3f}   {acc_std:.3f}   {auc_mean:.3f}   {auc_std:.3f}   "
          f"{bal_acc_mean:.3f}   {bal_acc_std:.3f}")
print("─" * 70)
print(f"\n✅ ALL DONE!")
print(f"Results saved to: {SAVE_PATH}")

print("\n── Statistical Significance (AUC) ──")
for pair in [('original', '1024'), ('512', '1024')]:
    key_a, key_b = pair
    aucs_a = np.array([r[10] for r in all_results[key_a]])
    aucs_b = np.array([r[10] for r in all_results[key_b]])
    if len(aucs_a) >= 2 and len(aucs_b) >= 2:
        t_stat, p_value = ttest_ind(aucs_a, aucs_b)
        print(f"{key_a} vs {key_b}: t={t_stat:.3f}, p={p_value:.4f} "
              f"{'(significant)' if p_value < 0.05 else '(not significant)'}")
    else:
        print(f"{key_a} vs {key_b}: insufficient seeds for test")

print("\n── Peak GPU Memory (MB) ──")
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]
    memories = np.array([r[9] for r in runs if r[9] is not None])
    if len(memories) > 0:
        print(f"{name:<22} {memories.mean():.1f} ± {memories.std():.1f} MB")
    # skip first epoch of each run (often includes one-time setup overhead)
    all_per_epoch = np.concatenate([np.array(r[13][1:]) for r in runs if len(r[13]) > 1])
    print(f"{name:<22} {all_per_epoch.mean():.2f} ± {all_per_epoch.std():.2f} sec/epoch")

print("\n── Max Batch Size Found ──")
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0: continue
    bs_vals = np.array([r[16] for r in runs])
    print(f"{names[key]:<22} {bs_vals.mean():.1f} ± {bs_vals.std():.1f}")

print("\n── Inference Time (ms/image) ──")
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0: continue
    inf_means = np.array([r[14]*1000 for r in runs])
    print(f"{names[key]:<22} {inf_means.mean():.3f} ± {inf_means.std():.3f} ms/image")

print("\n── Per-Epoch Compute Time (pure step cost, excludes convergence speed) ──")
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0: continue
    all_times = np.concatenate([np.array(r[13][1:]) for r in runs if len(r[13]) > 1])
    print(f"{names[key]:<22} {all_times.mean():.2f} ± {all_times.std():.2f} sec/epoch")