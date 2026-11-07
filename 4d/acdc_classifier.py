# -*- coding: utf-8 -*-
# ACDC 4D Classification using MDBPE Tokens + ViT
# Dynamic labels + dynamic num_classes
# Experiments: Original, 1024, 2048

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import math
import glob
from opacus import PrivacyEngine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH_ORIGINAL = os.path.expanduser(
    "/your/path/mdbpe_output_acdc4d_original")
BASE_PATH_1024 = os.path.expanduser(
    "/your/path/mdbpe_output_acdc4d_1024")
BASE_PATH_2048 = os.path.expanduser(
    "/your/path/mdbpe_output_acdc4d_2048")
ACDC_ROOT_TRAIN = "/your/path/acdc/training"   # ← ADD (replaces ACDC_ROOT)
ACDC_ROOT_TEST  = "/home/anushka1/scratch/acdc/testing"    # ← ADD

SAVE_PATH = os.path.expanduser(
    "/home/anushka1/scratch/codes/4d/k_fold")
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DIMENSIONS  = 128 # 128 previously
NUM_HEADS   = 4 # 4
NUM_LAYERS  = 2 #2
BASE_TOKENS = 256
FILE_MODULO = 128
EPOCHS      = 1000
BATCH_SIZE  = 2
USE_NPE     = True
USE_IPE     = True
SEEDS       = [0, 1 , 2, 3]
PATIENCE    = 30

TARGET_H = 32
TARGET_W = 27
TARGET_D = 2
TARGET_T = 8

USE_DP      = False
DP_NOISE    = 0.5
DP_MAX_GRAD = 1.0
DP_DELTA    = 1e-5

# ── Dynamic label loading ──────────────────────────────────────────────────────
def load_acdc_labels_dynamic(train_root, test_root):
    train_dirs = sorted(glob.glob(os.path.join(train_root, "patient*")))
    test_dirs  = sorted(glob.glob(os.path.join(test_root,  "patient*")))
    all_dirs   = train_dirs + test_dirs

    all_groups = set()
    dir_groups = {}
    for d in all_dirs:
        cfg = os.path.join(d, "Info.cfg")
        if not os.path.exists(cfg):
            continue
        with open(cfg) as f:
            for line in f:
                if line.startswith("Group:"):
                    group = line.strip().split(": ")[1].strip()
                    all_groups.add(group)
                    dir_groups[d] = group
                    break

    class_map   = {g: i for i, g in enumerate(sorted(all_groups))}
    num_classes = len(class_map)
    print(f"Discovered classes: {class_map}")

    def dirs_to_labels(dirs):
        labels = []
        for d in dirs:
            group = dir_groups.get(d, None)
            if group is not None:
                labels.append(class_map[group])
            else:
                print(f"⚠️  No label found for {d}, defaulting to 0")
                labels.append(0)
        return labels

    return (dirs_to_labels(train_dirs),
            dirs_to_labels(test_dirs),
            class_map, num_classes)

train_labels, test_labels, CLASS_MAP, NUM_CLASSES = \
    load_acdc_labels_dynamic(ACDC_ROOT_TRAIN, ACDC_ROOT_TEST)

print(f"Train: {len(train_labels)}, dist: {Counter(train_labels)}")
print(f"Test:  {len(test_labels)},  dist: {Counter(test_labels)}")
print(f"Num classes: {NUM_CLASSES}")

#k fold

def build_pooled_file_label_list(base_path):
    entries = []
    for folder_name, labels in [('tokensequences', train_labels),
                                 ('tokensequences_test', test_labels)]:
        for i in range(FILE_MODULO):
            folder = os.path.join(base_path, folder_name, str(i))
            if not os.path.exists(folder):
                continue
            for f in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, f)
                orig_idx = int(f.replace(".dat", ""))
                label = labels[orig_idx] if orig_idx < len(labels) else 0
                entries.append((fpath, label))
    return entries


# ── 4D Positional encoding ─────────────────────────────────────────────────────
def grid_positional_encoding_4d(embed_dims):
    grid = torch.zeros(TARGET_H, TARGET_W, TARGET_D, TARGET_T, 4)
    for x in range(TARGET_H):
        grid[x, :, :, :, 0] = x / TARGET_H
    for y in range(TARGET_W):
        grid[:, y, :, :, 1] = y / TARGET_W
    for z in range(TARGET_D):
        grid[:, :, z, :, 2] = z / TARGET_D
    for t in range(TARGET_T):
        grid[:, :, :, t, 3] = t / TARGET_T
    rets = []
    for i in range(embed_dims):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * grid))
    return torch.cat(rets, -1)

positional_encodings_half    = grid_positional_encoding_4d(int(DIMENSIONS / 16))
positional_encodings_quarter = grid_positional_encoding_4d(int(DIMENSIONS / 32))

def get_pre_loaded_tokens(base_path, num_tokens):
    pre_loaded = []
    for i in range(BASE_TOKENS, num_tokens):
        path = os.path.join(base_path, "tokenshapes", str(i) + ".dat")
        if os.path.exists(path):
            pre_loaded.append(torch.load(path, weights_only=False))
    return pre_loaded

def compute_positional_encoding_4d(
        x, y, z, t, tokenClass,
        n_x, n_y, n_z, n_t,
        pre_loaded_tokens, num_tokens):
    pe = torch.zeros(DIMENSIONS)
    pe[0:int(DIMENSIONS/4)] = \
        positional_encodings_quarter[x][y][z][t]
    if n_x is not None and USE_NPE:
        pe[int(DIMENSIONS/4):int(DIMENSIONS/2)] = \
            positional_encodings_quarter[n_x][n_y][n_z][n_t]
    if tokenClass < BASE_TOKENS and USE_IPE:
        pe[int(DIMENSIONS/2):] = \
            positional_encodings_half[x][y][z][t]
    elif USE_IPE and \
            tokenClass - BASE_TOKENS < len(pre_loaded_tokens):
        token_data             = pre_loaded_tokens[tokenClass - BASE_TOKENS]
        crop                   = token_data[0]
        offset_x, offset_y     = token_data[1], token_data[2]
        offset_z, offset_t     = token_data[3], token_data[4]
        indices                = torch.nonzero(crop != -1)
        indices[:, 0]         += x + offset_x
        indices[:, 1]         += y + offset_y
        indices[:, 2]         += z + offset_z
        indices[:, 3]         += t + offset_t
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < TARGET_H) &
            (indices[:, 1] >= 0) & (indices[:, 1] < TARGET_W) &
            (indices[:, 2] >= 0) & (indices[:, 2] < TARGET_D) &
            (indices[:, 3] >= 0) & (indices[:, 3] < TARGET_T))
        indices = indices[valid]
        if indices.size(0) > 0:
            full_area = positional_encodings_half[
                indices[:, 0], indices[:, 1],
                indices[:, 2], indices[:, 3], :]
            pe[int(DIMENSIONS/2):] = full_area.sum(0)
    return pe

# ── Dataset ────────────────────────────────────────────────────────────────────
class ACDCTokenDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, labels,
                 num_tokens, pre_loaded_tokens,
                 split='train'):
        self.labels            = labels
        self.num_tokens        = num_tokens
        self.pre_loaded_tokens = pre_loaded_tokens
        self.files             = []

        folder_name = 'tokensequences_test' \
            if split == 'test' else 'tokensequences'

        for i in range(FILE_MODULO):
            folder = os.path.join(base_path, folder_name, str(i))
            if not os.path.exists(folder):
                continue
            for f in sorted(os.listdir(folder)):
                self.files.append(os.path.join(folder, f))

        print(f"Found {len(self.files)} sequence files ({split})")

    def __len__(self):
        return len(self.files)

    def tokenise(self, data):
        SOS_token  = self.num_tokens
        EOS_token  = self.num_tokens + 1
        actual_len = len(data)
        seq_len    = actual_len + 1
        sequence   = torch.ones(seq_len).long() * EOS_token
        sequence[0] = SOS_token
        pos_enc    = torch.zeros(seq_len, DIMENSIONS)

        for i in range(actual_len):
            tokenClass          = data[i][0]
            x, y, z, t         = data[i][1], data[i][2], data[i][3], data[i][4]
            n_x, n_y, n_z, n_t = None, None, None, None
            if i + 1 < actual_len:
                n_x, n_y = data[i+1][1], data[i+1][2]
                n_z, n_t = data[i+1][3], data[i+1][4]
            pos_enc[i+1]  = compute_positional_encoding_4d(
                x, y, z, t, tokenClass,
                n_x, n_y, n_z, n_t,
                self.pre_loaded_tokens, self.num_tokens)
            sequence[i+1] = tokenClass

        return sequence, pos_enc

    def __getitem__(self, idx):
        data     = torch.load(self.files[idx], weights_only=False)
        seq, pos = self.tokenise(data)
        fname    = os.path.basename(self.files[idx])
        orig_idx = int(fname.replace(".dat", ""))
        label    = self.labels[orig_idx] \
            if orig_idx < len(self.labels) else 0
        return seq, pos, torch.tensor(label).long()

# ── ADD ACDCTokenDatasetFromList HERE ──────────────────────────────────────
class ACDCTokenDatasetFromList(torch.utils.data.Dataset):
    def __init__(self, file_label_list, num_tokens, pre_loaded_tokens):
        self.entries = file_label_list
        self.num_tokens = num_tokens
        self.pre_loaded_tokens = pre_loaded_tokens

    def __len__(self):
        return len(self.entries)

    def tokenise(self, data):
        # copy this method verbatim from ACDCTokenDataset — identical logic,
        # doesn't reference self.files/self.labels, so no changes needed
        SOS_token  = self.num_tokens
        EOS_token  = self.num_tokens + 1
        actual_len = len(data)
        seq_len    = actual_len + 1
        sequence   = torch.ones(seq_len).long() * EOS_token
        sequence[0] = SOS_token
        pos_enc    = torch.zeros(seq_len, DIMENSIONS)
        for i in range(actual_len):
            tokenClass          = data[i][0]
            x, y, z, t         = data[i][1], data[i][2], data[i][3], data[i][4]
            n_x, n_y, n_z, n_t = None, None, None, None
            if i + 1 < actual_len:
                n_x, n_y = data[i+1][1], data[i+1][2]
                n_z, n_t = data[i+1][3], data[i+1][4]
            pos_enc[i+1]  = compute_positional_encoding_4d(
                x, y, z, t, tokenClass,
                n_x, n_y, n_z, n_t,
                self.pre_loaded_tokens, self.num_tokens)
            sequence[i+1] = tokenClass
        return sequence, pos_enc

    def __getitem__(self, idx):
        fpath, label = self.entries[idx]        # ← the only real difference:
        data = torch.load(fpath, weights_only=False)  # reads directly from the
        seq, pos = self.tokenise(data)                # pooled list instead of
        return seq, pos, torch.tensor(label).long() 

# ── Collate ────────────────────────────────────────────────────────────────────
def make_collate_fn(num_tokens):
    EOS_token = num_tokens + 1
    def collate_fn(batch):
        seqs, pos_encs, labels = zip(*batch)
        max_len     = max(s.size(0) for s in seqs)
        padded_seqs = torch.stack([
            F.pad(s, (0, max_len - s.size(0)), value=EOS_token)
            for s in seqs])
        padded_pos  = torch.stack([
            F.pad(p, (0, 0, 0, max_len - p.size(0)), value=0.0)
            for p in pos_encs])
        masks = torch.stack([
            torch.cat([
                torch.zeros(s.size(0), dtype=torch.bool),
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
        self.embedding   = nn.Embedding(num_tokens + 2, DIMENSIONS)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, DIMENSIONS))
        self.layer       = nn.TransformerEncoderLayer(
            d_model=DIMENSIONS, nhead=NUM_HEADS,
            dim_feedforward=512, batch_first=True,
            norm_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(
            self.layer, num_layers=NUM_LAYERS)
        self.linear      = nn.Linear(DIMENSIONS, num_tokens + 2)

    def get_features(self, tokens, pos_emb, padding_mask=None):
        x   = self.embedding(tokens) + pos_emb
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        if padding_mask is not None:
            cls_mask     = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)

class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(DIMENSIONS),
            nn.Linear(DIMENSIONS, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, features):
        return self.classifier(features[:, 0])

# ── Train one seed ─────────────────────────────────────────────────────────────
def train_classifier(base_path, num_tokens, run_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*50}")
    print(f"{run_name} | Seed: {seed} | "
          f"Classes: {NUM_CLASSES} | Tokens: {num_tokens}")
    print(f"{'='*50}")

    pre_loaded   = get_pre_loaded_tokens(base_path, num_tokens)
    print(f"Loaded {len(pre_loaded)} token shapes")

    train_ds     = ACDCTokenDataset(base_path, train_labels,
                                    num_tokens, pre_loaded, 'train')
    test_ds      = ACDCTokenDataset(base_path, test_labels,
                                    num_tokens, pre_loaded, 'test')
    collate      = make_collate_fn(num_tokens)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=collate)

    model     = TransformerDecoderModel(num_tokens).to(DEVICE)
    head      = ClassificationHead(NUM_CLASSES).to(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()               # release cached blocks from previous seed
        torch.cuda.reset_peak_memory_stats()   # reset peak counter so THIS run measures fresh

    total        = len(train_labels)
    class_counts = [train_labels.count(i) for i in range(NUM_CLASSES)]
    weights      = torch.tensor([
        total / (NUM_CLASSES * c + 1e-8)
        for c in class_counts]).float().to(DEVICE)
    criterion    = nn.CrossEntropyLoss(weight=weights)

    # N         = sum(p.numel() for p in
    #                 list(model.parameters()) + list(head.parameters()))
    # scaled_lr = 0.0002 * (1.0 / math.sqrt(N / 54_000_000))
    # print(f"Scaled lr: {scaled_lr:.6f}, Params: {N:,}")

    # optimizer = torch.optim.AdamW(
    #     list(model.parameters()) + list(head.parameters()),
    #     lr=scaled_lr, weight_decay=0.01)

    N        = sum(p.numel() for p in
                   list(model.parameters()) + list(head.parameters()))
    fixed_lr = 1e-4
    print(f"Fixed lr: {fixed_lr:.6f}, Params: {N:,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=fixed_lr, weight_decay=0.1)
    
    if USE_DP:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = \
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=DP_NOISE,
                max_grad_norm=DP_MAX_GRAD)

    train_accs, test_accs, auc_scores = [], [], []
    best_auc = 0.0; best_epoch = 0; best_acc = 0.0; ewi = 0; best_ece = 0.0
    time_start = time.time()
    epoch_times = []
    peak_memory_mb = None
    ece_scores = []

    safe_name      = run_name.replace(' ', '_')
    ckpt_path      = os.path.join(SAVE_PATH, f"ckpt_{safe_name}_seed{seed}.pt")
    best_ckpt_path = os.path.join(SAVE_PATH, f"best_ckpt_{safe_name}_fold{fold}_seed{seed}.pt")

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        ckpt        = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        train_accs  = ckpt['train_accs']
        test_accs   = ckpt['test_accs']
        auc_scores  = ckpt.get('auc_scores', [])
        start_epoch = ckpt['epoch'] + 1
        best_auc    = ckpt.get('best_auc', 0.0)
        best_epoch  = ckpt.get('best_epoch', 0)
        ewi         = ckpt.get('ewi', 0)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train(); head.train()
        correct = total_b = 0; total_loss = 0.0

        for seq, pos, mask, labels in train_loader:
            seq    = seq.to(DEVICE);  pos    = pos.to(DEVICE)
            mask   = mask.to(DEVICE); labels = labels.to(DEVICE)
            features = model.get_features(seq[:, :-1], pos[:, :-1], mask[:, :-1])
            logits   = head(features)
            loss     = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct    += (logits.argmax(dim=1) == labels).sum()
            total_b    += labels.size(0)
            total_loss += loss.item()

        train_accs.append((correct / total_b).item())

        if epoch == start_epoch and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Peak GPU memory (epoch {epoch}): {peak_memory_mb:.1f} MB")
        
        elapsed = time.time() - time_start
        epoch_times.append(elapsed)



        # ── Test ──────────────────────────────────────────────────────────────
        model.eval(); head.eval()
        correct = total_b = 0
        all_probs, all_labels_list = [], []

        with torch.no_grad():
            for seq, pos, mask, labels in test_loader:
                seq    = seq.to(DEVICE);  pos    = pos.to(DEVICE)
                mask   = mask.to(DEVICE); labels = labels.to(DEVICE)
                features = model.get_features(seq[:, :-1], pos[:, :-1], mask[:, :-1])
                logits   = head(features)
                correct += (logits.argmax(dim=1) == labels).sum()
                total_b += labels.size(0)
                all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        test_acc = (correct / total_b).item() if total_b > 0 else 0.0
        test_accs.append(test_acc)

        try:
            all_probs_np  = np.array(all_probs)
            all_labels_np = np.array(all_labels_list)
            if NUM_CLASSES == 2:
                auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            else:
                auc = roc_auc_score(all_labels_np, all_probs_np,
                                    multi_class='ovr', average='macro')
        except ValueError:
            auc = 1.0 / NUM_CLASSES
        auc_scores.append(auc)

        ece = compute_ece(all_probs_np, all_labels_np)   # ← ADD

        ece_scores.append(ece)

        pred_dist = Counter(np.argmax(all_probs_np, axis=1).tolist())

        if auc > best_auc:
            best_auc = auc; best_epoch = epoch
            best_acc = test_acc; ewi = 0
            best_ece = ece
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'head': head.state_dict(), 'auc': auc, 
                        'test_acc': test_acc, 'class_map': CLASS_MAP,
                        'all_probs': all_probs_np, 'all_labels': all_labels_np},
                       best_ckpt_path)
        else:
            ewi += 1

        print(f"  Epoch {epoch}: train={train_accs[-1]:.3f} "
              f"test={test_acc:.3f} auc={auc:.3f} ece={ece:.4f}"
              f"loss={total_loss/len(train_loader):.4f} "
              f"pred_dist={dict(pred_dist)} "
              f"time={(time.time()-time_start)/60:.1f}min")

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'train_accs': train_accs, 'test_accs': test_accs,
            'auc_scores': auc_scores, 'best_auc': best_auc,
            'best_epoch': best_epoch, 'ewi': ewi,
        }, ckpt_path)

        if ewi >= PATIENCE:
            print(f"\nEarly stopping @ epoch {epoch}. "
                  f"Best AUC={best_auc:.4f} @ epoch {best_epoch}")
            break
    if USE_DP:
        epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
        print(f"\n✅ ε={epsilon:.2f} δ={DP_DELTA}")
    else:
        epsilon = float('inf')

    print(f"\n{run_name} seed={seed} | "
          f"best_auc={best_auc:.3f} best_acc={best_acc:.3f} "
          f"@ epoch {best_epoch}")
    return train_accs, test_accs, auc_scores, best_auc, best_acc, best_epoch, epsilon, epoch_times, peak_memory_mb

# ── ADD train_one_fold HERE ────────────────────────────────────────────────────
def train_one_fold(base_path, num_tokens, run_name, fold_idx, seed,
                    train_entries, test_entries, pre_loaded):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ACDCTokenDatasetFromList(train_entries, num_tokens, pre_loaded)
    test_ds  = ACDCTokenDatasetFromList(test_entries,  num_tokens, pre_loaded)
    collate  = make_collate_fn(num_tokens)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)

    model = TransformerDecoderModel(num_tokens).to(DEVICE)
    head  = ClassificationHead(NUM_CLASSES).to(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fold_train_labels = [lbl for _, lbl in train_entries]
    total = len(fold_train_labels)
    class_counts = [fold_train_labels.count(i) for i in range(NUM_CLASSES)]
    weights = torch.tensor([total / (NUM_CLASSES * c + 1e-8) for c in class_counts]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    N = sum(p.numel() for p in list(model.parameters()) + list(head.parameters()))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                   lr=1e-4, weight_decay=0.1)

    train_accs, test_accs, auc_scores = [], [], []
    best_auc = 0.0; best_epoch = 0; best_acc = 0.0; ewi = 0
    best_bal_acc = 0.0; best_macro_f1 = 0.0
    peak_memory_mb = None

    for epoch in range(EPOCHS):
        model.train(); head.train()
        correct = total_b = 0; total_loss = 0.0
        for seq, pos, mask, labels in train_loader:
            seq, pos, mask, labels = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            features = model.get_features(seq[:, :-1], pos[:, :-1], mask[:, :-1])
            logits = head(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(dim=1) == labels).sum()
            total_b += labels.size(0)
        train_accs.append((correct / total_b).item())

        if epoch == 0 and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        model.eval(); head.eval()
        correct = total_b = 0
        all_probs, all_labels_list = [], []
        with torch.no_grad():
            for seq, pos, mask, labels in test_loader:
                seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
                features = model.get_features(seq[:, :-1], pos[:, :-1], mask[:, :-1])
                logits = head(features)
                correct += (logits.argmax(dim=1) == labels.to(DEVICE)).sum()
                total_b += labels.size(0)
                all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
                all_labels_list.extend(labels.tolist())

        test_acc = (correct / total_b).item() if total_b > 0 else 0.0
        test_accs.append(test_acc)
        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels_list)
        y_pred = np.argmax(all_probs_np, axis=1)

        try:
            auc = (roc_auc_score(all_labels_np, all_probs_np[:, 1]) if NUM_CLASSES == 2
                   else roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro'))
        except ValueError:
            auc = 1.0 / NUM_CLASSES
        auc_scores.append(auc)

        bal_acc = balanced_accuracy_score(all_labels_np, y_pred)
        macro_f1 = f1_score(all_labels_np, y_pred, average='macro')

        if auc > best_auc:
            best_auc, best_epoch, best_acc, ewi = auc, epoch, test_acc, 0
            best_bal_acc, best_macro_f1 = bal_acc, macro_f1
            torch.save({
                'model': model.state_dict(),
                'head': head.state_dict(),
                'fold_idx': fold_idx, 'seed': seed, 'run_name': run_name,
                'epoch': epoch, 'best_auc': best_auc,
            }, os.path.join(SAVE_PATH, f"model_{run_name.replace(' ','_')}_fold{fold_idx}_seed{seed}.pt"))
        else:
            ewi += 1

        if ewi >= PATIENCE:
            print(f"Fold {fold_idx} early stopping @ epoch {epoch}. Best AUC={best_auc:.4f}")
            break

    print(f"{run_name} fold={fold_idx} seed={seed} | best_auc={best_auc:.3f} "
          f"best_acc={best_acc:.3f} best_bal_acc={best_bal_acc:.3f} best_macro_f1={best_macro_f1:.3f}")
    return best_auc, best_acc, best_bal_acc, best_macro_f1, peak_memory_mb

# ── Experiments ────────────────────────────────────────────────────────────────
experiments = {
    'original': (BASE_PATH_ORIGINAL, 256,  'Original'),
    '1024':     (BASE_PATH_1024,     1280, 'MDBPE 1024'),
    '2048':     (BASE_PATH_2048,     2304, 'MDBPE 2048')
}

# verify paths and get actual num_tokens from metadata
for key, (path, fallback_tokens, name) in experiments.items():
    if not os.path.exists(path):
        print(f"⚠️  Missing: {path}")
    else:
        try:
            m = torch.load(f'{path}/metadata.pt', weights_only=False)
            actual = m.get('MAXIMUM_NUMBER_OF_TOKENS', fallback_tokens)
            experiments[key] = (path, actual, name)
            print(f"✅ {name}: num_tokens={actual}")
        except Exception as e:
            print(f"⚠️  {name} metadata error: {e}")

all_results = {k: [] for k in experiments}

for key, (base_path, num_tokens, run_name) in experiments.items():
    if not os.path.exists(base_path):
        print(f"⚠️  Skipping {run_name} — path missing")
        continue

    pooled = build_pooled_file_label_list(base_path)
    pooled_labels = np.array([lbl for _, lbl in pooled])
    print(f"\n{run_name}: pooled {len(pooled)} sequences across 5 folds")

    pre_loaded = get_pre_loaded_tokens(base_path, num_tokens)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(pooled)), pooled_labels)):
        
        train_entries = [pooled[i] for i in train_idx]
        test_entries  = [pooled[i] for i in test_idx]
        for seed in SEEDS:
            result_path = os.path.join(SAVE_PATH, f"model_{run_name.replace(' ','_')}_fold{fold_idx}_seed{seed}.pt")
            if os.path.exists(result_path):
                saved = torch.load(result_path, weights_only=False, map_location='cpu')
                all_results[key].append((saved['best_auc'], saved.get('best_acc', 0.0),
                                         saved.get('best_bal_acc', 0.0), saved.get('best_macro_f1', 0.0),
                                         saved.get('peak_memory_mb')))
                print(f"  {run_name} fold={fold_idx} seed={seed} already done, skipping")
                continue
            result = train_one_fold(base_path, num_tokens, run_name, fold_idx, seed,
                                    train_entries, test_entries, pre_loaded)
            all_results[key].append(result)

    print(f"\n✅ {run_name} all 5 folds, {len(SEEDS)} seeds done")

# ── Final table ────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"ACDC K-FOLD Results | {NUM_CLASSES} classes: {CLASS_MAP}")
print(f"{'Run':<20} {'Acc':>8} {'±':>6} {'AUC':>8} {'±':>6} {'BalAcc':>8} {'±':>6} {'MacroF1':>8} {'±':>6}")
print(f"{'='*72}")
for key, (_, _, run_name) in experiments.items():
    runs = all_results[key]
    if len(runs) == 0:
        continue
    aucs      = np.array([r[0] for r in runs])
    accs      = np.array([r[1] for r in runs])
    bal_accs  = np.array([r[2] for r in runs])
    macro_f1s = np.array([r[3] for r in runs])
    n_collapsed = sum(1 for r in runs if r[2] < (1.0 / NUM_CLASSES) * 1.1)
    print(f"{run_name:<20} {accs.mean():.3f}   {accs.std():.3f}   "
          f"{aucs.mean():.3f}   {aucs.std():.3f}   "
          f"{bal_accs.mean():.3f}   {bal_accs.std():.3f}   "
          f"{macro_f1s.mean():.3f}   {macro_f1s.std():.3f}   "
          f"[{n_collapsed}/{len(runs)} near-chance]")
print(f"{'='*72}")
print(f"\n✅ ALL DONE! Saved to {SAVE_PATH}")

print("\n── Statistical Significance (AUC) ──")
for pair in [('original', '1024'), ('original', '2048'),   ('1024', '2048')]:
    key_a, key_b = pair
    if key_a not in all_results or key_b not in all_results:
        continue
    aucs_a = np.array([r[0] for r in all_results[key_a]])
    aucs_b = np.array([r[0] for r in all_results[key_b]])
    if len(aucs_a) >= 2 and len(aucs_b) >= 2:
        t_stat, p_value = ttest_ind(aucs_a, aucs_b)
        print(f"{key_a} vs {key_b}: t={t_stat:.3f}, p={p_value:.4f} "
              f"{'(significant)' if p_value < 0.05 else '(not significant)'}")

print("\n── Peak GPU Memory (MB) ──")
for key, (_, _, run_name) in experiments.items():
    runs = all_results[key]
    memories = np.array([r[4] for r in runs if r[4] is not None])
    if len(memories) > 0:
        print(f"{run_name:<20} {memories.mean():.1f} ± {memories.std():.1f} MB")