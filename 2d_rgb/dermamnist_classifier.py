# -*- coding: utf-8 -*-
# BreastMNIST 64x64 Classification using MDBPE Tokens + ViT
# Fixes:
# 1. Label alignment: use filename index not loop idx ✅
# 2. Variable sequence length: use actual length not WIDTH*HEIGHT ✅
# 3. Custom collate_fn for padding ✅
# 4. 4 seeds [0,1,2,3] mean±std ✅

import torch
from opacus import PrivacyEngine
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, roc_curve, classification_report)
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_ind

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
DIMENSIONS  = 512
NUM_HEADS   = 8
NUM_LAYERS  = 4
BASE_TOKENS = 1000
FILE_MODULO = 128
EPOCHS      = 100
BATCH_SIZE  = 16
USE_NPE     = True
USE_IPE     = True
SEEDS       = [0, 1, 2, 3]  # ← 4 seeds ✅

# privacy
USE_DP      = False
DP_NOISE    = 0.5
DP_MAX_GRAD = 1.0
DP_DELTA    = 1e-5

# ── Load metadata ──────────────────────────────────────────────────────────────
def load_metadata(base_path):
    return torch.load(
        os.path.join(base_path, "metadata.pt"),
        weights_only=False)

meta_512 = load_metadata(BASE_PATH_512)
WIDTH     = meta_512["WIDTH"]
HEIGHT    = meta_512["HEIGHT"]
print(f"Image size: {WIDTH}x{HEIGHT}")

# ── MedMNIST labels ────────────────────────────────────────────────────────────
data_flag = 'dermamnist'
info      = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor()])

train_dataset = DataClass(
    split='train', download=True,
    transform=transform)
test_dataset = DataClass(
    split='test', download=True,
    transform=transform)

train_labels = [int(train_dataset[i][1][0])
                for i in range(len(train_dataset))]
test_labels  = [int(test_dataset[i][1][0])
                for i in range(len(test_dataset))]

print(f"Train: {len(train_labels)} labels")
print(f"Test:  {len(test_labels)} labels")

# ── Positional encoding ────────────────────────────────────────────────────────
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

positional_encodings_half    = grid_positional_encoding(
    int(DIMENSIONS / 8))
positional_encodings_quarter = grid_positional_encoding(
    int(DIMENSIONS / 16))

def get_pre_loaded_tokens(base_path, num_tokens):
    pre_loaded = []
    for i in range(BASE_TOKENS, num_tokens):
        path = os.path.join(
            base_path, "tokenshapes",
            str(i) + ".dat")
        if os.path.exists(path):
            pre_loaded.append(
                torch.load(path, weights_only=False))
    return pre_loaded

def compute_positional_encoding(
        x, y, tokenClass, n_x, n_y,
        pre_loaded_tokens, num_tokens):
    positional_encoding = torch.zeros(DIMENSIONS)
    positional_encoding[
        0:int(DIMENSIONS / 4)] = \
        positional_encodings_quarter[x][y]

    if n_x is not None and USE_NPE:
        positional_encoding[
            int(DIMENSIONS/4):
            int(DIMENSIONS/4*2)] = \
            positional_encodings_quarter[n_x][n_y]

    if tokenClass < BASE_TOKENS and USE_IPE:
        positional_encoding[
            int(DIMENSIONS / 2):] = \
            positional_encodings_half[x][y]
    elif USE_IPE and \
            tokenClass - BASE_TOKENS < \
            len(pre_loaded_tokens):
        tokenshape = pre_loaded_tokens[
            tokenClass - BASE_TOKENS][0]
        offset_x   = pre_loaded_tokens[
            tokenClass - BASE_TOKENS][1]
        indices    = torch.nonzero(
            tokenshape != -1)
        indices[:, 0] += x + offset_x
        indices[:, 1] += y
        valid = (
            (indices[:, 0] >= 0) &
            (indices[:, 0] < WIDTH) &
            (indices[:, 1] >= 0) &
            (indices[:, 1] < HEIGHT))
        indices = indices[valid]
        if indices.size(0) > 0:
            x_idx = indices[:, 0].long()
            y_idx = indices[:, 1].long()
            full_area = positional_encodings_half[
                x_idx, y_idx, :]
            positional_encoding[
                int(DIMENSIONS / 2):] = \
                full_area.sum(0)
    return positional_encoding


class TokenSequenceDataset(
        torch.utils.data.Dataset):

    def __init__(self, base_path,
                 labels, num_tokens,
                 pre_loaded_tokens,
                 split='train'):
        self.base_path         = base_path
        self.labels            = labels
        self.num_tokens        = num_tokens
        self.pre_loaded_tokens = pre_loaded_tokens
        self.files             = []

        # ── recovered labels: train files are in processing order, test files are not
        split_tag = 'test' if split == 'test' else 'train'
        self.recovered = (torch.load(os.path.join(base_path, "recovered_labels_train.pt"),
                                     weights_only=False)
                          if split_tag == 'train' else None)

        folder_name = \
            'tokensequences_test' \
            if split == 'test' \
            else 'tokensequences'

        for i in range(FILE_MODULO):
            folder = os.path.join(
                base_path,
                folder_name,
                str(i))
            if not os.path.exists(folder):
                continue
            for file in sorted(
                    os.listdir(folder)):
                self.files.append(
                    os.path.join(folder, file))

        print(f"Found {len(self.files)} "
              f"sequence files ({split})")

    def __len__(self):
        return len(self.files)

    def tokenise(self, data):
        SOS_token = self.num_tokens
        EOS_token = self.num_tokens + 1

        # ── FIX 2: use actual sequence
        #   length not WIDTH*HEIGHT ✅
        actual_len = len(data)
        seq_len    = actual_len + 1

        sequence = torch.ones(
            seq_len) * EOS_token
        sequence[0] = SOS_token

        positional_encoding = torch.zeros(
            seq_len, DIMENSIONS)

        for index in range(actual_len):
            tokenClass = data[index][0]
            x, y       = data[index][1], \
                         data[index][2]
            n_x, n_y   = None, None
            if index + 1 < actual_len:
                n_x = data[index+1][1]
                n_y = data[index+1][2]
            positional_encoding[index+1] = \
                compute_positional_encoding(
                    x, y, tokenClass,
                    n_x, n_y,
                    self.pre_loaded_tokens,
                    self.num_tokens)
            sequence[index+1] = tokenClass

        return sequence.long(), \
               positional_encoding

    def __getitem__(self, idx):
        data = torch.load(
            self.files[idx],
            weights_only=False)
        sequence, pos_enc = \
            self.tokenise(data)

        # ── FIX 1: use filename index
        #   not loop idx ✅
        fname    = os.path.basename(
            self.files[idx])
        orig_idx = int(
            fname.replace(".dat", ""))
        
        label = self.recovered[orig_idx] if self.recovered is not None else self.labels[orig_idx]

        return sequence, pos_enc, \
               torch.tensor(label).long()


# ── FIX 3: custom collate_fn ✅
# ── handles variable length seqs ✅
def make_collate_fn(num_tokens):
    EOS_token = num_tokens + 1

    def collate_fn(batch):
        seqs, pos_encs, labels = \
            zip(*batch)

        # find max length in batch ✅
        max_len = max(
            s.size(0) for s in seqs)

        # pad sequences ✅
        padded_seqs = torch.stack([
            F.pad(
                s,
                (0, max_len - s.size(0)),
                value=EOS_token)
            for s in seqs])

        # pad positional encodings ✅
        padded_pos = torch.stack([
            F.pad(
                p,
                (0, 0,
                 0, max_len - p.size(0)),
                value=0.0)
            for p in pos_encs])
        
        masks = torch.stack([
        torch.cat([
            torch.zeros(
                s.size(0),
                dtype=torch.bool),
            torch.ones(
                max_len - s.size(0),
                dtype=torch.bool)
        ])
        for s in seqs])

        return (padded_seqs,
                padded_pos,
                masks,
                torch.stack(labels))

    return collate_fn


# ── Calibration: Expected Calibration Error ✅
def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == np.array(labels))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & \
                (confidences <= bin_boundaries[i+1])
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
            d_model=DIMENSIONS,
            nhead=NUM_HEADS,
            dim_feedforward=1024,
            batch_first=True,
            norm_first=True,
            dropout=0.1)
        self.transformer = \
            nn.TransformerEncoder(
                self.layer,
                num_layers=NUM_LAYERS)
        self.embedding = nn.Embedding(
            num_tokens + 2, DIMENSIONS)
        self.linear = nn.Linear(
            DIMENSIONS, num_tokens + 2)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, DIMENSIONS))

    # def get_target_mask(self, seq_len):
    #     mask = torch.tril(
    #         torch.ones(seq_len, seq_len) == 1
    #     ).float()
    #     mask = mask.masked_fill(
    #         mask == 0, float('-inf'))
    #     mask = mask.masked_fill(
    #         mask == 1, float(0.0))
    #     return mask

    def get_features(self, tokens, pos_emb, padding_mask=None):
        x   = self.embedding(tokens)
        x   = x + pos_emb
        cls = self.cls_token.expand(
            x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # extend mask for CLS token ✅
        if padding_mask is not None:
            cls_mask = torch.zeros(
                x.size(0), 1,
                dtype=torch.bool,
                device=x.device)
            padding_mask = torch.cat(
                [cls_mask, padding_mask],
            dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)

    # def forward(self, tokens, pos_embedding):
    #     tokens = self.embedding(tokens)
    #     tokens = tokens + pos_embedding
    #     return self.linear(
    #         self.transformer(
    #             tokens[:, :-1],
    #             self.get_target_mask(
    #                 tokens.size()[1]-1
    #             ).to(tokens.device),
    #             is_causal=True))


class ClassificationHead(nn.Module):
    def __init__(self, dimensions=DIMENSIONS,
                 num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dimensions),
            nn.Linear(dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, features):
        return self.classifier(features[:, 0])

class FocalLoss(nn.Module):
    def __init__(self,
                 gamma,
                 weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self,
                logits,
                targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()



# ── Train one seed ─────────────────────────────────────────────────────────────
def train_classifier(base_path,
                     train_labels,
                     test_labels,
                     num_tokens,
                     run_name,
                     seed,
                     epochs=EPOCHS):

    # set seed ✅
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*50}")
    print(f"Run: {run_name} | Seed: {seed}")
    print(f"vocab_size: {num_tokens}")
    print(f"{'='*50}")

    pre_loaded = get_pre_loaded_tokens(
        base_path, num_tokens)
    print(f"Loaded {len(pre_loaded)} "
          f"token shapes")

    train_ds = TokenSequenceDataset(
        base_path, train_labels,
        num_tokens, pre_loaded,
        split='train')

    test_ds = TokenSequenceDataset(
        base_path, test_labels,
        num_tokens, pre_loaded,
        split='test')

    # ── FIX 3: use collate_fn ✅
    collate = make_collate_fn(num_tokens)
    nw = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate)   # ← ✅
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate)   # ← ✅

    model = TransformerDecoderModel(
        num_tokens).to(DEVICE)
    num_classes  = len(set(train_labels))
    head  = ClassificationHead(num_classes=num_classes).to(DEVICE)

    # # weighted loss %d classes ✅
    total        = len(train_labels)
    class_counts = [
        train_labels.count(i)
        for i in range(num_classes)]
    weights = torch.tensor([
        total / (num_classes * c + 1e-8)
        for c in class_counts
    ]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=weights)
    print("Class counts:", class_counts)
    
    # max_count = max(class_counts)
    # weights = torch.tensor([
    #     (max_count / (c + 1e-8)) ** 0.5 # gamma value
    #     for c in class_counts
    # ]).float().to(DEVICE)
    
    USE_FOCAL_LOSS = True
    GAMMA_VALUE    = 2  # ← set to False for this test ✅

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=GAMMA_VALUE,
            weight=weights)
        print(f"Using FocalLoss(gamma={GAMMA_VALUE}, weighted)")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights)
        print("Using plain weighted CrossEntropyLoss")
    N = sum(p.numel() for p in
        list(model.parameters()) +
        list(head.parameters()))
    base_lr   = 0.0002
    scaled_lr = base_lr * \
        (1.0 / math.sqrt(
            N / 54_000_000))
    print(f"Scaled lr: {scaled_lr:.6f}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(head.parameters()),
        lr=scaled_lr,
        weight_decay=0.01)
    scheduler = \
        torch.optim.lr_scheduler\
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6)
    best_auc = 0.0
    best_epoch = 0
    patience = 20
    epochs_without_improvement = 0

    best_ckpt_path = os.path.join(
        SAVE_PATH,
        f"best_ckpt_{run_name.replace(' ','_')}_seed{seed}.pt"
    )

    if USE_DP:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = \
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=DP_NOISE,
                max_grad_norm=DP_MAX_GRAD)

    train_accs = []
    test_accs  = []
    auc_scores = []
    bal_accs   = []
    macro_f1_scores    = []
    per_class_f1_scores = []
    ece_scores          = []
    epoch_times = []
    time_start = time.time()

    ckpt_path = os.path.join(
        SAVE_PATH,
        f"ckpt_{run_name.replace(' ','_')}"
        f"_seed{seed}.pt")

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        ckpt = torch.load(
            ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(
            ckpt['optimizer'])
        train_accs  = ckpt['train_accs']
        test_accs   = ckpt['test_accs']
        auc_scores  = ckpt.get(
            'auc_scores', [])
        bal_accs    = ckpt.get(
            'bal_accs', [])
        macro_f1_scores      = ckpt.get('macro_f1_scores', [])
        per_class_f1_scores  = ckpt.get('per_class_f1_scores', [])
        ece_scores           = ckpt.get('ece_scores', [])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch "
              f"{start_epoch}")
        for _ in range(start_epoch):
            scheduler.step()

    peak_memory_mb = None 

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, epochs):
        # ── Train ─────────────────────
        model.train()
        head.train()
        correct = total_b = 0
        total_loss = 0.0

        for seq, pos, mask, labels \
                in train_loader:
            seq    = seq.to(DEVICE)
            pos    = pos.to(DEVICE)
            mask   = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            features = model.get_features(
                seq[:, :-1],
                pos[:, :-1],
                padding_mask=mask[:, :-1])
            logits = head(features)
            loss   = criterion(
                logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds      = logits.argmax(
                dim=1)
            correct   += (
                preds == labels).sum()
            total_b   += labels.size(0)
            total_loss += loss.item()
        scheduler.step()
        train_acc = (correct/total_b)\
            .item()
        train_accs.append(train_acc)

        if epoch == start_epoch and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  Peak GPU memory (epoch {epoch}): {peak_memory_mb:.1f} MB")

        # ── Test ──────────────────────
        model.eval()
        head.eval()
        correct = total_b = 0
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for seq, pos, mask, labels \
                    in test_loader:
                seq    = seq.to(DEVICE)
                pos    = pos.to(DEVICE)
                mask   = mask.to(DEVICE)
                labels = labels.to(DEVICE)

                features = model.get_features(
                    seq[:, :-1],
                    pos[:, :-1],
                    padding_mask=mask[:, :-1])
                logits = head(features)
                preds  = logits.argmax(
                    dim=1)
                correct  += (
                    preds == labels).sum()
                total_b  += labels.size(0)

                probs = F.softmax(
                    logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(
                    labels.cpu().numpy())

        if total_b == 0:
            print("⚠️ No test samples!")
            test_acc = 0.0
        else:
            test_acc = (
                correct/total_b).item()
        test_accs.append(test_acc)

        try:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.5
        auc_scores.append(auc)

        # ── Balanced accuracy: averages
        #    per-class recall, robust to
        #    class imbalance ✅
        bal_acc = balanced_accuracy_score(
            all_labels,
            np.argmax(all_probs, axis=1))
        bal_accs.append(bal_acc)
        # ── Macro F1 + per-class F1 ✅
        preds_array = np.argmax(all_probs, axis=1)
        macro_f1 = f1_score(
            all_labels, preds_array, average='macro')
        per_class_f1 = f1_score(
            all_labels, preds_array, average=None,
            labels=list(range(num_classes)))
        n_unique = len(set(preds_array.tolist()))
        print(f"  unique predicted classes: {n_unique}/{num_classes}")
        macro_f1_scores.append(macro_f1)
        per_class_f1_scores.append(per_class_f1)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
                'auc': auc,
                'bal_acc': bal_acc,
                'macro_f1_scores': macro_f1_scores,
                'per_class_f1_scores': per_class_f1_scores,
                'ece_scores': ece_scores,
                'all_probs': all_probs,    # ← NEW ✅
                'all_labels': all_labels   # ← NEW ✅
            }, best_ckpt_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best Macro AUC={best_auc:.4f} "
                f"at epoch {best_epoch}"
            )
            break

        elapsed = time.time() - time_start
        print(f"  Epoch {epoch}: "
              f"train={train_acc:.3f} "
              f"test={test_acc:.3f} "
              f"auc={auc:.3f} "
              f"loss="
              f"{total_loss/len(train_loader):.4f} "
              f"time={elapsed/60:.1f}min")
        
        epoch_times.append(elapsed)

        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'head':       head.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'train_accs': train_accs,
            'test_accs':  test_accs,
            'auc_scores': auc_scores,
            'bal_accs':   bal_accs,
            'macro_f1_scores': macro_f1_scores,
            'per_class_f1_scores': per_class_f1_scores,
            'ece_scores': ece_scores
        }, ckpt_path)

    if USE_DP:
        epsilon = privacy_engine\
            .get_epsilon(delta=DP_DELTA)
        print(f"\n✅ ε={epsilon:.2f} "
              f"δ={DP_DELTA}")
    else:
        epsilon = float('inf')

    print(f"\n{run_name} seed={seed} "
          f"final test acc: "
          f"{test_accs[-1]:.3f} "
          f"auc: {auc_scores[-1]:.3f}")

    return train_accs, test_accs, \
           auc_scores, epsilon, \
           bal_accs, macro_f1_scores, \
           per_class_f1_scores, ece_scores, epoch_times, peak_memory_mb


# ── Run all experiments with 4 seeds ──────────────────────────────────────────
experiments = {
    'original': (BASE_PATH_ORIGINAL,
                 1000,
                 'Original Sequence'),
    '512':      (BASE_PATH_512,
                 1512,
                 '512 Extra Tokens'),
    '1024':     (BASE_PATH_1024,
                 2024,
                 '1024 Extra Tokens')
}

# check paths exist ✅
for key, (path, _, name) in \
        experiments.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
    else:
        print(f"✅ Found: {name}")

# results[key] = list of
# (train_accs, test_accs,
#  auc_scores, eps) per seed
all_results = {k: [] for k in
               experiments}

for key, (base_path,
          num_tokens,
          run_name) in \
        experiments.items():

    if not os.path.exists(base_path):
        print(f"⚠️ Skipping {run_name} "
              f"— path missing")
        continue

    for seed in SEEDS:
        result = train_classifier(
            base_path    = base_path,
            train_labels = train_labels,
            test_labels  = test_labels,
            num_tokens   = num_tokens,
            run_name     = run_name,
            seed         = seed,
            epochs       = EPOCHS)
        all_results[key].append(result)

    print(f"\n✅ {run_name} "
          f"all {len(SEEDS)} seeds done")


# ── Plot with mean±std ─────────────────────────────────────────────────────────
def ema_smooth(values, alpha=0.9):
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(
            alpha * smoothed[-1] +
            (1 - alpha) * v)
    return smoothed

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
     'original': '#1f77b4',
     '512':      '#ff7f0e',
    '1024':     '#2ca02c',
}
names = {
    'original': 'Original Sequence',
    '512':      '512 Extra Tokens',
    '1024':     '1024 Extra Tokens',
}

for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue

    # collect test accs
    # across seeds ✅
    min_len = min(len(r[1]) for r in runs)
    all_accs = np.array([
        r[1][:min_len]
        for r in runs])

    mean = all_accs.mean(axis=0)
    std  = all_accs.std(axis=0)
    epochs_x = list(range(len(mean)))

    color = colors[key]
    name  = names[key]

    # smoothed mean ✅
    smoothed_mean = ema_smooth(
        mean.tolist(), alpha=0.9)
    smoothed_std  = ema_smooth(
        std.tolist(), alpha=0.9)
    sm = np.array(smoothed_mean)
    ss = np.array(smoothed_std)

    final_accs_leg = np.array([
        np.mean(r[1][-10:])
        for r in runs])
    acc_mean_leg = final_accs_leg.mean()
    acc_std_leg  = final_accs_leg.std()

    # mean line ✅
    ax.plot(epochs_x, sm,
            color=color,
            linewidth=2,
            label=f"{name} "
                  f"({acc_mean_leg:.3f}±"
                  f"{acc_std_leg:.3f})")

    # std band ✅
    ax.fill_between(
        epochs_x,
        sm - ss,
        sm + ss,
        color=color,
        alpha=0.2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title(
    'Classification Test Accuracy, '
    'ViT on Dermamnist 28×28\n'
    f'({len(SEEDS)} seeds, mean±std)',
    fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

plt.tight_layout()
plot_path = os.path.join(
    SAVE_PATH,
    "classification.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n✅ Plot saved: {plot_path}")

# ── Final table ────────────────────────────────────────────────────────────────
print(f"\n{'Run':<22} {'Acc':>8} "
      f"{'±':>6} {'AUC':>8} "
      f"{'±':>6} {'Bal. Acc':>8} "
      f"{'±':>6}")
print("─" * 70)

for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]

    times_to_best = []
    for r in runs:
        auc_list        = r[2]   # auc_scores
        epoch_times_list = r[8]  # epoch_times ← NEW index ✅
        best_idx = int(np.argmax(auc_list))
        times_to_best.append(epoch_times_list[best_idx] / 60)

    times_to_best = np.array(times_to_best)
    print(f"{name:<22} {times_to_best.mean():.1f} ± {times_to_best.std():.1f} min")

    # final epoch stats ✅
    final_accs = np.array([
        np.mean(r[1][-10:])  # ← last 10 ✅
        for r in runs])
    final_aucs = np.array([
        np.mean(r[2][-10:])  # ← last 10 ✅
        for r in runs])
    final_bal_accs = np.array([
    np.mean(r[4][-10:])  # ← r[4] = bal_accs ✅
    for r in runs])

    acc_mean = final_accs.mean()
    acc_std  = final_accs.std()
    auc_mean = final_aucs.mean()
    auc_std  = final_aucs.std()
    bal_acc_mean = final_bal_accs.mean()
    bal_acc_std  = final_bal_accs.std()

    print(f"{name:<22} "
          f"{acc_mean:.3f}   "
          f"{acc_std:.3f}   "
          f"{auc_mean:.3f}   "
          f"{auc_std:.3f}   "
          f"{bal_acc_mean:.3f}   "
          f"{bal_acc_std:.3f}")

    print("─" * 70)
print(f"\n✅ ALL DONE!")
print(f"Results saved to: {SAVE_PATH}")

print("\n── Statistical Significance (AUC) ──")
for pair in [('original', '512'), ('original', '1024'), ('512', '1024')]:
    key_a, key_b = pair
    aucs_a = np.array([
        np.mean(r[2][-10:]) for r in all_results[key_a]])
    aucs_b = np.array([
        np.mean(r[2][-10:]) for r in all_results[key_b]])

    if len(aucs_a) >= 2 and len(aucs_b) >= 2:
        t_stat, p_value = ttest_ind(aucs_a, aucs_b)
        print(f"{key_a} vs {key_b}: "
              f"t={t_stat:.3f}, p={p_value:.4f} "
              f"{'(significant)' if p_value < 0.05 else '(not significant)'}")
    else:
        print(f"{key_a} vs {key_b}: insufficient seeds for test")

print("\n── Peak GPU Memory (MB) ──")
for key in ['original', '512', '1024']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]
    memories = np.array([
        r[9] for r in runs if r[9] is not None])
    if len(memories) > 0:
        print(f"{name:<22} {memories.mean():.1f} ± {memories.std():.1f} MB")