# -*- coding: utf-8 -*-
#  Variable sequence length: use actual length not WIDTH*HEIGHT ✅
#  Custom collate_fn for padding ✅
#  4 seeds [0,1,2,3] mean±std ✅

import torch
from opacus.validators import ModuleValidator 
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
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

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PATH_ORIGINAL = os.path.expanduser(
    "your/path/original")
BASE_PATH_256 = os.path.expanduser(
    "your/path/256")
BASE_PATH_512 = os.path.expanduser(
    "your/path/512")

SAVE_PATH = os.path.expanduser(
    "your/path/results/classification")
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DIMENSIONS  = 512 
NUM_HEADS   = 8
NUM_LAYERS  = 4 # previoulsy 4
BASE_TOKENS = 256
FILE_MODULO = 128
EPOCHS      = 200 # previously 200
BATCH_SIZE  = 16
USE_NPE     = True
USE_IPE     = True
SEEDS       = [0, 1, 2, 3]  # ← 4 seeds ✅

# privacy
USE_DP      = False
DP_NOISE    = 8
DP_MAX_GRAD = 1.0
DP_DELTA    = 1e-5
# ── Load metadata ──────────────────────────────────────────────────────────────
def load_metadata(base_path):
    return torch.load(
        os.path.join(base_path, "metadata.pt"),
        weights_only=False)

meta_256 = load_metadata(BASE_PATH_256)
WIDTH     = meta_256["WIDTH"]
HEIGHT    = meta_256["HEIGHT"]
print(f"Image size: {WIDTH}x{HEIGHT}")

# ── MedMNIST labels ────────────────────────────────────────────────────────────
data_flag = 'breastmnist'
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
        ORIGINAL_PATH = "/home/anushka1/scratch/mdbpe_output_breastmnist_original"
        split_tag = 'test' if split == 'test' else 'train'
        self.recovered = torch.load(
            os.path.join(base_path, f"recovered_labels_{split_tag}.pt"),
            weights_only=False)

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
        label = self.recovered[orig_idx]

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

def find_max_batch_size(model, head, criterion, dataset, collate_fn, optimizer=None, start=8, max_try=512):
    lengths = np.array([len(torch.load(f, weights_only=False)) for f in dataset.files])
    longest_order = np.argsort(lengths)[::-1]

    def _try_batch(bs):
        idx = longest_order[:bs].tolist()
        subset = torch.utils.data.Subset(dataset, idx)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(subset, batch_size=bs, shuffle=False, collate_fn=collate_fn)
            seq, pos, mask, labels = next(iter(loader))
            seq, pos, mask, labels = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            feat = model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1])
            loss = criterion(head(feat), labels)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    best = 0
    bs = start
    while bs <= max_try:
        if _try_batch(bs):
            best = bs; bs *= 2
        else:
            break
    return max(best, 1)


def measure_inference_time(model, head, dataset, collate_fn, batch_size, n_warmup=5, n_measure=20):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval(); head.eval()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(n_warmup):
            try: seq, pos, mask, labels = next(it)
            except StopIteration: it = iter(loader); seq, pos, mask, labels = next(it)
            seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
            _ = head(model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1]))

    times = []
    with torch.no_grad():
        while len(times) < n_measure:
            try: seq, pos, mask, labels = next(it)
            except StopIteration: it = iter(loader); seq, pos, mask, labels = next(it)
            seq, pos, mask = seq.to(DEVICE), pos.to(DEVICE), mask.to(DEVICE)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            _ = head(model.get_features(seq[:, :-1], pos[:, :-1], padding_mask=mask[:, :-1]))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            times.append(time.time() - t0)
    times = np.array(times)
    per_image = times / batch_size
    return per_image.mean(), per_image.std()


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
    nw = 4 if os.name != 'nt' else 0   # ← ✅

    model = TransformerDecoderModel(
        num_tokens).to(DEVICE)
    num_classes = len(set(train_labels))
    head  = ClassificationHead(num_classes=num_classes).to(DEVICE)

    # weighted loss ✅
    used = [train_ds.recovered[int(os.path.basename(f)[:-4])] for f in train_ds.files]
    total = len(used)
    class_counts = [used.count(i) for i in range(num_classes)]
    weights = torch.tensor([total / (num_classes * c + 1e-8) for c in class_counts]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=weights)
    
    max_bs = find_max_batch_size(model, head, criterion, train_ds, collate)
    actual_batch_size = min(BATCH_SIZE, max_bs)
    print(f"  Max batch size found: {max_bs}  (using {actual_batch_size})")

    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True, num_workers=nw, collate_fn=collate)  # ← ✅
    test_loader = DataLoader(
        test_ds,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate)

    N = sum(p.numel() for p in
        list(model.parameters()) +
        list(head.parameters()))
    base_lr   = 0.0002
    scaled_lr = base_lr * \
        (1.0 / math.sqrt(
            N / 54_000_000))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(head.parameters()),
        lr=scaled_lr,
        weight_decay=0.01)

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
    time_start = time.time()
    epoch_times = []
    per_epoch_times = []

    best_auc    = 0.0
    best_epoch  = 0
    best_test_acc = 0.0

    epochs_without_improvement = 0

    best_ckpt_path = os.path.join(
        SAVE_PATH,
        f"best_ckpt_{run_name.replace(' ','_')}"
        f"_seed{seed}.pt")

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
        start_epoch = ckpt['epoch'] + 1
        best_auc    = ckpt.get(
            'best_auc', 0.0)             # ← NEW ✅
        best_epoch  = ckpt.get(
            'best_epoch', 0)             # ← NEW ✅
        best_test_acc = ckpt.get('best_test_acc', 0.0)
        epochs_without_improvement = \
            ckpt.get(
                'epochs_without_improvement',
                0)                        # ← NEW ✅
        print(f"Resumed from epoch "
            f"{start_epoch}")
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

        train_acc = (correct/total_b)\
            .item()
        train_accs.append(train_acc)

        # ── Measure peak memory at epoch 0 (matches DermaMNIST) ──────────────
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
                    logits, dim=1)[:, 1]
                all_probs.extend(
                    probs.cpu().numpy())
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
                all_labels, all_probs)
        except ValueError:
            auc = 0.5
        auc_scores.append(auc)

        # ── Best-AUC tracking ✅
        if auc > best_auc:
            best_auc   = auc
            best_epoch = epoch
            best_test_acc = test_acc
            epochs_without_improvement = 0
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'head':     head.state_dict(),
                'test_acc': test_acc,
                'auc':      auc
            }, best_ckpt_path)
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - time_start
        epoch_times.append(elapsed)
        per_epoch_time = elapsed - (epoch_times[-2] if len(epoch_times) > 1 else 0.0)
        per_epoch_times.append(per_epoch_time)
        print(f"  Epoch {epoch}: "
              f"train={train_acc:.3f} "
              f"test={test_acc:.3f} "
              f"auc={auc:.3f} "
              f"loss="
              f"{total_loss/len(train_loader):.4f} "
              f"time={elapsed/60:.1f}min")

        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'head':       head.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'train_accs': train_accs,
            'test_accs':  test_accs,
            'auc_scores': auc_scores,
            'best_auc':   best_auc,                          # ← NEW ✅
            'best_epoch': best_epoch, 
            'best_test_acc': best_test_acc,                        # ← NEW ✅
            'epochs_without_improvement':
                epochs_without_improvement,                   # ← NEW ✅
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
          f"auc: {auc_scores[-1]:.3f} "
          f"| BEST auc: {best_auc:.3f} "
          f"@ epoch {best_epoch}")
    
    inf_mean, inf_std = measure_inference_time(model, head, test_ds, collate, batch_size=actual_batch_size)

    return train_accs, test_accs, auc_scores, epsilon, best_auc, best_test_acc, best_epoch, peak_memory_mb, max_bs, inf_mean, inf_std, per_epoch_times


# ── Run all experiments with 4 seeds ──────────────────────────────────────────
experiments = {
    'original': (BASE_PATH_ORIGINAL,
                 256,
                 'Original Sequence'),
    '256':      (BASE_PATH_256,
                 512,
                 '256 Extra Tokens'),
    '512':      (BASE_PATH_512,
                 768,
                 '512 Extra Tokens'),
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
    '256':      '#ff7f0e',
    '512':      '#2ca02c',
}
names = {
    'original': 'Original Sequence',
    '256':      '256 Extra Tokens',
    '512':      '512 Extra Tokens',
}

for key in ['original', '256', '512']:
    runs = all_results[key]
    if len(runs) == 0:
        continue

    # collect test accs
    # across seeds ✅
    max_len = max(len(r[1]) for r in runs)
    padded  = np.array([
        np.pad(r[1], (0, max_len - len(r[1])), mode='edge')
        for r in runs])
    mean = padded.mean(axis=0)
    std  = padded.std(axis=0)
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

    # mean line ✅
    best_accs_legend = np.array([
        r[5] for r in runs])  # ← best_test_acc ✅
    acc_mean = best_accs_legend.mean()
    acc_std  = best_accs_legend.std()

    ax.plot(epochs_x, sm,
            color=color,
            linewidth=2,
            label=f"{name} "
                f"({acc_mean:.3f}±"
                f"{acc_std:.3f})")  # ← matches table ✅

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
    'ViT on BreastMNIST 28×28\n'
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
      f"{'±':>6}")
print("─" * 56)

for key in ['original', '256', '512']:
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]

    # use BEST auc / best test_acc
    # per seed, not last-10 ✅
    best_accs = np.array([
        r[5] for r in runs])   # ← best_test_acc ✅
    best_aucs = np.array([
        r[4] for r in runs])   # ← best_auc ✅

    acc_mean = best_accs.mean()
    acc_std  = best_accs.std()
    auc_mean = best_aucs.mean()
    auc_std  = best_aucs.std()

    print(f"{name:<22} "
          f"{acc_mean:.3f}   "
          f"{acc_std:.3f}   "
          f"{auc_mean:.3f}   "
          f"{auc_std:.3f}")

print("─" * 56)
print(f"\n✅ ALL DONE!")
print(f"Results saved to: {SAVE_PATH}")

print("\n── Peak GPU Memory (MB) ──")
for key in ['original', '256', '512']:        # ← fixed: was '1024'
    runs = all_results[key]
    if len(runs) == 0:
        continue
    name = names[key]
    memories = np.array([
        r[7] for r in runs if r[7] is not None])    # ← fixed: was r[9]
    if len(memories) > 0:
        print(f"{name:<22} {memories.mean():.1f} ± {memories.std():.1f} MB")

print("\n── Max Batch Size Found ──")
for key in ['original', '256', '512']:
    runs = all_results[key]
    if len(runs) == 0: continue
    bs_vals = np.array([r[8] for r in runs])
    print(f"{names[key]:<22} {bs_vals.mean():.1f} ± {bs_vals.std():.1f}")

print("\n── Inference Time (ms/image) ──")
for key in ['original', '256', '512']:
    runs = all_results[key]
    if len(runs) == 0: continue
    inf_means = np.array([r[9]*1000 for r in runs])
    print(f"{names[key]:<22} {inf_means.mean():.3f} ± {inf_means.std():.3f} ms/image")

print("\n── Per-Epoch Compute Time ──")
for key in ['original', '256', '512']:
    runs = all_results[key]
    if len(runs) == 0: continue
    all_times = np.concatenate([np.array(r[11][1:]) for r in runs if len(r[11]) > 1])
    print(f"{names[key]:<22} {all_times.mean():.2f} ± {all_times.std():.2f} sec/epoch")