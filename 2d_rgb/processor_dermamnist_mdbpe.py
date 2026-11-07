# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('Agg')  # ← non-interactive backend, no display needed
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn as nn
import random
import medmnist
from medmnist import INFO
import os
import shutil

# ── Setup ─────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 28, 28
MAXIMUM_NUMBER_OF_TOKENS = 1000 + 512
LIMITED_EXAMPLES = None  # use full dataset

# Save path - all outputs go here
SAVE_PATH = os.path.expanduser("/your/save/path")
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

data_flag = 'dermamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor()
])

train_dataset = DataClass(split='train', download=True, transform=transform)
test_dataset  = DataClass(split='test',  transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)  
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))
print(len(train_loader))
print(len(test_loader))

for data, _ in train_loader:
    WIDTH, HEIGHT = data.size()[2], data.size()[3]
    break

print(f"Image size: {WIDTH}x{HEIGHT}")

# ── Step 1: Transcribe ────────────────────────────────────────────────────────
# Remove old folders and start fresh
for folder in ["transcribed_data", "transcribed_data_og", "tokensequences", "tokenshapes"]:
    path = os.path.join(SAVE_PATH, folder)
    if os.path.exists(path):
        shutil.rmtree(path)

FILE_MODULO = 128
assert(len(train_loader.dataset) / FILE_MODULO < 500)

for i in range(FILE_MODULO):
    os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data',    str(i)), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data_og', str(i)), exist_ok=True)

its    = 0
unique = None
labels_train = {}                      
for img, label in train_loader:
    if unique is None:
        unique = torch.zeros_like(img[0, 0].clone()).long()
        for x in range(img.size()[2]):
            for y in range(img.size()[3]):
                unique[x][y] = x * img.size()[2] + y
    img *= 255.0
    img  = img.int()
    for k in range(img.size()[0]):
    # colour quantisation ✅
        r = img[k, 0] // 26  # 0-9 ✅
        g = img[k, 1] // 26  # 0-9 ✅
        b = img[k, 2] // 26  # 0-9 ✅
        combined = r*100 + g*10 + b
        torch.save([unique, combined.clone()],
                   os.path.join(SAVE_PATH, "transcribed_data",    str(its % FILE_MODULO), str(its) + ".dat"))
        torch.save(combined.clone(),
                   os.path.join(SAVE_PATH, "transcribed_data_og", str(its % FILE_MODULO), str(its) + ".dat"))
        labels_train[its] = int(label[k].view(-1)[0])
        its += 1
        if LIMITED_EXAMPLES != None and its >= LIMITED_EXAMPLES:
            break
    if LIMITED_EXAMPLES != None and its >= LIMITED_EXAMPLES:
        break

print(f"Transcribed {its} images")

# ── Step 2: BPE dataset ───────────────────────────────────────────────────────
class UniqueDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.files = []
        for i in range(FILE_MODULO):
            folder = os.path.join(SAVE_PATH, "transcribed_data", str(i))
            for root, dirs, files in os.walk(folder):
                for file in files:
                    self.files.append(str(i) + "/" + file)
        self.files.sort(key=lambda f: int(os.path.basename(f).replace(".dat", "")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(SAVE_PATH, "transcribed_data", self.files[idx])
        unique, classes = torch.load(path, weights_only=False)
        return unique, classes, path

unique_train = UniqueDataset()
print(f"UniqueDataset size: {len(unique_train)}")

for a, _, _ in unique_train:
    display_colours = torch.rand(3, a.size()[0] * a.size()[1])
    break

def visualise_unique(input, save_name=None):
    global display_colours
    assert(len(input.size()) == 2)
    img = display_colours.clone()[:, input.view(-1)].view(3, input.size()[0], input.size()[1])
    plt.imshow(img.permute(1, 2, 0))
    if save_name:
        plt.savefig(os.path.join(SAVE_PATH, save_name))  # ← save instead of show
    plt.close()  # ← close instead of show

# Find largest token
NEW_TOKEN_ID = -1
for unique, classes, path in unique_train:
    NEW_TOKEN_ID = max(NEW_TOKEN_ID, torch.max(classes).item())
NEW_TOKEN_ID += 1

print("NEW TOKEN STARTS AT", NEW_TOKEN_ID)
START_OF_NEW_TOKENS = NEW_TOKEN_ID
CODEWORDS_TO_ADD    = MAXIMUM_NUMBER_OF_TOKENS - START_OF_NEW_TOKENS

def get_anchor(unique, pos_x, pos_y):
    id       = unique[pos_x][pos_y]
    top_left = torch.nonzero(unique == id)[0]
    return top_left

# ── Step 3: MDBPE loop ────────────────────────────────────────────────────────
CODEWORDS_ADDED  = 0
token_path_sizes = {}
time_start       = time.time()
last_output      = time_start
rules            = []

while NEW_TOKEN_ID < MAXIMUM_NUMBER_OF_TOKENS:
    unique_train = UniqueDataset()

    # Pass 1: count
    map_bpe = None
    map_bpe = {}
    it = 0
    for unique, classes, _ in unique_train:
        if time.time() - last_output > 60:
            print("\tDONE WITH", it / len(unique_train) * 100, "% on counting pass")
            last_output = time.time()

        if it == 0:
            visualise_unique(unique, save_name=f"merge_step_{CODEWORDS_ADDED}.png")

        it += 1
        used = {}
        for x in range(unique.size()[0]):
            for y in range(unique.size()[1]):
                for n in [(x + 1, y), (x, y + 1)]:
                    n_x, n_y = n[0], n[1]
                    if n_x >= unique.size()[0] or n_y >= unique.size()[1]:
                        continue
                    if unique[x][y] == unique[n_x][n_y]:
                        continue
                    class_a  = classes[x][y].item()
                    class_b  = classes[n_x][n_y].item()
                    anchor_a = get_anchor(unique, x, y)
                    anchor_b = get_anchor(unique, n_x, n_y)
                    anchor_a_key = tuple(anchor_a.tolist())
                    anchor_b_key = tuple(anchor_b.tolist())
                    if not (anchor_a_key, anchor_b_key) in used:
                        used[(anchor_a_key, anchor_b_key)] = True
                    else:
                        continue
                    v   = (anchor_a - anchor_b)

                    #count occurence of this constellation of features:
                    if not (class_a, class_b, v[0].item(), v[1].item()) in map_bpe:
                        #add if not in map
                        map_bpe[(class_a, class_b, v[0].item(), v[1].item())] = 1
                    else: #just increment
                        map_bpe[(class_a, class_b, v[0].item(), v[1].item())] += 1

    # Pass 2: find most frequent
    max_count = 0          
    max_key = None
    for key in map_bpe:
        if map_bpe[key] > max_count:
            max_count = map_bpe[key]
            max_key = key
    rules.append([max_key[0], max_key[1], max_key[2], max_key[3]])

    # Pass 3: replace
    it          = 0
    total_saved = 0
    for unique, classes, path in unique_train:
        if time.time() - last_output > 60:
            print("\tDONE WITH", it / len(unique_train) * 100, "% on replacement pass")
            last_output = time.time()

        changed      = False
        used_uniques = {}
        for x in range(unique.size()[0]):
            for y in range(unique.size()[1]):
                if unique[x][y].item() in used_uniques:
                    continue
                for n in [(x + 1, y), (x, y + 1)]:
                    n_x, n_y = n[0], n[1]
                    if n_x >= unique.size()[0] or n_y >= unique.size()[1]:
                        continue
                    if unique[n_x][n_y].item() in used_uniques:
                        continue
                    if unique[x][y] == unique[n_x][n_y]:
                        continue
                    class_a = classes[x][y].item()
                    class_b = classes[n_x][n_y].item()
                    if max_key[0] != class_a or max_key[1] != class_b:
                        continue
                    anchor_a = get_anchor(unique, x, y)
                    anchor_b = get_anchor(unique, n_x, n_y)
                    v        = (anchor_a - anchor_b)
                    vx, vy   = v[0].item(), v[1].item()
                    if max_key[2] == vx and max_key[3] == vy:
                        a_unique           = unique[x][y]
                        indices_to_relabel_a          = torch.nonzero(unique == unique[x, y])
                        indices_to_relabel_b          = torch.nonzero(unique == unique[n_x, n_y])
                        for i in range(indices_to_relabel_b.size()[0]):
                            unique[indices_to_relabel_b[i][0]][indices_to_relabel_b[i][1]]  = a_unique
                            classes[indices_to_relabel_b[i][0]][indices_to_relabel_b[i][1]] = NEW_TOKEN_ID
                        for i in range(indices_to_relabel_a.size()[0]):
                            classes[indices_to_relabel_a[i][0]][indices_to_relabel_a[i][1]] = NEW_TOKEN_ID
                        changed = True
                        used_uniques[unique[x][y].item()] = True

        if changed:
            total_saved += 1
            torch.save([unique, classes], path)
        it += 1

    NEW_TOKEN_ID   += 1
    assert(total_saved > 0)
    CODEWORDS_ADDED += 1

    time_so_far = time.time() - time_start
    time_so_far = time_so_far / CODEWORDS_ADDED * (CODEWORDS_TO_ADD - CODEWORDS_ADDED)
    print(f"\n---> DONE ADDING CODEWORD {CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}, Time left (est.): {time_so_far/60:.1f} mins\n")

# Save rules
torch.save(rules, os.path.join(SAVE_PATH, "rules.dat"))
print("✅ rules.dat saved!")

# ── Step 4: Export token shapes & sequences ───────────────────────────────────
unique_train = UniqueDataset()

os.makedirs(os.path.join(SAVE_PATH, 'tokenshapes'),   exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'tokensequences'), exist_ok=True)
for k in range(FILE_MODULO):
    os.makedirs(os.path.join(SAVE_PATH, f"tokensequences/{k}"), exist_ok=True)

exported    = {}
token_sizes = {}
for unique, classes, path in unique_train:
    path_v2         = path.replace("transcribed_data", "transcribed_data_og")
    tokens_original = torch.load(path_v2, weights_only=False)
    total_size      = 0
    uniques_exported = {}

    for x in range(unique.size()[0]):
        for y in range(unique.size()[1]):
            class_to_export = classes[x][y].item()
            if class_to_export not in exported:
                uid_to_export = unique[x][y].item()
                most_left     = WIDTH + 1
                most_top      = HEIGHT + 1
                indices       = torch.nonzero(unique == uid_to_export)
                for i in range(indices.size()[0]):
                    most_left = min(most_left, indices[i][0])
                    most_top  = min(most_top,  indices[i][1])
                offset_x        = most_left - x
                crop_unique     = unique.clone()[most_left:, most_top:]
                crop_classes    = classes.clone()[most_left:, most_top:]
                crop_original   = tokens_original.clone()[most_left:, most_top:]
                wrong           = torch.nonzero(crop_unique != uid_to_export)
                for i in range(wrong.size()[0]):
                    crop_classes[wrong[i][0]][wrong[i][1]]  = -1
                    crop_original[wrong[i][0]][wrong[i][1]] = -1
                while (crop_classes[-1] != -1).sum() == 0:
                    crop_classes  = crop_classes[:-1]
                    crop_original = crop_original[:-1]
                while (crop_classes[:, -1] != -1).sum() == 0:
                    crop_classes  = crop_classes[:, :-1]
                    crop_original = crop_original[:, :-1]
                token_size                          = (crop_original != -1).sum().item()
                token_sizes[classes[x, y].item()]   = token_size
                total_size                         += token_size
                uniques_exported[uid_to_export]     = True
                torch.save([crop_original, offset_x],
                           os.path.join(SAVE_PATH, "tokenshapes", str(class_to_export) + ".dat"))
                exported[class_to_export] = True
            elif not unique[x][y].item() in uniques_exported:
                total_size += token_sizes[classes[x, y].item()]
                uniques_exported[unique[x][y].item()] = True

    assert(total_size == WIDTH * HEIGHT)

print("✅ tokenshapes exported!")

# Export token sequences
index            = 0
sequence_lengths = []
for unique, classes, path in unique_train:
    file_idx = int(os.path.basename(path).replace(".dat", ""))   # ORDER FIX: index of the source image
    sequence = []
    used_ids = {}
    for x in range(unique.size()[0]):
        for y in range(unique.size()[1]):
            if unique[x][y].item() in used_ids:
                continue
            used_ids[unique[x][y].item()] = True
            sequence.append([classes[x][y].item(), x, y])
    if index == 0:
        visualise_unique(unique, save_name=f"final_unique_{index}.png")
    sequence_lengths.append(len(sequence))
    if index % 100 == 0:
        avg = sum(sequence_lengths) / len(sequence_lengths)
        print(f"DONE EXPORTING {index}/{len(unique_train)} SEQUENCES, AVG LENGTH: {avg:.1f}, "
              f"{avg/(WIDTH*HEIGHT)*100:.1f}% of initial size")
    torch.save(sequence,
               os.path.join(SAVE_PATH, "tokensequences", str(file_idx % FILE_MODULO), str(file_idx) + ".dat"))
    index += 1

print("✅ tokensequences exported!")
print(f"✅ ALL DONE! Files saved to {SAVE_PATH}")
print(f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths):.1f} vs original {WIDTH*HEIGHT}")

# ── Dataset Pruning (top 5%) ───────────
# Remove longest 5% sequences ✅
# Reduces padding overhead ✅
# Faster training ✅
# Matches paper protocol ✅

print("\n── Dataset Pruning ───────────────")

# calculate all sequence lengths
all_lengths = []
all_files   = []

for i in range(FILE_MODULO):
    folder = os.path.join(
        SAVE_PATH, "tokensequences", str(i))
    if not os.path.exists(folder):
        continue
    for f in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, f)
        seq   = torch.load(
            fpath, weights_only=False)
        all_lengths.append(len(seq))
        all_files.append(fpath)

print(f"Total sequences: {len(all_files)}")
print(f"Max length before: "
      f"{max(all_lengths)}")
print(f"Avg length before: "
      f"{sum(all_lengths)/len(all_lengths):.1f}")

# find 95th percentile threshold
threshold = np.percentile(
    all_lengths, 95)
print(f"95th percentile: {threshold:.0f}")

# remove top 5% longest
removed = 0
for fpath, length in zip(
        all_files, all_lengths):
    if length > threshold:
        os.remove(fpath)
        removed += 1

print(f"Removed {removed} sequences "
      f"({removed/len(all_files)*100:.1f}%)")
print(f"Remaining: "
      f"{len(all_files) - removed} sequences")

# recalculate stats
remaining_lengths = [
    l for l in all_lengths
    if l <= threshold]
if remaining_lengths:
    print(f"Max length after: "
          f"{max(remaining_lengths)}")
    print(f"Avg length after: "
          f"{sum(remaining_lengths)/len(remaining_lengths):.1f}")

print("✅ Dataset pruning done!")

# ── Step 5: Test Tokenization ──────────
print("\n── Step 5: Test Tokenization ─────")

import shutil

# create test folders
test_trans_path = os.path.join(
    SAVE_PATH, "transcribed_test")
test_seq_path   = os.path.join(
    SAVE_PATH, "tokensequences_test")

if os.path.exists(test_trans_path):
    shutil.rmtree(test_trans_path)
if os.path.exists(test_seq_path):
    shutil.rmtree(test_seq_path)

for i in range(FILE_MODULO):
    os.makedirs(os.path.join(
        test_trans_path, str(i)),
        exist_ok=True)
    os.makedirs(os.path.join(
        test_seq_path, str(i)),
        exist_ok=True)

# transcribe test images
print("Transcribing test images...")
its_test   = 0
unique_tst = None
labels_test = {}                       # ORDER FIX: file index -> label

for img, label in test_loader:
    if unique_tst is None:
        unique_tst = torch.zeros_like(
            img[0, 0].clone()).long()
        for x in range(img.size()[2]):
            for y in range(img.size()[3]):
                unique_tst[x][y] = \
                    x * img.size()[2] + y
    img *= 255.0
    img  = img.int()
    for k in range(img.size()[0]):
        r = img[k, 0] // 26  # ✅
        g = img[k, 1] // 26  # ✅
        b = img[k, 2] // 26  # ✅
        combined = r*100 + g*10 + b
        torch.save(
            [unique_tst.clone(),
             combined.clone()],
            os.path.join(
                test_trans_path,
                str(its_test % FILE_MODULO),
                str(its_test) + ".dat"))
        labels_test[its_test] = int(label[k].view(-1)[0])
        its_test += 1

print(f"Transcribed {its_test} test images")

# load all test files
test_files = []
for i in range(FILE_MODULO):
    folder = os.path.join(
        test_trans_path, str(i))
    if not os.path.exists(folder):
        continue
    for f in sorted(os.listdir(folder)):
        test_files.append(
            os.path.join(folder, f))

print(f"Found {len(test_files)} test files")
print(f"Applying {len(rules)} rules...")

# apply rules (Pass 3 only) ✅
current_token = START_OF_NEW_TOKENS

for rule_idx, rule in enumerate(rules):
    class_a = rule[0]
    class_b = rule[1]
    vx      = rule[2]
    vy      = rule[3]

    for path in test_files:
        unique_t, classes_t = torch.load(
            path, weights_only=False)
        changed      = False
        used_uniques = {}

        for x in range(unique_t.size()[0]):
            for y in range(unique_t.size()[1]):
                if unique_t[x][y].item() \
                        in used_uniques:
                    continue
                for n in [(x+1,y),(x,y+1)]:
                    n_x, n_y = n[0], n[1]
                    if n_x >= unique_t.size()[0] \
                    or n_y >= unique_t.size()[1]:
                        continue
                    if unique_t[n_x][n_y]\
                            .item() \
                            in used_uniques:
                        continue
                    if unique_t[x][y] == \
                       unique_t[n_x][n_y]:
                        continue
                    ca = classes_t[x][y]\
                        .item()
                    cb = classes_t[n_x][n_y]\
                        .item()
                    if class_a != ca or \
                       class_b != cb:
                        continue
                    anchor_a = get_anchor(
                        unique_t, x, y)
                    anchor_b = get_anchor(
                        unique_t, n_x, n_y)
                    v   = (anchor_a - anchor_b)
                    vx_ = v[0].item()
                    vy_ = v[1].item()
                    if vx == vx_ and \
                       vy == vy_:
                        a_u   = unique_t[x][y]
                        idx_a = torch.nonzero(
                            unique_t == \
                            unique_t[x, y])
                        idx_b = torch.nonzero(
                            unique_t == \
                            unique_t[n_x, n_y])
                        for i in range(
                                idx_b.size()[0]):
                            unique_t[
                                idx_b[i][0]]\
                                [idx_b[i][1]]\
                                = a_u
                            classes_t[
                                idx_b[i][0]]\
                                [idx_b[i][1]]\
                                = current_token
                        for i in range(
                                idx_a.size()[0]):
                            classes_t[
                                idx_a[i][0]]\
                                [idx_a[i][1]]\
                                = current_token
                        changed = True
                        used_uniques[
                            unique_t[x][y]\
                            .item()] = True

        if changed:
            torch.save(
                [unique_t, classes_t],
                path)

    current_token += 1
    if (rule_idx + 1) % 50 == 0:
        print(f"  Applied {rule_idx+1}"
              f"/{len(rules)} rules")

print("✅ Rules applied!")

# export test sequences
print("Exporting test sequences...")
index_test       = 0
test_seq_lengths = []

for path in test_files:
    file_idx = int(os.path.basename(path).replace(".dat", ""))   
    unique_t, classes_t = torch.load(
        path, weights_only=False)
    sequence = []
    used_ids = {}
    for x in range(unique_t.size()[0]):
        for y in range(unique_t.size()[1]):
            uid = unique_t[x][y].item()
            if uid in used_ids:
                continue
            used_ids[uid] = True
            sequence.append([
                classes_t[x][y].item(),
                x, y])
    test_seq_lengths.append(len(sequence))
    torch.save(
        sequence,
        os.path.join(
            test_seq_path,
            str(file_idx % FILE_MODULO),
            str(file_idx) + ".dat"))
    index_test += 1

avg_test = sum(test_seq_lengths) / \
           len(test_seq_lengths)
print(f"✅ Test sequences exported!")
print(f"   {index_test} sequences")
print(f"   Avg length: {avg_test:.1f}")

# save metadata
metadata = {
    "NEW_TOKEN_ID":             NEW_TOKEN_ID,
    "START_OF_NEW_TOKENS":      START_OF_NEW_TOKENS,
    "MAXIMUM_NUMBER_OF_TOKENS": MAXIMUM_NUMBER_OF_TOKENS,
    "WIDTH":                    WIDTH,
    "HEIGHT":                   HEIGHT,
    "LIMITED_EXAMPLES":         LIMITED_EXAMPLES,
    "FILE_MODULO":              FILE_MODULO,
    "total_images":             its,
    "total_test_images":        its_test,
    "mdbpe_type":               "2D",
    "dataset":                  "dermamnist_512"
}
torch.save(metadata,
    os.path.join(SAVE_PATH, "metadata.pt"))
print("✅ metadata.pt saved!")

for split, labs in (("train", labels_train), ("test", labels_test)):
    torch.save(labs, os.path.join(SAVE_PATH, f"recovered_labels_{split}.pt"))
    torch.save({k: k for k in labs}, os.path.join(SAVE_PATH, f"recovered_index_{split}.pt"))
print("✅ labels saved (recovered_labels_*.pt, recovered_index_*.pt; file k = image k)")
print(f"\n✅ FULLY DONE!")
print(f"   Train: {its} sequences ✅")
print(f"   Test:  {its_test} sequences ✅")
print(f"   Rules: {len(rules)} ✅")