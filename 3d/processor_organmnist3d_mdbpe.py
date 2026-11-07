# -*- coding: utf-8 -*-
# OrganMNIST3D @ 28x28x28, 512 extra tokens (256 base + 512 = 768 total)
#
# Earlier fixes:
# FIXES applied vs. the script that produced the log you shared:
#   1. CORRECTNESS: raw-tensor dedup bug fixed. The running version used
#      `used[(anchor_a, anchor_b)]` with anchor_a/anchor_b as raw tensors
#      from get_anchor() — tensors hash by object identity, not value, so
#      that dedup never actually fired, inflating map_bpe counts by
#      boundary-contact length instead of true occurrence count.
#   2. SPEED: anchor caching (build_anchor_map, ~8x faster per the observed
#      33min/step vs ~4min/step expected-with-caching gap).
#   3. SPEED: parallelized across N_WORKERS CPU cores for Pass 1 / Pass 3.
#   4. RESILIENCE: checkpoint saved every SINGLE step (not every N), so a
#      Slurm walltime kill can never leave the checkpoint out of sync with
#      the on-disk image state.
#
# IMPORTANT: your Slurm submission must request --cpus-per-task=N_WORKERS
# or this oversubscribes a smaller core allocation.

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn as nn
import random
import medmnist
from medmnist import INFO
import os
import shutil
import multiprocessing as mp

# ── Parallelism settings ───────────────────────────────────────────────────
N_WORKERS = 8  # must match --cpus-per-task in your Slurm submission

TARGET = 28
def downsample_volume(vol):
    v = vol.float().unsqueeze(0).unsqueeze(0)
    v = F.interpolate(v, size=(TARGET, TARGET, TARGET), mode='trilinear', align_corners=False)
    return v.squeeze().long()

# ── Setup ─────────────────────────────────────────────────────────────────────
MAXIMUM_NUMBER_OF_TOKENS = 256 + 512
LIMITED_EXAMPLES = None

SAVE_PATH = os.path.expanduser("/your/path/mdbpe_output_organmnist3d_512")
os.makedirs(SAVE_PATH, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_PATH, "mdbpe_checkpoint.pt")
CHECKPOINT_EVERY_N_STEPS = 1  # every step, to avoid the resume-mismatch risk entirely

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
print(f"Using {N_WORKERS} worker processes for Pass 1 / Pass 3")

data_flag = 'organmnist3d'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
transform = None

FILE_MODULO = 128

# ── 3D anchor caching: same output as full-scan get_anchor, ~8x faster ───────
def build_anchor_map(unique):
    anchor_map = {}
    for x in range(unique.size()[0]):
        for y in range(unique.size()[1]):
            for z in range(unique.size()[2]):
                uid = unique[x][y][z].item()
                if uid not in anchor_map:
                    anchor_map[uid] = (x, y, z)
    return anchor_map

def get_anchor_cached(anchor_map, unique, x, y, z):
    return anchor_map[unique[x][y][z].item()]

# kept for Step 4/5 (smaller, already-mostly-merged tensors)
def get_anchor(unique, pos_x, pos_y, pos_z):
    id_ = unique[pos_x][pos_y][pos_z]
    return torch.nonzero(unique == id_)[0]

NEIGHBORS_3D = [(1,0,0), (0,1,0), (0,0,1)]

# ── Pass 1 worker: counts constellations for one CHUNK of images ─────────────
def count_pass_chunk_worker(file_paths):
    local_map_bpe = {}
    for path in file_paths:
        unique, classes = torch.load(path, weights_only=False)
        anchor_map = build_anchor_map(unique)
        used = {}
        sx, sy, sz = unique.size()
        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    for dx, dy, dz in NEIGHBORS_3D:
                        n_x, n_y, n_z = x+dx, y+dy, z+dz
                        if n_x >= sx or n_y >= sy or n_z >= sz:
                            continue
                        if unique[x][y][z] == unique[n_x][n_y][n_z]:
                            continue
                        class_a = classes[x][y][z].item()
                        class_b = classes[n_x][n_y][n_z].item()
                        anchor_a = get_anchor_cached(anchor_map, unique, x, y, z)
                        anchor_b = get_anchor_cached(anchor_map, unique, n_x, n_y, n_z)

                        # anchor_a/anchor_b are plain (x,y,z) tuples already —
                        # value-hashable directly, no .tolist() needed (that
                        # was only ever necessary for the tensor version)
                        if (anchor_a, anchor_b) in used:
                            continue
                        used[(anchor_a, anchor_b)] = True

                        vx = anchor_a[0] - anchor_b[0]
                        vy = anchor_a[1] - anchor_b[1]
                        vz = anchor_a[2] - anchor_b[2]

                        key = (class_a, class_b, vx, vy, vz)
                        local_map_bpe[key] = local_map_bpe.get(key, 0) + 1
    return local_map_bpe

# ── Pass 3 worker: replaces + saves one CHUNK of images ──────────────────────
def replace_pass_chunk_worker(args):
    file_paths, max_key, new_token_id = args
    changed_count = 0
    for path in file_paths:
        unique, classes = torch.load(path, weights_only=False)
        anchor_map = build_anchor_map(unique)
        changed = False
        used_uniques = {}
        sx, sy, sz = unique.size()
        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    if unique[x][y][z].item() in used_uniques:
                        continue
                    for dx, dy, dz in NEIGHBORS_3D:
                        n_x, n_y, n_z = x+dx, y+dy, z+dz
                        if n_x >= sx or n_y >= sy or n_z >= sz:
                            continue
                        if unique[n_x][n_y][n_z].item() in used_uniques:
                            continue
                        if unique[x][y][z] == unique[n_x][n_y][n_z]:
                            continue
                        class_a = classes[x][y][z].item()
                        class_b = classes[n_x][n_y][n_z].item()
                        if max_key[0] != class_a or max_key[1] != class_b:
                            continue
                        anchor_a = get_anchor_cached(anchor_map, unique, x, y, z)
                        anchor_b = get_anchor_cached(anchor_map, unique, n_x, n_y, n_z)
                        vx = anchor_a[0] - anchor_b[0]
                        vy = anchor_a[1] - anchor_b[1]
                        vz = anchor_a[2] - anchor_b[2]
                        if max_key[2] == vx and max_key[3] == vy and max_key[4] == vz:
                            a_unique = unique[x][y][z]
                            b_unique_id = unique[n_x, n_y, n_z].item()
                            indices_to_relabel_a = torch.nonzero(unique == unique[x, y, z])
                            indices_to_relabel_b = torch.nonzero(unique == unique[n_x, n_y, n_z])
                            for i in range(indices_to_relabel_b.size()[0]):
                                unique[indices_to_relabel_b[i][0]][indices_to_relabel_b[i][1]][indices_to_relabel_b[i][2]] = a_unique
                                classes[indices_to_relabel_b[i][0]][indices_to_relabel_b[i][1]][indices_to_relabel_b[i][2]] = new_token_id
                            for i in range(indices_to_relabel_a.size()[0]):
                                classes[indices_to_relabel_a[i][0]][indices_to_relabel_a[i][1]][indices_to_relabel_a[i][2]] = new_token_id
                            changed = True
                            used_uniques[unique[x][y][z].item()] = True
                            anchor_map.pop(b_unique_id, None)

        if changed:
            torch.save([unique, classes], path)
            changed_count += 1
    return changed_count

def chunkify(lst, n_chunks):
    k, m = divmod(len(lst), n_chunks)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n_chunks)]


def main():
    train_dataset = DataClass(split='train', download=True, transform=transform)
    test_dataset  = DataClass(split='test',  transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)  # ORDER FIX: dataset order, so file k = volume k
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

    print(len(train_dataset)); print(len(test_dataset))
    print(len(train_loader)); print(len(test_loader))

    global DEPTH, WIDTH, HEIGHT
    DEPTH = WIDTH = HEIGHT = None
    for data, _ in train_loader:
        print(f"Data shape: {data.shape}")
        DEPTH = WIDTH = HEIGHT = TARGET
        break

    print(f"Volume: {DEPTH}×{WIDTH}×{HEIGHT}")
    print(f"Total voxels: {DEPTH*WIDTH*HEIGHT}")

    RESUMING = os.path.exists(CHECKPOINT_PATH)

    if not RESUMING:
        for folder in ["transcribed_data", "transcribed_data_og", "tokensequences", "tokenshapes"]:
            path = os.path.join(SAVE_PATH, folder)
            if os.path.exists(path):
                shutil.rmtree(path)
        assert(len(train_loader.dataset) / FILE_MODULO < 500)
        for i in range(FILE_MODULO):
            os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data', str(i)), exist_ok=True)
            os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data_og', str(i)), exist_ok=True)

        its = 0
        unique = None
        for img, label in train_loader:
            img = img.float() * 255.0
            img = img.int()
            if unique is None:
                unique = torch.zeros_like(torch.zeros(TARGET, TARGET, TARGET)).long()
                for x in range(TARGET):
                    for y in range(TARGET):
                        for z in range(TARGET):
                            unique[x][y][z] = x*TARGET*TARGET + y*TARGET + z
            for k in range(img.size()[0]):
                vol = img[k,0].clone()
                vol = downsample_volume(vol)
                torch.save([unique.clone(), vol],
                           os.path.join(SAVE_PATH, "transcribed_data", str(its % FILE_MODULO), str(its) + ".dat"))
                torch.save(vol,
                           os.path.join(SAVE_PATH, "transcribed_data_og", str(its % FILE_MODULO), str(its) + ".dat"))
                its += 1
                if LIMITED_EXAMPLES != None and its >= LIMITED_EXAMPLES:
                    break
            if LIMITED_EXAMPLES != None and its >= LIMITED_EXAMPLES:
                break
        print(f"Transcribed {its} 3D volumes ✅")
    else:
        print("Resuming — skipping transcription step (already done in a previous run).")
        its = sum(len(files) for i in range(FILE_MODULO)
                  for _, _, files in os.walk(os.path.join(SAVE_PATH, "transcribed_data", str(i))))

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

        def full_paths(self):
            return [os.path.join(SAVE_PATH, "transcribed_data", f) for f in self.files]

    unique_train = UniqueDataset()
    print(f"UniqueDataset size: {len(unique_train)}")

    global display_colours
    for a, _, _ in unique_train:
        n_unique = a.size()[0]*a.size()[1]*a.size()[2]
        display_colours = torch.rand(3, n_unique)
        break

    def visualise_unique_3d(unique, save_name=None):
        assert len(unique.size()) == 3
        mid = unique.size()[0] // 2
        slice_2d = unique[mid]
        flat = slice_2d.reshape(-1)
        flat = flat % display_colours.size()[1]
        img = display_colours.clone()[:, flat].reshape(3, slice_2d.size()[0], slice_2d.size()[1])
        plt.imshow(img.permute(1, 2, 0).numpy())
        if save_name:
            plt.savefig(os.path.join(SAVE_PATH, save_name))
        plt.close()

    if RESUMING:
        print(f"\n### RESUMING from checkpoint: {CHECKPOINT_PATH} ###\n")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        NEW_TOKEN_ID = ckpt['NEW_TOKEN_ID']
        START_OF_NEW_TOKENS = ckpt['START_OF_NEW_TOKENS']
        CODEWORDS_TO_ADD = ckpt['CODEWORDS_TO_ADD']
        CODEWORDS_ADDED = ckpt['CODEWORDS_ADDED']
        rules = ckpt['rules']
        time_start = ckpt.get('time_start', time.time())
        print(f"Resuming at NEW_TOKEN_ID={NEW_TOKEN_ID}, "
              f"CODEWORDS_ADDED={CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}, {len(rules)} rules saved")
    else:
        NEW_TOKEN_ID = -1
        for unique, classes, path in unique_train:
            NEW_TOKEN_ID = max(NEW_TOKEN_ID, torch.max(classes).item())
        NEW_TOKEN_ID += 1
        print("NEW TOKEN STARTS AT", NEW_TOKEN_ID)
        START_OF_NEW_TOKENS = NEW_TOKEN_ID
        CODEWORDS_TO_ADD = MAXIMUM_NUMBER_OF_TOKENS - START_OF_NEW_TOKENS
        CODEWORDS_ADDED = 0
        rules = []
        time_start = time.time()

    def save_checkpoint():
        torch.save({
            'NEW_TOKEN_ID': NEW_TOKEN_ID, 'START_OF_NEW_TOKENS': START_OF_NEW_TOKENS,
            'CODEWORDS_TO_ADD': CODEWORDS_TO_ADD, 'CODEWORDS_ADDED': CODEWORDS_ADDED,
            'rules': rules, 'time_start': time_start,
        }, CHECKPOINT_PATH)

    pool = mp.Pool(processes=N_WORKERS)

    while NEW_TOKEN_ID < MAXIMUM_NUMBER_OF_TOKENS:
        unique_train = UniqueDataset()
        all_paths = unique_train.full_paths()

        first_unique, _ = torch.load(all_paths[0], weights_only=False)
        visualise_unique_3d(first_unique, save_name=f"merge_step_{CODEWORDS_ADDED}.png")

        chunks = chunkify(all_paths, N_WORKERS)
        partial_dicts = pool.map(count_pass_chunk_worker, chunks)
        map_bpe = {}
        for pd in partial_dicts:
            for key, count in pd.items():
                map_bpe[key] = map_bpe.get(key, 0) + count

        max_count = 0
        max_key = None
        for key in map_bpe:
            if map_bpe[key] > max_count:
                max_count = map_bpe[key]
                max_key = key
        rules.append([max_key[0], max_key[1], max_key[2], max_key[3], max_key[4]])

        chunk_args = [(chunk, max_key, NEW_TOKEN_ID) for chunk in chunks]
        changed_counts = pool.map(replace_pass_chunk_worker, chunk_args)
        total_saved = sum(changed_counts)

        NEW_TOKEN_ID += 1
        assert(total_saved > 0)
        CODEWORDS_ADDED += 1

        if CODEWORDS_ADDED % CHECKPOINT_EVERY_N_STEPS == 0:
            save_checkpoint()
            print(f"    [checkpoint saved at step {CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}]")

        time_so_far = time.time() - time_start
        time_so_far = time_so_far / CODEWORDS_ADDED * (CODEWORDS_TO_ADD - CODEWORDS_ADDED)
        print(f"\n---> DONE ADDING CODEWORD {CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}, Time left (est.): {time_so_far/60:.1f} mins\n")

    pool.close()
    pool.join()

    save_checkpoint()
    torch.save(rules, os.path.join(SAVE_PATH, "rules.dat"))
    print("✅ rules.dat saved!")

    # ── Step 4: Export token shapes & sequences (serial, one-time, cheap) ─────
    unique_train = UniqueDataset()
    os.makedirs(os.path.join(SAVE_PATH, 'tokenshapes'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, 'tokensequences'), exist_ok=True)
    for k in range(FILE_MODULO):
        os.makedirs(os.path.join(SAVE_PATH, f"tokensequences/{k}"), exist_ok=True)

    exported = {}
    token_sizes = {}
    for unique, classes, path in unique_train:
        path_v2 = path.replace("transcribed_data", "transcribed_data_og")
        tokens_original = torch.load(path_v2, weights_only=False)
        total_size = 0
        uniques_exported = {}
        for x in range(unique.size()[0]):
            for y in range(unique.size()[1]):
                for z in range(unique.size()[2]):
                    class_to_export = classes[x][y][z].item()
                    if class_to_export not in exported:
                        uid_to_export = unique[x][y][z].item()
                        most_x = DEPTH + 1; most_y = WIDTH + 1; most_z = HEIGHT + 1
                        indices = torch.nonzero(unique == uid_to_export)
                        for i in range(indices.size()[0]):
                            most_x = min(most_x, indices[i][0])
                            most_y = min(most_y, indices[i][1])
                            most_z = min(most_z, indices[i][2])
                        offset_x = most_x - x
                        crop_unique = unique.clone()[most_x:, most_y:, most_z:]
                        crop_classes = classes.clone()[most_x:, most_y:, most_z:]
                        crop_original = tokens_original.clone()[most_x:, most_y:, most_z:]
                        wrong = torch.nonzero(crop_unique != uid_to_export)
                        for i in range(wrong.size()[0]):
                            crop_classes[wrong[i][0]][wrong[i][1]][wrong[i][2]] = -1
                            crop_original[wrong[i][0]][wrong[i][1]][wrong[i][2]] = -1
                        while (crop_classes[-1] != -1).sum() == 0:
                            crop_classes = crop_classes[:-1]; crop_original = crop_original[:-1]
                        while (crop_classes[:, -1] != -1).sum() == 0:
                            crop_classes = crop_classes[:, :-1]; crop_original = crop_original[:, :-1]
                        while crop_classes.size()[2] > 0 and (crop_classes[:, :, -1] != -1).sum() == 0:
                            crop_classes = crop_classes[:, :, :-1]; crop_original = crop_original[:, :, :-1]
                        token_size = (crop_original != -1).sum().item()
                        token_sizes[classes[x, y, z].item()] = token_size
                        total_size += token_size
                        uniques_exported[uid_to_export] = True
                        torch.save([crop_original, offset_x],
                                   os.path.join(SAVE_PATH, "tokenshapes", str(class_to_export) + ".dat"))
                        exported[class_to_export] = True
                    elif not unique[x][y][z].item() in uniques_exported:
                        total_size += token_sizes[classes[x, y, z].item()]
                        uniques_exported[unique[x][y][z].item()] = True
        assert total_size == DEPTH * WIDTH * HEIGHT, f"{total_size} != {DEPTH*WIDTH*HEIGHT}"
    print("✅ tokenshapes exported!")

    index = 0
    sequence_lengths = []
    for unique, classes, path in unique_train:
        file_idx = int(os.path.basename(path).replace(".dat", ""))   # ORDER FIX: index of the source volume
        sequence = []
        used_ids = {}
        for x in range(unique.size()[0]):
            for y in range(unique.size()[1]):
                for z in range(unique.size()[2]):
                    if unique[x][y][z].item() in used_ids:
                        continue
                    used_ids[unique[x][y][z].item()] = True
                    sequence.append([classes[x][y][z].item(), x, y, z])
        if index == 0:
            visualise_unique_3d(unique, save_name=f"final_unique_{index}.png")
        sequence_lengths.append(len(sequence))
        if index % 100 == 0:
            avg = sum(sequence_lengths) / len(sequence_lengths)
            total = DEPTH * WIDTH * HEIGHT
            print(f"DONE EXPORTING {index}/{len(unique_train)} SEQUENCES, AVG LENGTH: {avg:.1f}, "
                  f"{avg/(total)*100:.1f}% of initial size")
        torch.save(sequence, os.path.join(SAVE_PATH, "tokensequences", str(file_idx % FILE_MODULO), str(file_idx) + ".dat"))
        index += 1

    print("✅ tokensequences exported!")
    print(f"✅ ALL DONE! Files saved to {SAVE_PATH}")
    print(f"Average sequence length: {sum(sequence_lengths)/len(sequence_lengths):.1f} vs original {DEPTH*WIDTH*HEIGHT}")

    print("\n── Dataset Pruning ───────────────")
    all_lengths = []; all_files = []
    for i in range(FILE_MODULO):
        folder = os.path.join(SAVE_PATH, "tokensequences", str(i))
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, f)
            seq = torch.load(fpath, weights_only=False)
            all_lengths.append(len(seq)); all_files.append(fpath)

    print(f"Total sequences: {len(all_files)}")
    print(f"Max length before: {max(all_lengths)}")
    print(f"Avg length before: {sum(all_lengths)/len(all_lengths):.1f}")
    threshold = np.percentile(all_lengths, 95)
    print(f"95th percentile: {threshold:.0f}")
    removed = 0
    for fpath, length in zip(all_files, all_lengths):
        if length > threshold:
            os.remove(fpath); removed += 1
    print(f"Removed {removed} sequences ({removed/len(all_files)*100:.1f}%)")
    print(f"Remaining: {len(all_files) - removed} sequences")
    remaining_lengths = [l for l in all_lengths if l <= threshold]
    if remaining_lengths:
        print(f"Max length after: {max(remaining_lengths)}")
        print(f"Avg length after: {sum(remaining_lengths)/len(remaining_lengths):.1f}")
    print("✅ Dataset pruning done!")

    print("\n── Step 5: Test Tokenization ─────")
    test_trans_path = os.path.join(SAVE_PATH, "transcribed_test")
    test_seq_path = os.path.join(SAVE_PATH, "tokensequences_test")
    if os.path.exists(test_trans_path):
        shutil.rmtree(test_trans_path)
    if os.path.exists(test_seq_path):
        shutil.rmtree(test_seq_path)
    for i in range(FILE_MODULO):
        os.makedirs(os.path.join(test_trans_path, str(i)), exist_ok=True)
        os.makedirs(os.path.join(test_seq_path, str(i)), exist_ok=True)

    print("Transcribing test images...")
    its_test = 0
    unique_tst = None
    for img, label in test_loader:
        img = img.float() * 255.0
        img = img.int()
        if unique_tst is None:
            unique_tst = torch.zeros_like(torch.zeros(TARGET, TARGET, TARGET)).long()
            for x in range(TARGET):
                for y in range(TARGET):
                    for z in range(TARGET):
                        unique_tst[x][y][z] = x*TARGET*TARGET + y*TARGET + z
        for k in range(img.size()[0]):
            vol = downsample_volume(img[k,0].clone())
            torch.save([unique_tst.clone(), vol],
                       os.path.join(test_trans_path, str(its_test % FILE_MODULO), str(its_test) + ".dat"))
            its_test += 1
    print(f"Transcribed {its_test} test images")

    test_files = []
    for i in range(FILE_MODULO):
        folder = os.path.join(test_trans_path, str(i))
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            test_files.append(os.path.join(folder, f))
    print(f"Found {len(test_files)} test files")
    print(f"Applying {len(rules)} rules...")

    current_token = START_OF_NEW_TOKENS
    for rule_idx, rule in enumerate(rules):
        class_a, class_b, vx, vy, vz = rule
        for path in test_files:
            unique_t, classes_t = torch.load(path, weights_only=False)
            changed = False
            used_uniques = {}
            for x in range(unique_t.size()[0]):
                for y in range(unique_t.size()[1]):
                    for z in range(unique_t.size()[2]):
                        if unique_t[x][y][z].item() in used_uniques:
                            continue
                        for dx, dy, dz in NEIGHBORS_3D:
                            n_x, n_y, n_z = x+dx, y+dy, z+dz
                            if n_x >= unique_t.size()[0] or n_y >= unique_t.size()[1] or n_z >= unique_t.size()[2]:
                                continue
                            if unique_t[n_x][n_y][n_z].item() in used_uniques:
                                continue
                            if unique_t[x][y][z] == unique_t[n_x][n_y][n_z]:
                                continue
                            ca = classes_t[x][y][z].item()
                            cb = classes_t[n_x][n_y][n_z].item()
                            if class_a != ca or class_b != cb:
                                continue
                            anchor_a = get_anchor(unique_t, x, y, z)
                            anchor_b = get_anchor(unique_t, n_x, n_y, n_z)
                            v = anchor_a - anchor_b
                            if vx == v[0].item() and vy == v[1].item() and vz == v[2].item():
                                a_u = unique_t[x][y][z]
                                idx_a = torch.nonzero(unique_t == unique_t[x, y, z])
                                idx_b = torch.nonzero(unique_t == unique_t[n_x, n_y, n_z])
                                for i in range(idx_b.size()[0]):
                                    unique_t[idx_b[i][0]][idx_b[i][1]][idx_b[i][2]] = a_u
                                    classes_t[idx_b[i][0]][idx_b[i][1]][idx_b[i][2]] = current_token
                                for i in range(idx_a.size()[0]):
                                    classes_t[idx_a[i][0]][idx_a[i][1]][idx_a[i][2]] = current_token
                                changed = True
                                used_uniques[unique_t[x][y][z].item()] = True
            if changed:
                torch.save([unique_t, classes_t], path)
        current_token += 1
        if (rule_idx + 1) % 50 == 0:
            print(f"  Applied {rule_idx+1}/{len(rules)} rules")

    print("✅ Rules applied!")
    print("Exporting test sequences...")
    index_test = 0
    test_seq_lengths = []
    for path in test_files:
        file_idx = int(os.path.basename(path).replace(".dat", ""))   # ORDER FIX: index of the source volume
        unique_t, classes_t = torch.load(path, weights_only=False)
        sequence = []
        used_ids = {}
        for x in range(unique_t.size()[0]):
            for y in range(unique_t.size()[1]):
                for z in range(unique_t.size()[2]):
                    uid = unique_t[x][y][z].item()
                    if uid in used_ids:
                        continue
                    used_ids[uid] = True
                    sequence.append([classes_t[x][y][z].item(), x, y, z])
        test_seq_lengths.append(len(sequence))
        torch.save(sequence, os.path.join(test_seq_path, str(file_idx % FILE_MODULO), str(file_idx) + ".dat"))
        index_test += 1

    avg_test = sum(test_seq_lengths) / len(test_seq_lengths)
    print(f"✅ Test sequences exported!")
    print(f"   {index_test} sequences")
    print(f"   Avg length: {avg_test:.1f}")

    metadata = {
        "NEW_TOKEN_ID": NEW_TOKEN_ID, "START_OF_NEW_TOKENS": START_OF_NEW_TOKENS,
        "MAXIMUM_NUMBER_OF_TOKENS": MAXIMUM_NUMBER_OF_TOKENS,
        "WIDTH": WIDTH, "HEIGHT": HEIGHT, "DEPTH": DEPTH,
        "LIMITED_EXAMPLES": LIMITED_EXAMPLES, "FILE_MODULO": FILE_MODULO,
        "total_images": its, "total_test_images": its_test,
        "mdbpe_type": "3D", "dataset": "organmnist3d_512"
    }
    torch.save(metadata, os.path.join(SAVE_PATH, "metadata.pt"))
    print("✅ metadata.pt saved!")

    # ORDER FIX: labels per file index (file k = dataset volume k), in the format
    # the classifier and analysis scripts already read; taken from the dataset
    # so this also works after resuming from a checkpoint
    def dataset_labels(ds, n):
        labs = getattr(ds, "labels", None)
        get = (lambda i: labs[i]) if labs is not None else (lambda i: ds[i][1])
        return {i: int(np.asarray(get(i)).reshape(-1)[0]) for i in range(n)}
    for split, labs in (("train", dataset_labels(train_dataset, its)),
                        ("test",  dataset_labels(test_dataset, its_test))):
        torch.save(labs, os.path.join(SAVE_PATH, f"recovered_labels_{split}.pt"))
        torch.save({k: k for k in labs}, os.path.join(SAVE_PATH, f"recovered_index_{split}.pt"))
    print("✅ labels saved (recovered_labels_*.pt, recovered_index_*.pt; file k = volume k)")
    print(f"\n✅ FULLY DONE!")
    print(f"   Train: {its} sequences ✅")
    print(f"   Test:  {its_test} sequences ✅")
    print(f"   Rules: {len(rules)} ✅")


if __name__ == "__main__":
    main()