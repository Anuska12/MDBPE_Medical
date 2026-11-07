# -*- coding: utf-8 -*-
# Extended from 2D BreastMNIST MDBPE → 4D ACDC
# Key changes from 2D version:
# 1. nibabel loader replaces MedMNIST DataLoader
# 2. WIDTH, HEIGHT, DEPTH, TIME replace WIDTH, HEIGHT
# 3. unique tensor is 4D: (H, W, D, T)
# 4. Neighbours: [(x+1,y,z,t),(x,y+1,z,t),(x,y,z+1,t),(x,y,z,t+1)]
# 5. Keys have 6 elements: (class_a, class_b, vx, vy, vz, vt)
# 7. Variable-length sequences → padding in DataLoader
# 8. Sequence entries: [class, x, y, z, t]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import glob
import nibabel as nib

# ── Setup ──────────────────────────────────────────────────────────────────────
MAXIMUM_NUMBER_OF_TOKENS = 256 + 1024
LIMITED_EXAMPLES         = None   # ← set None for full dataset
                                 #   use 10 first to verify pipeline works

SAVE_PATH       = os.path.expanduser("/your/path")
CHECKPOINT_PATH = os.path.join(SAVE_PATH, "checkpoint.pt")
ACDC_ROOT_TRAIN      = "/your/path/acdc/training"  # ← change to your path
ACDC_ROOT_TEST = "/your/path/acdc/testing"
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ── ACDC loader ────────────────────────────────────────────────────────────────
def load_acdc_4d(patient_dir,
    target_H=32, target_W=27,
    target_D=2,  target_T=8):
    """
    Load 4D cardiac MRI for one patient.
    Returns (H, W, D, T) tensor 0-255 int.
    No labels ✅
    """
    files_4d = glob.glob(os.path.join(
        patient_dir, "*_4d.nii.gz"))
    if len(files_4d) == 0:
        return None
 
    img  = nib.load(files_4d[0])
    data = img.get_fdata()
 
    mn, mx = data.min(), data.max()
    if mx - mn < 1e-8:
        return None
    data = (data - mn) / (mx - mn)
 
    H, W, D, T = data.shape
 
    frames = []
    for t in range(T):
        frame = torch.tensor(
            data[:,:,:,t]).float()
        frame = frame.unsqueeze(0)\
                     .unsqueeze(0)
        frame = F.interpolate(
            frame,
            size=(target_H,
                  target_W,
                  target_D),
            mode='trilinear',
            align_corners=False)
        frames.append(frame.squeeze())
 
    vol  = torch.stack(frames, dim=-1)
    step = max(1, T // target_T)
    vol  = vol[:,:,:,::step]
    vol  = vol[:,:,:,:target_T]
    vol  = (vol * 255).long()
    return vol


# ── Step 0: Verify ACDC data ───────────────────────────────────────────────────
patient_dirs = sorted(glob.glob(os.path.join(ACDC_ROOT_TRAIN, "patient*")))
print(f"Found {len(patient_dirs)} patients")
assert len(patient_dirs) > 0, f"No patients found at {ACDC_ROOT_TRAIN}"

# Print shapes of first 3 patients to verify
for pd in patient_dirs[:3]:
    vol = load_acdc_4d(pd)
    if vol is not None:
        print(f"  {os.path.basename(pd)}: "
              f"shape={vol.shape} ✅")
print("✅ ACDC data verified!")

# ── Setup folders ──────────────────────────────────────────────────────────────
FILE_MODULO = 128

is_fresh_start = not os.path.exists(CHECKPOINT_PATH)

if is_fresh_start:
    print("Fresh start — transcribing ACDC 4D data...")

    for folder in ["transcribed_data", "transcribed_data_og",
                   "tokensequences", "tokenshapes",
                   "transcribed_data_test"]:
        path = os.path.join(SAVE_PATH, folder)
        if os.path.exists(path):
            shutil.rmtree(path)

    for i in range(FILE_MODULO):
        os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data',     str(i)), exist_ok=True)
        os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data_og',  str(i)), exist_ok=True)
        os.makedirs(os.path.join(SAVE_PATH, 'transcribed_data_test', str(i)), exist_ok=True)

# ── Step 1: Transcribe ACDC ────────────────────────────────────────────────────
# Replaces: for img, label in train_loader: ...
# Now:      for each patient folder, load 4D NIfTI

if is_fresh_start:
    # split: 100 train, 50 test (standard ACDC split)
    train_dirs = patient_dirs[:100]
    test_dirs   = sorted(glob.glob(os.path.join(ACDC_ROOT_TEST, "patient*")))

    # ── Transcribe TRAIN ───────────────────────────────────────────────────────
    its = 0
    for patient_dir in train_dirs:
        vol = load_acdc_4d(patient_dir)   # (H, W, D, T)
        if vol is None:
            print(f"  Skipping {patient_dir} (no 4D file)")
            continue

        H, W, D, T = vol.shape

        # ── 4D unique ID tensor ────────────────────────────────────────────────
        # Replaces: unique = torch.zeros(W, H).long()
        # Each voxel gets a flat unique ID across all 4 dimensions
        unique = torch.zeros(H, W, D, T).long()
        for x in range(H):
            for y in range(W):
                for z in range(D):
                    for t in range(T):
                        unique[x][y][z][t] = x*W*D*T + y*D*T + z*T + t

        # save [unique, vol, label] — label added for disease-guided merging
        torch.save([unique.clone(), vol.clone()],
                   os.path.join(SAVE_PATH, "transcribed_data",
                                str(its % FILE_MODULO), str(its) + ".dat"))
        # save original vol separately (needed for token shape export)
        torch.save(vol.clone(),
                   os.path.join(SAVE_PATH, "transcribed_data_og",
                                str(its % FILE_MODULO), str(its) + ".dat"))
        its += 1
        print(f"  Transcribed {its}/{len(train_dirs)}: "
              f"{os.path.basename(patient_dir)} shape={vol.shape}")

        if LIMITED_EXAMPLES != None and its >= LIMITED_EXAMPLES:
            break

    print(f"✅ Transcribed {its} 4D volumes (train)")

    # ── Transcribe TEST ─────────────────────────────────────────────────────────
    its_test = 0
    for patient_dir in test_dirs:
        vol = load_acdc_4d(patient_dir)
        if vol is None:
            continue
        H, W, D, T = vol.shape
        unique = torch.zeros(H, W, D, T).long()
        for x in range(H):
            for y in range(W):
                for z in range(D):
                    for t in range(T):
                        unique[x][y][z][t] = x*W*D*T + y*D*T + z*T + t

        torch.save([unique.clone(), vol.clone()],
                   os.path.join(SAVE_PATH, "transcribed_data_test",
                                str(its_test % FILE_MODULO), str(its_test) + ".dat"))
        its_test += 1
        if LIMITED_EXAMPLES != None and its_test >= (LIMITED_EXAMPLES // 5):
            break

    print(f"✅ Transcribed {its_test} 4D volumes (test)")

    # Save metadata
    torch.save({
        "MAXIMUM_NUMBER_OF_TOKENS": MAXIMUM_NUMBER_OF_TOKENS,
        "FILE_MODULO":              FILE_MODULO,
        "LIMITED_EXAMPLES":        LIMITED_EXAMPLES,
        "total_train":             its,
        "total_test":              its_test,
        "mdbpe_type":              "4D",
        "dataset":                 "acdc"
    }, os.path.join(SAVE_PATH, "metadata.pt"))
    print("✅ metadata.pt saved!")

else:
    print("✅ Checkpoint found — skipping transcription!")

# ── Step 2: BPE Dataset ────────────────────────────────────────────────────────
# Same structure as 2D version but now loads [unique, classes, label]
class UniqueDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        self.files = []
        folder_name = 'transcribed_data' if split == 'train' \
                      else 'transcribed_data_test'
        for i in range(FILE_MODULO):
            folder = os.path.join(SAVE_PATH, folder_name, str(i))
            if not os.path.exists(folder):
                continue
            for root, dirs, files in os.walk(folder):
                for file in files:
                    self.files.append((folder_name, str(i) + "/" + file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        folder_name, rel_path = self.files[idx]
        path = os.path.join(SAVE_PATH, folder_name, rel_path)
        unique, classes = torch.load(path, weights_only=False)
        return unique, classes, path

unique_train = UniqueDataset(split='train')
print(f"UniqueDataset size: {len(unique_train)}")

# ── 4D Visualisation (3 orthogonal slices + 1 temporal slice) ─────────────────
# Replaces: visualise_unique(input, save_name)
for a, _, _ in unique_train:
    H, W, D, T = a.shape
    display_colours = torch.rand(3, H * W * D * T)
    break

def visualise_unique_4d(input_4d, save_name=None):
    """Show axial, coronal, sagittal, temporal centre slices"""
    global display_colours
    assert len(input_4d.size()) == 4, f"Expected 4D, got {input_4d.shape}"
    H, W, D, T = input_4d.shape
    mH, mW, mD, mT = H//2, W//2, D//2, T//2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    def sl2col(sl):
        return display_colours[:, sl.reshape(-1)].reshape(3, sl.shape[0], sl.shape[1])

    # axial: (W, T) at mid H, mid D
    axes[0].imshow(sl2col(input_4d[mH, :, mD, :]).permute(1,2,0).numpy())
    axes[0].set_title(f'Axial H={mH},D={mD}')
    axes[0].axis('off')

    # coronal: (H, T) at mid W, mid D
    axes[1].imshow(sl2col(input_4d[:, mW, mD, :]).permute(1,2,0).numpy())
    axes[1].set_title(f'Coronal W={mW},D={mD}')
    axes[1].axis('off')

    # sagittal: (H, W) at mid D, mid T
    axes[2].imshow(sl2col(input_4d[:, :, mD, mT]).permute(1,2,0).numpy())
    axes[2].set_title(f'Sagittal D={mD},T={mT}')
    axes[2].axis('off')

    # temporal: (H, W) at mid D, last T
    axes[3].imshow(sl2col(input_4d[:, :, mD, -1]).permute(1,2,0).numpy())
    axes[3].set_title(f'Last phase T={T-1}')
    axes[3].axis('off')

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(SAVE_PATH, save_name), dpi=80)
    plt.close()

# ── Find starting token ID ─────────────────────────────────────────────────────
NEW_TOKEN_ID = -1
for unique, classes, path in unique_train:
    NEW_TOKEN_ID = max(NEW_TOKEN_ID, torch.max(classes).item())
NEW_TOKEN_ID += 1

print(f"NEW TOKEN STARTS AT {NEW_TOKEN_ID}")
START_OF_NEW_TOKENS = NEW_TOKEN_ID
CODEWORDS_TO_ADD    = MAXIMUM_NUMBER_OF_TOKENS - START_OF_NEW_TOKENS

# ── 4D Anchor function ─────────────────────────────────────────────────────────
# Replaces: get_anchor(unique, pos_x, pos_y)
def get_anchor(unique, pos_x, pos_y, pos_z, pos_t):
    """Find top-left-front-first corner of the token at (x,y,z,t)"""
    uid      = unique[pos_x][pos_y][pos_z][pos_t]
    top_left = torch.nonzero(unique == uid)[0]  # returns [x, y, z, t]
    return top_left

# ── Helper: get test files ──────────────────────────────────────────────────────
def get_test_files():
    test_files = []
    for i in range(FILE_MODULO):
        folder = os.path.join(SAVE_PATH, "transcribed_data_test", str(i))
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    test_files.append(os.path.join(folder, file))
    return test_files

# ── Resume from checkpoint ─────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    print("Loading MDBPE checkpoint...")
    ckpt            = torch.load(CHECKPOINT_PATH, weights_only=False)
    NEW_TOKEN_ID    = ckpt['NEW_TOKEN_ID']
    CODEWORDS_ADDED = ckpt['CODEWORDS_ADDED']
    rules           = ckpt['rules']
    time_start      = time.time() - ckpt.get('elapsed', 0)
    print(f"✅ Resumed from codeword {CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}")
else:
    print("Starting fresh MDBPE...")
    CODEWORDS_ADDED = 0
    time_start      = time.time()
    rules           = []

last_output = time.time()

# ── Step 3: 4D MDBPE loop ──────────────────────────────────────────────────────
# Replaces the 2D while loop with 4D neighbours and disease-guided merging

while NEW_TOKEN_ID < MAXIMUM_NUMBER_OF_TOKENS:
    unique_train = UniqueDataset(split='train')

    # ── Pass 1: Count pairs (4D neighbours) ────────────────────────────────────
    # Replaces: for n in [(x+1,y),(x,y+1)]
    # Now:      for n in [(x+1,y,z,t),(x,y+1,z,t),(x,y,z+1,t),(x,y,z,t+1)]
    map_bpe = None
    map_bpe = {}

    it = 0
    for unique, classes, _ in unique_train:
        if time.time() - last_output > 60:
            print(f"\t{it/len(unique_train)*100:.1f}% counting pass")
            last_output = time.time()

        if it == 0:
            visualise_unique_4d(unique,
                save_name=f"merge_step_{CODEWORDS_ADDED}.png")

        it  += 1
        used = {}

        H, W, D, T = unique.shape

        for x in range(H):
            for y in range(W):
                for z in range(D):
                    for t in range(T):
                        # ← 4 neighbours: +x, +y, +z, +t
                        for n in [(x+1,y,z,t),
                                  (x,y+1,z,t),
                                  (x,y,z+1,t),
                                  (x,y,z,t+1)]:   # ← temporal neighbour NEW
                            n_x,n_y,n_z,n_t = n
                            if (n_x >= H or n_y >= W or
                                n_z >= D or n_t >= T):
                                continue
                            if unique[x][y][z][t] == unique[n_x][n_y][n_z][n_t]:
                                continue

                            class_a  = classes[x][y][z][t].item()
                            class_b  = classes[n_x][n_y][n_z][n_t].item()
                            anchor_a = get_anchor(unique, x, y, z, t)
                            anchor_b = get_anchor(unique, n_x, n_y, n_z, n_t)

                            anchor_a_key = tuple(anchor_a.tolist())
                            anchor_b_key = tuple(anchor_b.tolist())

                            if (anchor_a_key, anchor_b_key) in used:
                                continue
                            used[(anchor_a_key, anchor_b_key)] = True

                            v  = anchor_a - anchor_b
                            # ← key has 6 elements now (was 4 in 2D)
                            key = (class_a, class_b,
                                   v[0].item(), v[1].item(),
                                   v[2].item(), v[3].item())

                            if not key in map_bpe:
                              # add if not in map
                                map_bpe[key] = 1
                            else:
                              # just increment
                                map_bpe[key] += 1
    
    # Pass 2: find most frequent
    max = 0
    max_key = None
    for key in map_bpe:
        if map_bpe[key] > max:
            max = map_bpe[key]
            max_key = key
    rules.append([max_key[0], max_key[1], max_key[2], max_key[3], max_key[4], max_key[5]])

    # ── Pass 3: Replace in ALL train volumes ───────────────────────────────────
    # Same logic as 2D but extended to 4D loops
    it          = 0
    total_saved = 0

    for unique, classes, path in unique_train:
        if time.time() - last_output > 60:
            print(f"\t{it/len(unique_train)*100:.1f}% replacement pass")
            last_output = time.time()

        changed      = False
        used_uniques = {}
        H, W, D, T   = unique.shape

        for x in range(H):
            for y in range(W):
                for z in range(D):
                    for t in range(T):
                        if unique[x][y][z][t].item() in used_uniques:
                            continue
                        for n in [(x+1,y,z,t),
                                  (x,y+1,z,t),
                                  (x,y,z+1,t),
                                  (x,y,z,t+1)]:
                            n_x,n_y,n_z,n_t = n
                            if (n_x >= H or n_y >= W or
                                n_z >= D or n_t >= T):
                                continue
                            if unique[n_x][n_y][n_z][n_t].item() in used_uniques:
                                continue
                            if unique[x][y][z][t] == unique[n_x][n_y][n_z][n_t]:
                                continue

                            class_a = classes[x][y][z][t].item()
                            class_b = classes[n_x][n_y][n_z][n_t].item()
                            if max_key[0] != class_a or max_key[1] != class_b:
                                continue

                            anchor_a = get_anchor(unique, x, y, z, t)
                            anchor_b = get_anchor(unique, n_x, n_y, n_z, n_t)
                            v        = anchor_a - anchor_b
                            vx,vy,vz,vt = (v[0].item(), v[1].item(),
                                           v[2].item(), v[3].item())

                            if (max_key[2]==vx and max_key[3]==vy and
                                max_key[4]==vz and max_key[5]==vt):

                                a_unique = unique[x][y][z][t]
                                indices_to_relabel_a    = torch.nonzero(
                                    unique == unique[x,y,z,t])
                                indices_to_relabel_b    = torch.nonzero(
                                    unique == unique[n_x,n_y,n_z,n_t])

                                for i in range(indices_to_relabel_b.size()[0]):
                                    ix,iy,iz,it_ = (indices_to_relabel_b[i][0], indices_to_relabel_b[i][1],
                                                    indices_to_relabel_b[i][2], indices_to_relabel_b[i][3])
                                    unique[ix][iy][iz][it_]  = a_unique
                                    classes[ix][iy][iz][it_] = NEW_TOKEN_ID
                                for i in range(indices_to_relabel_a.size()[0]):
                                    ix,iy,iz,it_ = (indices_to_relabel_a[i][0], indices_to_relabel_a[i][1],
                                                    indices_to_relabel_a[i][2], indices_to_relabel_a[i][3])
                                    classes[ix][iy][iz][it_] = NEW_TOKEN_ID

                                changed = True
                                used_uniques[unique[x][y][z][t].item()] = True

        if changed:
            total_saved += 1
            torch.save([unique, classes], path)
        it += 1

    # ── Apply same merge to TEST ────────────────────────────────────────────────
    test_files = get_test_files()
    for path in test_files:
        unique_t, classes_t = torch.load(path, weights_only=False)
        changed      = False
        used_uniques = {}
        H, W, D, T   = unique_t.shape

        for x in range(H):
            for y in range(W):
                for z in range(D):
                    for t in range(T):
                        if unique_t[x][y][z][t].item() in used_uniques:
                            continue
                        for n in [(x+1,y,z,t),(x,y+1,z,t),
                                  (x,y,z+1,t),(x,y,z,t+1)]:
                            n_x,n_y,n_z,n_t = n
                            if (n_x >= H or n_y >= W or
                                n_z >= D or n_t >= T):
                                continue
                            if unique_t[n_x][n_y][n_z][n_t].item() in used_uniques:
                                continue
                            if unique_t[x][y][z][t] == unique_t[n_x][n_y][n_z][n_t]:
                                continue
                            class_a = classes_t[x][y][z][t].item()
                            class_b = classes_t[n_x][n_y][n_z][n_t].item()
                            if max_key[0] != class_a or max_key[1] != class_b:
                                continue
                            anchor_a = get_anchor(unique_t, x, y, z, t)
                            anchor_b = get_anchor(unique_t, n_x, n_y, n_z, n_t)
                            v        = anchor_a - anchor_b
                            vx,vy,vz,vt = (v[0].item(), v[1].item(),
                                           v[2].item(), v[3].item())
                            if (max_key[2]==vx and max_key[3]==vy and
                                max_key[4]==vz and max_key[5]==vt):
                                a_unique = unique_t[x][y][z][t]
                                indices_to_relabel_a    = torch.nonzero(
                                    unique_t == unique_t[x,y,z,t])
                                indices_to_relabel_b    = torch.nonzero(
                                    unique_t == unique_t[n_x,n_y,n_z,n_t])
                                for i in range(indices_to_relabel_b.size()[0]):
                                    ix,iy,iz,it_ = (indices_to_relabel_b[i][0],indices_to_relabel_b[i][1],
                                                    indices_to_relabel_b[i][2],indices_to_relabel_b[i][3])
                                    unique_t[ix][iy][iz][it_]  = a_unique
                                    classes_t[ix][iy][iz][it_] = NEW_TOKEN_ID
                                for i in range(indices_to_relabel_a.size()[0]):
                                    ix,iy,iz,it_ = (indices_to_relabel_a[i][0],indices_to_relabel_a[i][1],
                                                    indices_to_relabel_a[i][2],indices_to_relabel_a[i][3])
                                    classes_t[ix][iy][iz][it_] = NEW_TOKEN_ID
                                changed = True
                                used_uniques[unique_t[x][y][z][t].item()] = True

        if changed:
            torch.save([unique_t, classes_t], path)

    NEW_TOKEN_ID    += 1
    assert total_saved > 0, "No files updated — check merge logic"
    CODEWORDS_ADDED += 1

    # ── Save checkpoint ────────────────────────────────────────────────────────
    torch.save({
        'NEW_TOKEN_ID':    NEW_TOKEN_ID,
        'CODEWORDS_ADDED': CODEWORDS_ADDED,
        'rules':           rules,
        'elapsed':         time.time() - time_start,
    }, CHECKPOINT_PATH)

    time_so_far = time.time() - time_start
    eta = time_so_far / CODEWORDS_ADDED * (CODEWORDS_TO_ADD - CODEWORDS_ADDED)
    print(f"\n---> DONE CODEWORD {CODEWORDS_ADDED}/{CODEWORDS_TO_ADD}, "
          f"ETA: {eta/60:.1f} mins\n")

# Save rules
torch.save(rules, os.path.join(SAVE_PATH, "rules.dat"))
print("✅ rules.dat saved!")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("✅ Checkpoint deleted (MDBPE complete!)")

# ── Step 4: Export token shapes & sequences ────────────────────────────────────
# Replaces 2D export with 4D crops
unique_train = UniqueDataset(split='train')

os.makedirs(os.path.join(SAVE_PATH, 'tokenshapes'),        exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'tokensequences'),     exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'tokensequences_test'), exist_ok=True)
for k in range(FILE_MODULO):
    os.makedirs(os.path.join(SAVE_PATH, f"tokensequences/{k}"),     exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, f"tokensequences_test/{k}"), exist_ok=True)

# ── Export 4D token shapes ─────────────────────────────────────────────────────
exported     = {}
token_sizes  = {}

for unique, classes, path in unique_train:
    path_v2      = path.replace("transcribed_data", "transcribed_data_og")
    tokens_original    = torch.load(path_v2, weights_only=False)
    total_size   = 0
    uniques_exported    = {}
    H, W, D, T   = unique.shape

    for x in range(H):
        for y in range(W):
            for z in range(D):
                for t in range(T):
                    class_to_export = classes[x][y][z][t].item()

                    if class_to_export not in exported:
                        # find bounding box corner
                        uid_to_export = unique[x][y][z][t].item()
                        most_left  = H + 1   # was WIDTH + 1
                        most_top   = W + 1   # was HEIGHT + 1
                        most_front = D + 1   # NEW ← z dim
                        most_first = T + 1   # NEW ← t dim
                        indices  = torch.nonzero(unique == uid_to_export)
                        for i in range(indices.size()[0]):
                            most_left  = min(most_left, indices[i][0].item())
                            most_top   = min(most_top, indices[i][1].item())
                            most_front = min(most_front, indices[i][2].item())  # NEW
                            most_first = min(most_first, indices[i][3].item())  # 
                        offset_x = most_left  - x   # H offset
                        offset_y = most_top   - y   # W offset
                        offset_z = most_front - z   # D offset ← NEW
                        offset_t = most_first - t   # T offset ← NEW
                        
                        # 4D crop ← now consistent ✅
                        crop_unique   = unique.clone()[most_left:, most_top:, most_front:, most_first:]
                        crop_classes  = classes.clone()[most_left:, most_top:, most_front:, most_first:]
                        crop_original = tokens_original.clone()[most_left:, most_top:,most_front:, most_first:]

                        # mask out non-belonging voxels
                        wrong = torch.nonzero(crop_unique != uid_to_export)
                        for i in range(wrong.size()[0]):
                            wx,wy,wz,wt = (wrong[i][0],wrong[i][1],
                                           wrong[i][2],wrong[i][3])
                            crop_classes[wx][wy][wz][wt]  = -1
                            crop_original[wx][wy][wz][wt] = -1

                        # trim empty borders in all 4 dimensions
                        while crop_classes.size()[0]>0 and (crop_classes[-1]!=-1).sum()==0:
                            crop_classes  = crop_classes[:-1]
                            crop_original = crop_original[:-1]
                        while crop_classes.size()[1]>0 and (crop_classes[:,-1]!=-1).sum()==0:
                            crop_classes  = crop_classes[:,:-1]
                            crop_original = crop_original[:,:-1]
                        while crop_classes.size()[2]>0 and (crop_classes[:,:,-1]!=-1).sum()==0:
                            crop_classes  = crop_classes[:,:,:-1]
                            crop_original = crop_original[:,:,:-1]
                        while crop_classes.size()[3]>0 and (crop_classes[:,:,:,-1]!=-1).sum()==0:
                            crop_classes  = crop_classes[:,:,:,:-1]
                            crop_original = crop_original[:,:,:,:-1]

                        token_size       = (crop_original != -1).sum().item()
                        token_sizes[class_to_export] = token_size
                        total_size      += token_size
                        uniques_exported[uid_to_export]     = True

                        torch.save([crop_original, offset_x, offset_y, offset_z, offset_t],
                                   os.path.join(SAVE_PATH, "tokenshapes",
                                                str(class_to_export) + ".dat"))
                        exported[class_to_export] = True

                    elif not unique[x][y][z][t].item() in uniques_exported:
                        total_size    += token_sizes[class_to_export]
                        uniques_exported[unique[x][y][z][t].item()] = True

    # ← assertion: all voxels accounted for
    assert total_size == H * W * D * T, \
        f"total_size {total_size} != {H*W*D*T}"

print("✅ tokenshapes exported!")

# ── Export TRAIN token sequences ───────────────────────────────────────────────
# Replaces: sequence.append([classes[x][y].item(), x, y])
# Now:      sequence.append([classes[x][y][z][t].item(), x, y, z, t])
# Variable-length: different patients → different sequence lengths ✅

index            = 0
sequence_lengths = []

for unique, classes, _ in unique_train:
    sequence = []
    used_ids = {}

    H, W, D, T = unique.shape

    for x in range(H):
        for y in range(W):
            for z in range(D):
                for t in range(T):
                    uid = unique[x][y][z][t].item()
                    if uid in used_ids:
                        continue
                    used_ids[uid] = True
                    # ← 5-element entry: [class, x, y, z, t]
                    sequence.append([classes[x][y][z][t].item(),
                                     x, y, z, t])

    if index == 0:
        visualise_unique_4d(unique,
                            save_name="sequence_export_step0.png")

    sequence_lengths.append(len(sequence))

    if index % 10 == 0:
        avg = sum(sequence_lengths) / len(sequence_lengths)
        total_vox = H * W * D * T
        print(f"DONE {index}/{len(unique_train)} sequences, "
              f"AVG LENGTH: {avg:.1f}, "
              f"{avg/total_vox*100:.2f}% of original")

    torch.save(sequence,
               os.path.join(SAVE_PATH, "tokensequences",
                            str(index % FILE_MODULO), str(index) + ".dat"))
    index += 1

print("✅ tokensequences exported!")

# ── Export TEST token sequences ─────────────────────────────────────────────────
test_files  = get_test_files()
index_test  = 0
test_seq_lens = []

for path in test_files:
    unique_t, classes_t = torch.load(path, weights_only=False)
    sequence = []
    used_ids = {}
    H, W, D, T = unique_t.shape

    for x in range(H):
        for y in range(W):
            for z in range(D):
                for t in range(T):
                    uid = unique_t[x][y][z][t].item()
                    if uid in used_ids:
                        continue
                    used_ids[uid] = True
                    sequence.append([classes_t[x][y][z][t].item(),
                                     x, y, z, t])

    test_seq_lens.append(len(sequence))
    torch.save(sequence,
               os.path.join(SAVE_PATH, "tokensequences_test",
                            str(index_test % FILE_MODULO),
                            str(index_test) + ".dat"))
    index_test += 1

print(f"✅ Test sequences exported! ({index_test} sequences)")

# ── Step 5: Padding collate function for transformer training ──────────────────
# NEW — not in 2D version
# Variable-length sequences need padding for batched transformer training

PAD_TOKEN = MAXIMUM_NUMBER_OF_TOKENS  # special padding token ID

def collate_fn_4d(batch):
    """
    Pads variable-length 4D token sequences to same length in batch.
    Each sequence entry: [class_id, x, y, z, t]
    Returns:
        padded:  (batch, max_len, 5) — token + coordinates
        masks:   (batch, max_len)    — 1=real, 0=padding
        labels:  (batch,)            — disease class
    """
    sequences, labels = zip(*batch)
    max_len = max(len(s) for s in sequences)

    padded, masks = [], []
    for seq in sequences:
        pad_len = max_len - len(seq)
        # pad with [PAD_TOKEN, 0, 0, 0, 0]
        padded_seq = seq + [[PAD_TOKEN, 0, 0, 0, 0]] * pad_len
        mask       = [1] * len(seq) + [0] * pad_len
        padded.append(padded_seq)
        masks.append(mask)

    return (torch.tensor(padded),   # (B, max_len, 5)
            torch.tensor(masks),    # (B, max_len)
            torch.tensor(labels))   # (B,)

print(f"\n✅ ALL DONE! Files saved to {SAVE_PATH}")

avg_train = sum(sequence_lengths) / len(sequence_lengths)
avg_test   = sum(test_seq_lens) / len(test_seq_lens) if test_seq_lens else 0
print(f"Train avg sequence length: {avg_train:.1f}")
print(f"Test   avg sequence length: {avg_test:.1f}")
print(f"Compression vs raw voxels: shown above as % of original")
print(f"\nNext step: use tokensequences/ + collate_fn_4d")
print(f"to train your autoregressive transformer.")
print(f"PAD_TOKEN = {PAD_TOKEN}")