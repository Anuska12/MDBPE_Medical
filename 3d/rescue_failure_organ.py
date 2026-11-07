"""
Rescue / failure cases from the saved OrganMNIST3D rollout results -- no recomputation
of the overlaps.

Reads  overlap_preds_organ_seed<S>.npz  (written by organmnist3d_attention_rollout.py)
and, for the chosen seed:
  rescue  = MDBPE correct, Original wrong
  failure = MDBPE wrong,   Original correct
prints how many of each, the overlap change (MDBPE - Original) in each group and a
t-test between the groups, then draws the comparison figure for the MEDIAN rescue
and the MEDIAN failure case (median overlap change = a typical case, not a
hand-picked one), using the same plotting function as the rollout script.

Keep this file in the same folder as organmnist3d_attention_rollout.py.

usage:
  python3 -u rescue_failure_organ.py all 1024           # pool ALL seeds found (no seed choice)
  python3 -u rescue_failure_organ.py 2 1024             # a single seed
  python3 -u rescue_failure_organ.py all 1024 --no-fig  # numbers only, no model loading
"""
import os, sys
import numpy as np
from scipy.stats import ttest_ind

ARG    = sys.argv[1] if len(sys.argv) > 1 else '2'          # a seed number, or 'all'
CFG    = sys.argv[2] if len(sys.argv) > 2 else '1024'
NO_FIG = '--no-fig' in sys.argv
OUTPUT_DIR = "your/path/attention_rollout"
CLASS_NAMES = ["liver", "kidney-right", "kidney-left", "femur-right", "femur-left", "bladder",
               "heart", "lung-right", "lung-left", "spleen", "pancreas"]

if ARG == 'all':
    SEEDS = sorted(int(f[len("overlap_preds_organ_seed"):-4]) for f in os.listdir(OUTPUT_DIR)
                   if f.startswith("overlap_preds_organ_seed") and f.endswith(".npz"))
else:
    SEEDS = [int(ARG)]
print(f"Seeds used: {SEEDS}  (+{CFG} vs Original)")

rescue, failure = [], []            # entries: (seed, j, overlap change)
per_seed = []
ov_lookup = {}
for seed in SEEDS:
    z = np.load(os.path.join(OUTPUT_DIR, f"overlap_preds_organ_seed{seed}.npz"))
    def table(key):
        return (dict(zip(z[f"{key}_j"], z[f"{key}_ov"])),
                dict(zip(z[f"{key}_j"], z[f"{key}_pred"])),
                dict(zip(z[f"{key}_j"], z[f"{key}_true"])))
    ovo, po, tru = table('original')
    ovk, pk, tk  = table(CFG)
    common = sorted(set(ovo) & set(ovk))
    assert all(tru[j] == tk[j] for j in common), "true labels differ between configs -- check IDX_MAP"
    r = [j for j in common if pk[j] == tru[j] and po[j] != tru[j]]
    f = [j for j in common if pk[j] != tru[j] and po[j] == tru[j]]
    rescue  += [(seed, j, ovk[j] - ovo[j]) for j in r]
    failure += [(seed, j, ovk[j] - ovo[j]) for j in f]
    for j in common:
        ov_lookup[(seed, j)] = (tru[j], pk[j], po[j], ovk[j], ovo[j])
    dr = np.array([ovk[j] - ovo[j] for j in r]); df = np.array([ovk[j] - ovo[j] for j in f])
    p_seed = ttest_ind(dr, df).pvalue if len(dr) > 1 and len(df) > 1 else float('nan')
    per_seed.append((seed, len(common), len(r), len(f),
                     dr.mean() if len(dr) else float('nan'), df.mean() if len(df) else float('nan'), p_seed))

print(f"\n{'seed':>4} {'n':>5} {'rescue':>7} {'failure':>8} {'d_ov rescue':>12} {'d_ov failure':>13} {'p':>8}")
for seed, n, nr, nf, mr, mf, ps in per_seed:
    print(f"{seed:>4} {n:>5} {nr:>7} {nf:>8} {mr:>+12.4f} {mf:>+13.4f} {ps:>8.4g}")
if len(SEEDS) > 1:
    nr = np.array([x[2] for x in per_seed]); nf = np.array([x[3] for x in per_seed])
    print(f"per seed: rescue {nr.mean():.1f} ± {nr.std():.1f}, failure {nf.mean():.1f} ± {nf.std():.1f}")

def describe(name, grp):
    if not grp:
        print(f"  {name}: none"); return None
    d = np.array([x[2] for x in grp])
    cls = np.bincount([ov_lookup[(s, j)][0] for s, j, _ in grp], minlength=11)
    print(f"  {name}: overlap change (+{CFG} - Original) {d.mean():+.4f} ± {d.std():.4f} (n={len(grp)})")
    print(f"     true classes: " + ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in enumerate(cls) if n))
    return d

print(f"\nPooled over seeds {SEEDS}:")
dr = describe("rescue ", rescue)
df = describe("failure", failure)
if dr is not None and df is not None and len(dr) > 1 and len(df) > 1:
    t, p = ttest_ind(dr, df)
    print(f"  rescue vs failure overlap change: t={t:.4f}, p={p:.4g}"
          + ("  (pooled over seeds; the same image can appear once per seed)" if len(SEEDS) > 1 else ""))

# median (typical) case of each pooled group -> its seed follows from the case
picks = []
for kind, grp in (("rescue", rescue), ("failure", failure)):
    if grp:
        seed, j, d = sorted(grp, key=lambda x: x[2])[len(grp) // 2]
        t_, pk_, po_, ok_, oo_ = ov_lookup[(seed, j)]
        picks.append((kind, seed, j))
        print(f"\n{kind.upper()} median case: seed {seed}, volume {j} | true={CLASS_NAMES[t_]} | "
              f"pred +{CFG}={CLASS_NAMES[pk_]}, Original={CLASS_NAMES[po_]} | "
              f"overlap +{CFG}={ok_:.3f}, Original={oo_:.3f} (change {d:+.3f})")

if NO_FIG or not picks:
    sys.exit(0)

if CFG != '1024':
    print('\nFigures: the plotting function labels the third panel "MDBPE 1024" -- use config 1024 for figures.'); sys.exit(0)
# figures: reuse the exact plotting function (and models/data paths) of the rollout script
import organmnist3d_attention_rollout as R
bp = {'512': R.BASE_PATH_512, '1024': R.BASE_PATH_1024}[CFG]
sp = {'512': R.SAVE_PATH_512, '1024': R.SAVE_PATH_1024}[CFG]
rn = {'512': R.RUN_NAME_512, '1024': R.RUN_NAME_1024}[CFG]
nt = {'512': R.NUM_TOKENS_512, '1024': R.NUM_TOKENS_1024}[CFG]
for kind, seed, j in picks:
    print(f"\nPlotting {kind} case (seed {seed}, image {j})...")
    R.plot_final_layer_comparison(
        j, bp, sp, rn, seed, nt,
        R.BASE_PATH_ORIGINAL, R.SAVE_PATH_ORIGINAL, R.RUN_NAME_ORIGINAL, seed, R.NUM_TOKENS_ORIGINAL)
    src = os.path.join(R.OUTPUT_DIR, f"final_layer_overlay_organ_img{j}.png")
    dst = os.path.join(R.OUTPUT_DIR, f"{kind}_{CFG}_seed{seed}_img{j}.png")
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"  -> {dst}")
