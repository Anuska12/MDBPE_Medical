"""
Rescue / failure cases from the saved ACDC rollout results -- same method as
rescue_failure_derma.py / rescue_failure_breast.py, no recomputation of overlaps.

Reads  overlap_preds_acdc_fold<F>_seed<S>.npz  (written by attention_rollout_acdc.py),
pools ALL runs (5 folds x 4 seeds), and for MDBPE <cfg> vs Original (same fold + seed):
  rescue  = MDBPE correct, Original wrong
  failure = MDBPE wrong,   Original correct
prints counts, overlap change (MDBPE - Original) per group and a t-test between the
groups, then draws the figure for the MEDIAN rescue and MEDIAN failure case.

usage:
  python3 -u rescue_failure_acdc.py 2048 --no-fig   # numbers only
  python3 -u rescue_failure_acdc.py 2048            # + figures
  python3 -u rescue_failure_acdc.py 1024 --no-fig
"""
import os, sys, glob, re
import numpy as np
from scipy.stats import ttest_ind

CFG    = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '2048'
NO_FIG = '--no-fig' in sys.argv
OUTPUT_DIR = "/your/path/4d/attention_rollout"
CLASS_NAMES = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

runs = []
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "overlap_preds_acdc_fold*_seed*.npz"))):
    fold, seed = map(int, re.search(r"fold(\d+)_seed(\d+)", f).groups())
    runs.append((fold, seed, np.load(f)))
print(f"Runs used: {len(runs)} (fold, seed)  (+{CFG} vs Original)")

rescue, failure, per_run, info = [], [], [], {}
for fold, seed, z in runs:
    def table(key):
        return (dict(zip(z[f"{key}_j"], z[f"{key}_ov"])),
                dict(zip(z[f"{key}_j"], z[f"{key}_pred"])),
                dict(zip(z[f"{key}_j"], z[f"{key}_true"])))
    ovo, po, tru = table('original')
    ovk, pk, tk = table(CFG)
    common = sorted(set(ovo) & set(ovk))
    assert all(tru[j] == tk[j] for j in common), "true labels differ between configs"
    r = [j for j in common if pk[j] == tru[j] and po[j] != tru[j]]
    f_ = [j for j in common if pk[j] != tru[j] and po[j] == tru[j]]
    rescue  += [(fold, seed, j, ovk[j] - ovo[j]) for j in r]
    failure += [(fold, seed, j, ovk[j] - ovo[j]) for j in f_]
    for j in common:
        info[(fold, seed, j)] = (tru[j], pk[j], po[j], ovk[j], ovo[j])
    per_run.append((fold, seed, len(common), len(r), len(f_)))

print(f"\n{'fold':>4} {'seed':>4} {'n':>4} {'rescue':>7} {'failure':>8}")
for fold, seed, n, nr, nf in per_run:
    print(f"{fold:>4} {seed:>4} {n:>4} {nr:>7} {nf:>8}")
nr = np.array([x[3] for x in per_run]); nf = np.array([x[4] for x in per_run])
print(f"per run: rescue {nr.mean():.1f} ± {nr.std():.1f}, failure {nf.mean():.1f} ± {nf.std():.1f}")

def describe(name, grp):
    if not grp:
        print(f"  {name}: none"); return None
    d = np.array([x[3] for x in grp])
    cls = np.bincount([info[x[:3]][0] for x in grp], minlength=len(CLASS_NAMES))
    print(f"  {name}: overlap change (+{CFG} - Original) {d.mean():+.4f} ± {d.std():.4f} (n={len(grp)})")
    print("     true classes: " + ", ".join(f"{CLASS_NAMES[c]}={n}" for c, n in enumerate(cls) if n))
    return d

print(f"\nPooled over {len(runs)} runs:")
dr = describe("rescue ", rescue)
df = describe("failure", failure)
if dr is not None and df is not None and len(dr) > 1 and len(df) > 1:
    t, p = ttest_ind(dr, df)
    print(f"  rescue vs failure overlap change: t={t:.4f}, p={p:.4g}"
          "  (pooled over runs; a patient can appear once per seed)")

picks = []
for kind, grp in (("rescue", rescue), ("failure", failure)):
    if grp:
        fold, seed, j, d = sorted(grp, key=lambda x: x[3])[len(grp) // 2]
        t_, pk_, po_, ok_, oo_ = info[(fold, seed, j)]
        picks.append((kind, fold, seed, j))
        print(f"\n{kind.upper()} median case: fold {fold} seed {seed}, patient {j:03d} | "
              f"true={CLASS_NAMES[t_]} | pred +{CFG}={CLASS_NAMES[pk_]}, Original={CLASS_NAMES[po_]} | "
              f"overlap +{CFG}={ok_:.3f}, Original={oo_:.3f} (change {d:+.3f})")

if NO_FIG or not picks:
    sys.exit(0)

import attention_rollout_acdc as R
for kind, fold, seed, j in picks:
    print(f"\nPlotting {kind} case (fold {fold}, seed {seed}, patient {j:03d})...")
    src = R.plot_final_layer_comparison(int(j), fold, seed, CFG)
    dst = os.path.join(R.OUTPUT_DIR, f"{kind}_{CFG}_fold{fold}_seed{seed}_patient{int(j):03d}.png")
    os.replace(src, dst)
    print(f"  -> {dst}")
