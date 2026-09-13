"""
Validate the rebuilt Tennessee Eastman simulator against the published
d00_te / d01_te data sets, and check that the three new knobs (fault
magnitude, measurement noise, random-walk amplitude) behave monotonically.

Reference data: d00_te.dat and d01_te.dat from the Braatz-group
distribution -- 960 samples at the 3 min rate, i.e. 48 h, with the fault
introduced after 8 h in the fault cases.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import tepsim                                             # noqa: E402

RAW = os.path.join("data", "raw", "tep_src")
OUT = os.path.join("results", "E1")
os.makedirs(OUT, exist_ok=True)

NSAMP = 180                 # 3 min at a 1 s integration step
HOURS = 48
NPTS = HOURS * 3600
NS = NPTS // NSAMP
TIDV = 8 * 3600             # fault switched on after 8 h


def run(idv_list=(), fmag=1.0, noise=1.0, walk=1.0, seed=1431655765.0,
        hours=HOURS, tidv=TIDV):
    npts = int(hours * 3600)
    ns = npts // NSAMP
    idv = np.zeros(20, dtype=np.int32)
    fm = np.ones(20)
    for k in idv_list:
        idv[k - 1] = 1
        fm[k - 1] = fmag
    ym, yv, ishut = tepsim.tesim(npts, NSAMP, idv, tidv, fm, noise, walk,
                                 1.0, seed, ns)
    return ym, yv, int(ishut)


def cmp_stats(sim, ref, name, cols=(0, 6, 8, 10, 12, 17, 19, 20, 21)):
    rows = []
    for c in cols:
        rows.append({
            "xmeas": c + 1,
            "sim_mean": float(sim[:, c].mean()),
            "ref_mean": float(ref[:, c].mean()),
            "sim_std": float(sim[:, c].std()),
            "ref_std": float(ref[:, c].std()),
        })
    print("\n%s  (sim vs published)" % name)
    print("  XMEAS   sim mean      ref mean     sim std    ref std   "
          "mean rel.err")
    worst = 0.0
    for r in rows:
        rel = abs(r["sim_mean"] - r["ref_mean"]) / max(abs(r["ref_mean"]), 1e-9)
        worst = max(worst, rel)
        print("  %5d %12.4f %12.4f %10.4f %10.4f %10.2e"
              % (r["xmeas"], r["sim_mean"], r["ref_mean"],
                 r["sim_std"], r["ref_std"], rel))
    print("  worst relative mean error: %.2e" % worst)
    return rows, worst


def main():
    report = {}

    # ---------------------------------------------------- nominal operation
    ym, _, ishut = run()
    ref = np.loadtxt(os.path.join(RAW, "d00_te.dat"))
    rows, worst = cmp_stats(ym, ref, "nominal (IDV none) vs d00_te")
    report["d00"] = {"rows": rows, "worst_rel_mean_err": worst,
                     "shutdown_step": ishut}

    # ------------------------------------------------------------- IDV(1)
    ym1, _, ishut1 = run(idv_list=(1,))
    ref1 = np.loadtxt(os.path.join(RAW, "d01_te.dat"))
    rows1, worst1 = cmp_stats(ym1, ref1, "IDV(1) vs d01_te")
    report["d01"] = {"rows": rows1, "worst_rel_mean_err": worst1,
                     "shutdown_step": ishut1}

    # ------------------------------------- knob 1: fault magnitude sweep
    print("\nfault magnitude sweep on IDV(1): mean |XMEAS| shift after the "
          "fault, relative to the pre-fault window")
    pre = slice(0, TIDV // NSAMP)
    post = slice(TIDV // NSAMP, NS)
    mag_rows = []
    base = run()[0]
    scale = base[pre].std(0) + 1e-9
    for g in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        y, _, sh = run(idv_list=(1,), fmag=g)
        shift = np.abs((y[post].mean(0) - y[pre].mean(0)) / scale).mean()
        mag_rows.append({"fmag": g, "mean_norm_shift": float(shift),
                         "shutdown": sh})
        print("  FMAG = %4.2f   mean normalised shift = %7.3f   shutdown=%d"
              % (g, shift, sh))
    report["magnitude_sweep_idv1"] = mag_rows

    # ------------------------------------------- knob 2: noise gain sweep
    print("\nmeasurement noise sweep (nominal run): std of XMEAS(9), "
          "reactor temperature")
    noise_rows = []
    for g in (0.0, 0.5, 1.0, 2.0, 4.0):
        y, _, sh = run(noise=g)
        noise_rows.append({"xnsg": g, "std_xmeas9": float(y[:, 8].std()),
                           "std_xmeas7": float(y[:, 6].std()), "shutdown": sh})
        print("  XNSG = %4.2f   std XMEAS9 = %.5f   std XMEAS7 = %.4f"
              % (g, y[:, 8].std(), y[:, 6].std()))
    report["noise_sweep"] = noise_rows

    # --------------------------------- knob 3: random-walk amplitude sweep
    print("\nrandom-walk amplitude sweep on IDV(9), D feed temperature")
    walk_rows = []
    for g in (0.0, 0.5, 1.0, 2.0, 4.0):
        y, _, sh = run(idv_list=(9,), walk=g)
        shift = np.abs((y[post].std(0) - y[pre].std(0)) / scale).mean()
        walk_rows.append({"sspang": g, "mean_norm_std_change": float(shift),
                          "shutdown": sh})
        print("  SSPANG = %4.2f  mean normalised std change = %7.4f"
              % (g, shift))
    report["walk_sweep_idv9"] = walk_rows

    # ----------------------------------------------- seed reproducibility
    a = run(seed=12345.0)[0]
    b = run(seed=12345.0)[0]
    c = run(seed=98765.0)[0]
    same = bool(np.allclose(a, b))
    diff = float(np.abs(a - c).max())
    print("\nseed reproducibility: identical for equal seeds = %s ; "
          "max abs diff across seeds = %.4g" % (same, diff))
    report["seed"] = {"reproducible": same, "max_diff_across_seeds": diff}

    with open(os.path.join(OUT, "sim_validation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % os.path.join(OUT, "sim_validation.json"))


if __name__ == "__main__":
    main()
