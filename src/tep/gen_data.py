"""
Generate the Tennessee Eastman corpus.

Splits follow the reproducibility protocol in the plan:

  nominal_cal   nominal runs reserved for conformal calibration.  They are
                never used for fitting, thresholding or model selection of
                anything else.
  nominal_fit   nominal runs used to fit the data-driven residual
                predictors.  Disjoint seeds from nominal_cal.
  nominal_test  nominal runs used at evaluation time.
  fault         one run per (fault, seed) at benchmark magnitude.
  sweep_mag     magnitude sweep, for the bound-tightness experiment.
  sweep_noise   measurement-noise sweep, likewise.

Seeds are drawn from disjoint blocks so that no calibration seed can
reappear in a fitting or test run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import tepsim                                             # noqa: E402

OUT = os.path.join("data", "tep")
os.makedirs(OUT, exist_ok=True)

NSAMP = 180                      # 3 min sampling, the benchmark rate
HOURS = 48
NPTS = HOURS * 3600
NS = NPTS // NSAMP               # 960 samples
TIDV = 8 * 3600                  # fault switched on after 8 h
IDV_ON_SAMPLE = TIDV // NSAMP    # sample index of fault onset

FAULTS = list(range(1, 21))

# disjoint seed blocks
SEED_BLOCKS = {
    "nominal_cal": (1_000_000, 40),
    "nominal_fit": (2_000_000, 40),
    "nominal_test": (3_000_000, 30),
    "fault": (4_000_000, 8),         # per fault
    "sweep_mag": (5_000_000, 5),
    "sweep_noise": (6_000_000, 5),
}

# Magnitude sweep: only the step faults have a meaningful FMAG knob; the
# random-variation faults are swept through the random-walk amplitude.
STEP_FAULTS = [1, 2, 3, 4, 5, 6, 7]
WALK_FAULTS = [8, 9, 10, 11, 12, 13, 16, 17, 18, 20]
MAG_GRID = [0.125, 0.25, 0.5, 1.0, 2.0]
NOISE_GRID = [0.25, 0.5, 1.0, 2.0, 3.0]


def simulate(idv=None, fmag=1.0, noise=1.0, walk=1.0, stiction=1.0,
             seed=1.0):
    idv_vec = np.zeros(20, dtype=np.int32)
    fm = np.ones(20)
    if idv:
        idv_vec[idv - 1] = 1
        fm[idv - 1] = fmag
    ym, yv, ishut = tepsim.tesim(NPTS, NSAMP, idv_vec, TIDV, fm, noise, walk,
                                 stiction, float(seed), NS)
    return ym, yv, int(ishut)


def _pack(records):
    X = np.stack([r["x"] for r in records]).astype(np.float32)
    meta = [{k: v for k, v in r.items() if k != "x"} for r in records]
    return X, meta


def gen_nominal(tag, n, seed0):
    recs = []
    for i in range(n):
        s = seed0 + i * 7919
        ym, yv, ishut = simulate(seed=s)
        recs.append({"x": np.hstack([ym, yv]), "split": tag, "fault": 0,
                     "seed": s, "fmag": 1.0, "noise": 1.0, "walk": 1.0,
                     "shutdown": ishut})
    return recs


def gen_faults(n_per_fault, seed0):
    recs = []
    for f in FAULTS:
        for i in range(n_per_fault):
            s = seed0 + f * 100_000 + i * 7919
            ym, yv, ishut = simulate(idv=f, seed=s)
            recs.append({"x": np.hstack([ym, yv]), "split": "fault",
                         "fault": f, "seed": s, "fmag": 1.0, "noise": 1.0,
                         "walk": 1.0, "shutdown": ishut})
    return recs


def gen_mag_sweep(n_seeds, seed0):
    recs = []
    for f in FAULTS:
        for g in MAG_GRID:
            for i in range(n_seeds):
                s = seed0 + f * 100_000 + int(g * 1000) * 13 + i * 7919
                if f in STEP_FAULTS:
                    ym, yv, ishut = simulate(idv=f, fmag=g, seed=s)
                    fm, wk = g, 1.0
                else:
                    ym, yv, ishut = simulate(idv=f, walk=g, seed=s)
                    fm, wk = 1.0, g
                recs.append({"x": np.hstack([ym, yv]), "split": "sweep_mag",
                             "fault": f, "seed": s, "fmag": fm,
                             "noise": 1.0, "walk": wk, "shutdown": ishut})
    return recs


def gen_noise_sweep(n_seeds, seed0):
    recs = []
    for f in [0] + FAULTS:
        for g in NOISE_GRID:
            for i in range(n_seeds):
                s = seed0 + f * 100_000 + int(g * 1000) * 13 + i * 7919
                ym, yv, ishut = simulate(idv=(f or None), noise=g, seed=s)
                recs.append({"x": np.hstack([ym, yv]), "split": "sweep_noise",
                             "fault": f, "seed": s, "fmag": 1.0,
                             "noise": g, "walk": 1.0, "shutdown": ishut})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="core",
                    choices=["core", "sweeps", "all"])
    args = ap.parse_args()

    t0 = time.time()
    groups = {}
    if args.what in ("core", "all"):
        for tag in ("nominal_cal", "nominal_fit", "nominal_test"):
            s0, n = SEED_BLOCKS[tag]
            groups[tag] = gen_nominal(tag, n, s0)
            print("%-13s %3d runs  %.1fs" % (tag, n, time.time() - t0),
                  flush=True)
        s0, n = SEED_BLOCKS["fault"]
        groups["fault"] = gen_faults(n, s0)
        print("%-13s %3d runs  %.1fs"
              % ("fault", len(groups["fault"]), time.time() - t0), flush=True)
    if args.what in ("sweeps", "all"):
        s0, n = SEED_BLOCKS["sweep_mag"]
        groups["sweep_mag"] = gen_mag_sweep(n, s0)
        print("%-13s %3d runs  %.1fs"
              % ("sweep_mag", len(groups["sweep_mag"]), time.time() - t0),
              flush=True)
        s0, n = SEED_BLOCKS["sweep_noise"]
        groups["sweep_noise"] = gen_noise_sweep(n, s0)
        print("%-13s %3d runs  %.1fs"
              % ("sweep_noise", len(groups["sweep_noise"]), time.time() - t0),
              flush=True)

    for tag, recs in groups.items():
        X, meta = _pack(recs)
        np.save(os.path.join(OUT, "%s_X.npy" % tag), X)
        with open(os.path.join(OUT, "%s_meta.json" % tag), "w") as fh:
            json.dump(meta, fh)
        shut = sum(1 for m in meta if m["shutdown"])
        print("saved %-13s X%s  shutdowns %d/%d"
              % (tag, X.shape, shut, len(meta)), flush=True)

    spec = {"nsamp": NSAMP, "hours": HOURS, "n_samples": NS,
            "idv_on_sample": IDV_ON_SAMPLE, "columns":
            ["XMEAS%d" % i for i in range(1, 42)] +
            ["XMV%d" % i for i in range(1, 13)],
            "seed_blocks": SEED_BLOCKS, "mag_grid": MAG_GRID,
            "noise_grid": NOISE_GRID, "step_faults": STEP_FAULTS,
            "walk_faults": WALK_FAULTS}
    with open(os.path.join(OUT, "spec.json"), "w") as fh:
        json.dump(spec, fh, indent=2)
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
