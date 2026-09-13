"""
Supplementary sweep data.

Two gaps in the first sweep pass:

1.  The noise sweep carried only five nominal runs per noise level, which is
    about 130 windows.  Split conformal at alpha = 0.01 needs at least 99
    calibration points AFTER the fit/calibration split, so the threshold came
    back infinite and nothing could fire.  This adds enough nominal runs at
    each noise level to calibrate properly at that level.

2.  The magnitude sweep topped out at twice the benchmark magnitude, which is
    not enough for the three deviations that stay below the detection floor
    (D feed temperature, A and C feed temperature, condenser coolant valve).
    Without pushing those further the merge gap cannot be shown to close.
    Magnitudes above 2x trip the plant for the strong disturbances, so the
    extension is applied only to the weak ones.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import tepsim                                             # noqa: E402
from src.tep import gen_data as G                         # noqa: E402

OUT = os.path.join("data", "tep")

WEAK_FAULTS = [3, 9, 10, 15]          # the ones below the detection floor
BIG_GRID = [4.0, 8.0, 16.0, 32.0]
N_NOM_PER_NOISE = 25


def main():
    t0 = time.time()

    # ---- more nominal runs at each noise level
    recs = []
    for g in G.NOISE_GRID:
        for i in range(N_NOM_PER_NOISE):
            s = 7_000_000 + int(g * 1000) * 31 + i * 7919
            ym, yv, ishut = G.simulate(noise=g, seed=s)
            recs.append({"x": np.hstack([ym, yv]), "split": "noise_nominal",
                         "fault": 0, "seed": s, "fmag": 1.0, "noise": g,
                         "walk": 1.0, "shutdown": ishut})
        print("noise %.2f: %d nominal runs  %.0fs"
              % (g, N_NOM_PER_NOISE, time.time() - t0), flush=True)
    X, meta = G._pack(recs)
    np.save(os.path.join(OUT, "noise_nominal_X.npy"), X)
    json.dump(meta, open(os.path.join(OUT, "noise_nominal_meta.json"), "w"))
    print("saved noise_nominal X%s" % (X.shape,), flush=True)

    # ---- push the weak deviations far above benchmark magnitude
    recs = []
    for f in WEAK_FAULTS:
        for g in BIG_GRID:
            for i in range(4):
                s = 8_000_000 + f * 100_000 + int(g * 100) * 13 + i * 7919
                if f in G.STEP_FAULTS:
                    ym, yv, ishut = G.simulate(idv=f, fmag=g, seed=s)
                    fm, wk = g, 1.0
                else:
                    ym, yv, ishut = G.simulate(idv=f, walk=g, seed=s)
                    fm, wk = 1.0, g
                recs.append({"x": np.hstack([ym, yv]), "split": "sweep_big",
                             "fault": f, "seed": s, "fmag": fm, "noise": 1.0,
                             "walk": wk, "shutdown": ishut})
        print("fault IDV%d extended  %.0fs" % (f, time.time() - t0),
              flush=True)
    X, meta = G._pack(recs)
    np.save(os.path.join(OUT, "sweep_big_X.npy"), X)
    json.dump(meta, open(os.path.join(OUT, "sweep_big_meta.json"), "w"))
    shut = sum(1 for m in meta if m["shutdown"])
    print("saved sweep_big X%s  shutdowns %d/%d  total %.0fs"
          % (X.shape, shut, len(meta), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
