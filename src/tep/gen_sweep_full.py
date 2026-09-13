"""
Extended magnitudes on the FULL fault set.

The first extended sweep covered only the four deviations that stay below the
detection floor at benchmark magnitude.  Plotting it on the same axis as the
0.125x to 2x sweep put two different fault populations on one curve, which
reads as the method collapsing at 4x when in fact the fault set changed.

This generates 4x and 8x for every active disturbance, so a comparable curve
can be drawn over one fixed population.

Two disturbances have a natural magnitude ceiling and are excluded rather than
driven to nonsense:

  IDV(6)  enters as FTM(3) = VPOS(3)*(1 - IDV(6)*FMAG), a fractional loss of
          the A feed.  FMAG above 1 is a negative flow.
  IDV(7)  enters as (1 - 0.2*IDV(7)*FMAG); FMAG above 5 is likewise negative.

Runs that trip the plant are kept and flagged; the analysis drops them by the
same rule it uses everywhere else.
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
from src.tep import gen_data as G                           # noqa: E402
from src.tep import tep_constants as C                      # noqa: E402

OUT = os.path.join("data", "tep")
LEVELS = [4.0, 8.0]
NSEED = 4
CAPPED = {6: 1.0, 7: 5.0}          # disturbances with a magnitude ceiling


def main():
    faults = [f for f in C.ACTIVE_IDV if f not in CAPPED]
    print("faults swept: %s" % faults)
    print("excluded for having a magnitude ceiling: %s" % sorted(CAPPED))
    recs = []
    t0 = time.time()
    for f in faults:
        for g in LEVELS:
            for i in range(NSEED):
                s = 1_1000_000 + f * 100_000 + int(g * 100) * 17 + i * 7919
                if f in G.STEP_FAULTS:
                    ym, yv, ish = G.simulate(idv=f, fmag=g, seed=s)
                    fm, wk = g, 1.0
                else:
                    ym, yv, ish = G.simulate(idv=f, walk=g, seed=s)
                    fm, wk = 1.0, g
                recs.append({"x": np.hstack([ym, yv]), "split": "sweep_full",
                             "fault": f, "seed": s, "fmag": fm, "noise": 1.0,
                             "walk": wk, "shutdown": ish})
        print("IDV%-3d done  %.0fs" % (f, time.time() - t0), flush=True)
    X, meta = G._pack(recs)
    np.save(os.path.join(OUT, "sweep_full_X.npy"), X)
    json.dump(meta, open(os.path.join(OUT, "sweep_full_meta.json"), "w"))
    trip = sum(1 for m in meta if m["shutdown"])
    print("saved sweep_full X%s  shutdowns %d/%d  %.0fs"
          % (X.shape, trip, len(meta), time.time() - t0))
    for g in LEVELS:
        sub = [m for m in meta if max(m["fmag"], m["walk"]) == g]
        ok = sum(1 for m in sub if not m["shutdown"])
        print("  level %.0fx: %d runs, %d survived without a trip" %
              (g, len(sub), ok))


if __name__ == "__main__":
    main()
