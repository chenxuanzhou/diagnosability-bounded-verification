"""Stiction-deadband sweep, so the valve faults have a magnitude axis.

IDV(14), (15) and (19) are boolean flags in the reference source with a
fixed deadband VST(j).  Without a deadband knob they have no severity to
sweep, which is why the condenser coolant valve stayed below the detection
floor at every level of the magnitude sweep.  The rebuilt simulator scales
VST, so their severity can be varied like any other deviation.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from src.tep import gen_data as G

GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
FAULTS = [14, 15, 19]
OUT = os.path.join("data", "tep")

recs = []
t0 = time.time()
for f in FAULTS:
    for g in GRID:
        for i in range(4):
            s = 9_000_000 + f * 100_000 + int(g * 100) * 13 + i * 7919
            ym, yv, ish = G.simulate(idv=f, stiction=g, seed=s)
            recs.append({"x": np.hstack([ym, yv]), "split": "sweep_stick",
                         "fault": f, "seed": s, "fmag": 1.0, "noise": 1.0,
                         "walk": 1.0, "stiction": g, "shutdown": ish})
    print("IDV%d done %.0fs" % (f, time.time() - t0), flush=True)
X, meta = G._pack(recs)
np.save(os.path.join(OUT, "sweep_stick_X.npy"), X)
json.dump(meta, open(os.path.join(OUT, "sweep_stick_meta.json"), "w"))
print("saved sweep_stick X%s  shutdowns %d/%d"
      % (X.shape, sum(1 for m in meta if m["shutdown"]), len(meta)))
