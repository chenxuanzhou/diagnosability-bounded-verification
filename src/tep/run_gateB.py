"""
Step 0 / Gate B.

Claim under test: a purely structural analysis of the Tennessee Eastman
Process -- carried out without touching a single data point -- reproduces
the empirical isolation difficulty that the TEP benchmarking community has
reported for two decades.  Concretely:

  B.1  every IDV the literature flags as hard (3, 9, 15, 21) lands in a
       NON-singleton indiscernibility class;
  B.2  every IDV the literature flags as easy (1, 2, 4, 6, 7, 14) is either
       a singleton or is separated from the hard group;
  B.3  IDV(3) and IDV(9) fall in the same class (same physical location,
       step vs random), and IDV(15) is not a singleton.

Gate passes only if B.1 and B.3 hold.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.structural import equiv                      # noqa: E402
from src.tep.tep_structural import (                  # noqa: E402
    FAULTS, IDVS, IDV_INFO, SENSORS, build_model,
)

OUT = os.path.join("results", "gates")
os.makedirs(OUT, exist_ok=True)

# pre-registered in results/gates/prereg_labels.json.  IDV21 is not
# implemented in teprob.f, so the hard set used for the test is {3,9,15}.
HARD = ["IDV3", "IDV9", "IDV15"]
EASY = ["IDV1", "IDV2", "IDV4", "IDV6", "IDV7", "IDV14"]


def main():
    model = build_model()
    print("TEP structural model: %d equations, %d unknowns, %d knowns, "
          "%d faults, redundancy %d"
          % (model.ne(), model.nx(), model.nz(), model.nf(),
             model.Redundancy()))

    res = equiv.quotient_summary(model, FAULTS)
    blocks = res["blocks"]
    cls = equiv.class_of(blocks)

    print("\nDetectability")
    print("  detectable  :", [f[1:] for f in res["detectable"]])
    print("  undetectable:", [f[1:] for f in res["undetectable"]])

    print("\nStructural indiscernibility quotient (%d classes)"
          % res["n_classes"])
    for k, b in enumerate(blocks):
        tags = {IDV_INFO[f[1:]][1] for f in b}
        print("  C%-2d %-28s  literature: %s"
              % (k, "{" + ", ".join(f[1:] for f in b) + "}",
                 "/".join(sorted(tags))))

    # ---------------------------------------------------------- gate tests
    hard_nonsingleton = {h: len(blocks[cls["f" + h]]) > 1 for h in HARD}
    b1 = all(hard_nonsingleton.values())
    b3 = (cls["fIDV3"] == cls["fIDV9"]) and len(blocks[cls["fIDV15"]]) > 1
    easy_singleton = {e: len(blocks[cls["f" + e]]) == 1 for e in EASY}
    b2 = all(cls["f" + e] != cls["f" + h] for e in EASY for h in HARD)

    print("\nGate B tests")
    print("  B.1 every literature-hard IDV sits in a non-singleton class:",
          "PASS" if b1 else "FAIL", hard_nonsingleton)
    print("  B.2 no literature-easy IDV shares a class with a hard one:",
          "PASS" if b2 else "FAIL")
    print("  B.3 IDV3 ~ IDV9 and IDV15 non-singleton:",
          "PASS" if b3 else "FAIL")
    print("      (easy faults that are singletons: %s)"
          % [e for e, v in easy_singleton.items() if v])

    verdict = b1 and b3
    print("\nGATE B: %s" % ("PASS" if verdict else "FAIL"))

    payload = {
        "model": {"ne": model.ne(), "nx": model.nx(), "nz": model.nz(),
                  "nf": model.nf(), "redundancy": int(model.Redundancy()),
                  "n_sensors": len(SENSORS)},
        "faults": IDVS,
        "blocks": [[f[1:] for f in b] for b in blocks],
        "n_classes": res["n_classes"],
        "max_class_size": res["max_class_size"],
        "indiscernible_mass": res["indiscernible_mass"],
        "detectable": [f[1:] for f in res["detectable"]],
        "undetectable": [f[1:] for f in res["undetectable"]],
        "tests": {"B1": bool(b1), "B2": bool(b2), "B3": bool(b3),
                  "verdict": bool(verdict)},
        "hard_set": HARD, "easy_set": EASY,
    }
    with open(os.path.join(OUT, "gateB_tep.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    np.savetxt(os.path.join(OUT, "gateB_isolability.csv"),
               res["isolability"], fmt="%d", delimiter=",",
               header=",".join(IDVS))
    print("\nwrote %s" % os.path.join(OUT, "gateB_tep.json"))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
