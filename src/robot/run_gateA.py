"""
Step 0 / Gate A.

Question: can the voraus-AD *anomaly recording categories* be mapped onto
structural fault modes of the robot model at all?  The dataset labels
recordings by what the operator did to the cell, not by which model
equation deviates, and the two are not the same thing.

Pass criteria fixed by the experiment plan:
  A.1  at least six structural fault modes are covered by mappable
       categories;
  A.2  at least one non-singleton indiscernibility class exists at the full
       sensor set -- otherwise the upper bound R1 is vacuous on this system
       and the empirical anchor has to move to the Tennessee Eastman model.

The mapping table itself is the deliverable and goes into the paper.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.structural import equiv                          # noqa: E402
from src.robot.robot_structural import (                  # noqa: E402
    CATEGORY_MAP, EXCLUDED_CHANNELS, FAULTS, FAULT_INFO, SENSORS, build_model,
)

OUT = os.path.join("results", "gates")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 72)
    print("Gate A: voraus-AD anomaly category -> structural fault mode")
    print("=" * 72)
    covered, excluded = set(), []
    for cat, (modes, why, keep) in CATEGORY_MAP.items():
        if keep:
            covered.update(modes)
            print("  %-20s -> %s" % (cat, ", ".join(modes)))
        else:
            excluded.append(cat)
            print("  %-20s -> EXCLUDED" % cat)
    print("\nexcluded categories: %s" % ", ".join(excluded))
    print("structural fault modes covered by mappable categories: %d of %d"
          % (len(covered), len(FAULT_INFO)))

    model = build_model()
    print("\nRobot structural model: %d equations, %d unknowns, %d sensors, "
          "%d faults, redundancy %d"
          % (model.ne(), model.nx(), model.nz(), model.nf(),
             model.Redundancy()))
    print("channels deliberately excluded from the sensor set:")
    for k, v in EXCLUDED_CHANNELS.items():
        print("    %-34s %s" % (k, v))

    res = equiv.quotient_summary(model, FAULTS)
    blocks = res["blocks"]
    print("\nStructural indiscernibility quotient at the full sensor set "
          "(%d classes)" % res["n_classes"])
    for k, b in enumerate(blocks):
        print("  C%-2d %s" % (k, "{" + ", ".join(f[1:] for f in b) + "}"))
    print("  undetectable: %s"
          % ([f[1:] for f in res["undetectable"]] or "none"))

    a1 = len(covered) >= 6
    a2 = res["max_class_size"] > 1
    print("\nGate A tests")
    print("  A.1 >= 6 structural fault modes mappable: %s (%d)"
          % ("PASS" if a1 else "FAIL", len(covered)))
    print("  A.2 at least one non-singleton class: %s (largest class %d)"
          % ("PASS" if a2 else "FAIL", res["max_class_size"]))
    verdict = a1 and a2
    print("\nGATE A: %s" % ("PASS" if verdict else "FAIL"))
    if not a2:
        print("  -> fall back: demote voraus-AD to a reality check (E4, E10 "
              "only) and anchor R1 empirically on the Tennessee Eastman "
              "model.")

    payload = {
        "mapping": {c: {"modes": m, "rationale": w, "included": k}
                    for c, (m, w, k) in CATEGORY_MAP.items()},
        "excluded_categories": excluded,
        "excluded_channels": EXCLUDED_CHANNELS,
        "n_modes_covered": len(covered),
        "n_modes_total": len(FAULT_INFO),
        "model": {"ne": model.ne(), "nx": model.nx(), "nz": model.nz(),
                  "nf": model.nf(), "redundancy": int(model.Redundancy()),
                  "n_sensors": len(SENSORS)},
        "blocks": [[f[1:] for f in b] for b in blocks],
        "n_classes": res["n_classes"],
        "max_class_size": res["max_class_size"],
        "indiscernible_mass": res["indiscernible_mass"],
        "undetectable": [f[1:] for f in res["undetectable"]],
        "tests": {"A1": bool(a1), "A2": bool(a2), "verdict": bool(verdict)},
    }
    with open(os.path.join(OUT, "gateA_voraus.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    np.savetxt(os.path.join(OUT, "gateA_isolability.csv"),
               res["isolability"], fmt="%d", delimiter=",",
               header=",".join(f[1:] for f in FAULTS))
    print("\nwrote %s" % os.path.join(OUT, "gateA_voraus.json"))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
