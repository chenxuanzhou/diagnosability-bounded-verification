"""
E1, augmentation direction.

The ablation ladder answers "what do I lose as instrumentation gets
poorer".  This script answers the sharper question the paper actually
needs: given a non-singleton indiscernibility class, is there ANY sensor
that splits it?

Two outcomes matter and they mean very different things.

  splittable    the class collapses once a particular instrument is added.
                The ambiguity is an instrumentation deficit, and the right
                engineering answer is to buy the sensor.

  irreducible   no sensor in the catalogue splits it, because the fault
                modes act on the very same model equation.  No amount of
                instrumentation can separate them, so any system that does
                separate them is using information that never travelled
                through a sensor channel at all.  This is the population
                the LLM has to earn its place on.

The irreducible classes are the theory's own pre-specification of where an
LLM must be useful, computed before any agent is run.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.structural import equiv                              # noqa: E402
from src.robot import robot_structural as rob                 # noqa: E402
from src.tep import tep_structural as tep                     # noqa: E402

OUT = os.path.join("results", "E1")
os.makedirs(OUT, exist_ok=True)


def analyse(name, build, base_sensors, optional, fault_names):
    print("\n" + "=" * 74)
    print("E1 augmentation: %s" % name)
    print("=" * 74)

    base = equiv.quotient_summary(build(sensors=list(base_sensors)),
                                  fault_names)
    base_blocks = [b for b in base["blocks"] if len(b) > 1]
    print("non-singleton classes at the installed sensor set: %d"
          % len(base_blocks))

    # one sensor at a time
    per_sensor = {}
    for s in optional:
        res = equiv.quotient_summary(
            build(sensors=list(base_sensors) + [s]), fault_names)
        per_sensor[s] = [[f[1:] for f in b] for b in res["blocks"]]

    # every optional sensor at once: the best any instrumentation upgrade
    # inside this catalogue can do
    allres = equiv.quotient_summary(
        build(sensors=list(base_sensors) + list(optional)), fault_names)
    all_cls = equiv.class_of(allres["blocks"])

    rows = []
    for blk in base_blocks:
        names = [f[1:] for f in blk]
        splitters = []
        for s, blocks in per_sensor.items():
            cls = {f: k for k, b in enumerate(blocks) for f in b}
            if len({cls[n] for n in names}) > 1:
                splitters.append(s)
        still_together = len({all_cls[f] for f in blk}) == 1
        rows.append({"class": names, "splitting_sensors": splitters,
                     "irreducible": bool(still_together)})
        tag = "IRREDUCIBLE" if still_together else "splittable"
        print("  {%s}  %-12s  %s"
              % (", ".join(names), tag,
                 ", ".join(splitters) if splitters else "-"))

    print("\n  with the whole catalogue added: %d classes, largest %d, "
          "indiscernible mass %.3f"
          % (allres["n_classes"], allres["max_class_size"],
             allres["indiscernible_mass"]))
    return {
        "installed": {"n_classes": base["n_classes"],
                      "max_class_size": base["max_class_size"],
                      "indiscernible_mass": base["indiscernible_mass"],
                      "blocks": [[f[1:] for f in b] for b in base["blocks"]]},
        "fully_augmented": {"n_classes": allres["n_classes"],
                            "max_class_size": allres["max_class_size"],
                            "indiscernible_mass": allres["indiscernible_mass"],
                            "blocks": [[f[1:] for f in b]
                                       for b in allres["blocks"]]},
        "classes": rows,
        "catalogue": list(optional),
    }


def main():
    out = {}
    out["tep"] = analyse("Tennessee Eastman", tep.build_model,
                         list(tep.SENSORS.keys()),
                         list(tep.OPTIONAL_SENSORS.keys()), tep.FAULTS)
    out["robot"] = analyse("voraus six-axis robot", rob.build_model,
                           list(rob.SENSORS.keys()),
                           list(rob.OPTIONAL_SENSORS.keys()), rob.FAULTS)

    irr = {k: [r["class"] for r in v["classes"] if r["irreducible"]]
           for k, v in out.items()}
    print("\n" + "=" * 74)
    print("Irreducible classes (no instrument in the catalogue separates "
          "them):")
    for k, v in irr.items():
        for c in v:
            print("  %-6s {%s}" % (k, ", ".join(c)))
    out["irreducible"] = irr

    with open(os.path.join(OUT, "E1_augmentation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E1_augmentation.json"))


if __name__ == "__main__":
    main()
