"""
E1: structural characterisation of both systems.

No data is touched.  For each system we compute, at the full sensor set and
along a ladder of progressively poorer instrumentation:

  * degree of structural redundancy
  * the indiscernibility quotient (blocks of mutually non-isolable faults)
  * the number of classes, the largest class, and the indiscernible mass
    (the fraction of fault modes that sit inside a non-singleton class --
    the population the verifier is provably blind inside)
  * which fault modes stop being detectable at all

The full-sensor row is Table I; the ladder is the theoretical curve that
panel (a) of the sensor-ablation figure is compared against.
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

# Ablation ladders: each rung removes one more sensor group, going from the
# fully instrumented plant down towards a minimal one.  The order follows
# what an operator would actually give up first: the most expensive and
# slowest instruments before the cheap fast ones.
TEP_LADDER = [
    ("all 41 measurements", []),
    ("no product analyser", ["analyser_product"]),
    ("no product/purge analysers", ["analyser_product", "analyser_purge"]),
    ("no composition analysers", ["analyser_product", "analyser_purge",
                                  "analyser_feed"]),
    ("no analysers, no duty meters", ["analyser_product", "analyser_purge",
                                      "analyser_feed", "duties"]),
    ("no analysers/duties/coolant temps",
     ["analyser_product", "analyser_purge", "analyser_feed", "duties",
      "coolant_temps"]),
    ("flows, pressures and vessel temps only",
     ["analyser_product", "analyser_purge", "analyser_feed", "duties",
      "coolant_temps", "levels"]),
    ("feed flows, pressures, vessel temps",
     ["analyser_product", "analyser_purge", "analyser_feed", "duties",
      "coolant_temps", "levels", "internal_flows"]),
]

ROBOT_LADDER = [
    ("full instrumentation", []),
    ("single joint torque sensor", ["torque_b"]),
    ("no joint torque sensing", ["torque_b", "torque_a"]),
    ("no joint torque, no link encoders",
     ["torque_b", "torque_a", "link_encoders"]),
    ("motor side only, with voltage",
     ["torque_b", "torque_a", "link_encoders", "bus_current"]),
    ("motor side, no voltage sensing",
     ["torque_b", "torque_a", "link_encoders", "bus_current",
      "motor_voltage"]),
    ("encoders and bus current only",
     ["torque_b", "torque_a", "link_encoders", "bus_current",
      "motor_voltage", "phase_current"]),
]

ROBOT_GROUPS = dict(rob.SENSOR_GROUPS)
ROBOT_GROUPS["torque_a"] = ["torque_sensor_a_%d" % i for i in rob.AX]
ROBOT_GROUPS["torque_b"] = ["torque_sensor_b_%d" % i for i in rob.AX]


def sweep(name, build, all_sensors, groups, ladder, fault_names):
    rows = []
    print("\n" + "=" * 74)
    print("E1 ladder: %s" % name)
    print("=" * 74)
    print("%-42s %5s %5s %6s %6s %7s %6s"
          % ("sensor set", "n_s", "red.", "class", "maxsz", "indisc", "undet"))
    for label, drop in ladder:
        removed = set()
        for g in drop:
            removed.update(groups[g])
        keep = [s for s in all_sensors if s not in removed]
        model = build(sensors=keep)
        res = equiv.quotient_summary(model, fault_names)
        rows.append({
            "label": label, "dropped_groups": drop,
            "n_sensors": len(keep),
            "redundancy": int(model.Redundancy()),
            "n_classes": res["n_classes"],
            "max_class_size": res["max_class_size"],
            "indiscernible_mass": res["indiscernible_mass"],
            "n_undetectable": len(res["undetectable"]),
            "undetectable": [f[1:] for f in res["undetectable"]],
            "blocks": [[f[1:] for f in b] for b in res["blocks"]],
        })
        print("%-42s %5d %5d %6d %6d %7.3f %6d"
              % (label[:42], len(keep), int(model.Redundancy()),
                 res["n_classes"], res["max_class_size"],
                 res["indiscernible_mass"], len(res["undetectable"])))
    return rows


def main():
    out = {}

    out["tep"] = sweep("Tennessee Eastman", tep.build_model,
                       list(tep.SENSORS.keys()), tep.SENSOR_GROUPS,
                       TEP_LADDER, tep.FAULTS)
    print("\nquotient at the full sensor set (TEP):")
    for k, b in enumerate(out["tep"][0]["blocks"]):
        print("   C%-2d {%s}" % (k, ", ".join(b)))

    out["robot"] = sweep("voraus six-axis robot", rob.build_model,
                         list(rob.SENSORS.keys()), ROBOT_GROUPS,
                         ROBOT_LADDER, rob.FAULTS)
    print("\nquotient at the full sensor set (robot):")
    for k, b in enumerate(out["robot"][0]["blocks"]):
        print("   C%-2d {%s}" % (k, ", ".join(b)))

    with open(os.path.join(OUT, "E1_ladders.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E1_ladders.json"))


if __name__ == "__main__":
    main()
