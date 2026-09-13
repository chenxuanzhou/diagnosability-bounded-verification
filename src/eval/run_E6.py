"""
E6: sensor ablation.  The core figure.

At each rung of an instrumentation ladder, three things are computed for the
same plant:

  (a) the theoretical bound -- the structural indiscernibility quotient of
      the fault locations given only the surviving sensors.  No data.
  (b) what the verifier actually achieves -- the residuals that are still
      computable are re-selected, re-calibrated on nominal data at that
      instrumentation, and their realised partition measured.
  (c) how well the pipeline then diagnoses.

The point of the figure is the shape of the curves, not their height.  The
bound moves when instruments are removed and only then.  A verifier built on
the surviving redundancy tracks it.  Nothing an agent does to its own
reasoning can move it, which is what makes the ambiguity an instrumentation
property rather than a modelling one.

The ladder removes the expensive, slow instruments first -- the composition
analysers -- then duty meters, then coolant thermometers, then vessel
levels, which is the order a plant would actually give them up.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import cases as CS                            # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.structural import equiv                            # noqa: E402
from src.structural import run_E1 as E1                     # noqa: E402
from src.tep import tep_structural as tep                   # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import residuals_tep as R                 # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402
from src.verifier import sensor_map as SM                   # noqa: E402

OUT = os.path.join("results", "E6")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA


def subset_bank(F_fit, F_cal, keep_residuals):
    """A conformal bank over a subset of the residual bank."""
    idx = [R.RESIDUAL_NAMES.index(n) for n in keep_residuals]
    m = len(R.RESIDUAL_NAMES)
    cols = idx + [m + i for i in idx]
    bank = CF.ConformalBank(names=keep_residuals)
    bank.fit_scale(F_fit[:, cols])
    bank.calibrate(F_cal[:, cols])
    return bank, cols


def theoretical(sensors):
    """Structural quotient over fault locations for a sensor subset."""
    model = tep.build_model(sensors=sensors)
    res = equiv.quotient_summary(model, tep.FAULTS)
    blocks = []
    for b in res["blocks"]:
        locs = sorted({P.IDV_TO_LOCATION[int(f[4:])] for f in b
                       if int(f[4:]) in P.IDV_TO_LOCATION})
        if locs:
            blocks.append(locs)
    merged = []
    for b in blocks:
        hit = [m for m in merged if set(m) & set(b)]
        if hit:
            hit[0][:] = sorted(set(hit[0]) | set(b))
        else:
            merged.append(sorted(b))
    n = len(P.LOCATION_IDS)
    mass = sum(len(b) for b in merged if len(b) > 1) / float(n)
    undet = [f[1:] for f in res["undetectable"]]
    return {"n_classes": len(merged), "blocks": merged,
            "indiscernible_mass": mass, "undetectable_idvs": undet}


def realised_structural(keep_residuals):
    """Partition induced by the IMPLEMENTED bank's signature columns.

    This is the middle line of the ablation panel and the one that carries
    R2.  It is a structural quantity, computed with no data: the partition of
    the fault locations by identical columns of the fault signature matrix of
    the residuals that survive this instrumentation.  At the full sensor set
    it sits exactly on the theoretical bound, which is the achievability
    result; below that it can only be coarser, and how much coarser is a
    statement about residual availability rather than about the plant.
    """
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    sel = [i for i, n in enumerate(rnames) if n in keep_residuals]
    if not sel:
        return {"n_classes": 1, "indiscernible_mass": 1.0,
                "blocks": [sorted(P.LOCATION_IDS)]}
    Mloc = P.fsm_over_locations(Mstruct, fnames)[sel]
    groups = {}
    for j, loc in enumerate(P.LOCATION_IDS):
        groups.setdefault(tuple(Mloc[:, j]), []).append(loc)
    blocks = sorted(groups.values(), key=lambda b: b[0])
    n = len(P.LOCATION_IDS)
    mass = sum(len(b) for b in blocks if len(b) > 1) / float(n)
    return {"n_classes": len(blocks), "indiscernible_mass": mass,
            "blocks": blocks}


def _mass(cols_emp, denom):
    """Indiscernible mass of an observed-column assignment.

    A location is indiscernible if it shares its column with another, or if
    its column is all zero: nothing fires, so nothing can be said.
    """
    groups = {}
    for loc, c in cols_emp.items():
        groups.setdefault(tuple(c), []).append(loc)
    silent = [l for l, c in cols_emp.items() if not c.any()]
    mass = (sum(len(v) for k, v in groups.items() if any(k) and len(v) > 1)
            + len(silent)) / float(denom)
    return mass, groups, silent


def empirical(bank, cols, keep_residuals, F_fault, y_fault, F_cases, truths):
    """What the verifier achieves on this instrumentation.

    Two empirical masses are reported, and the gap between them is the point.

    `empirical_mass` is what the trigger patterns literally give.  It can come
    out FINER than the bank's own structural partition, which detection power
    alone cannot do: weak detection merges classes, it never splits them.  A
    split can only come from a residual firing under a fault the structural
    model says is decoupled from it -- leakage.  Such a distinction exists in
    the data and is worthless as a diagnosis, because nothing about it is
    guaranteed: not its sign, not its persistence across operating points,
    and not that the truth stays in the admissible set it produces.

    `empirical_mass_consistent` masks each observed column by the structural
    support before partitioning, so only discriminations the structural model
    endorses are counted.  It satisfies the ordering the theory requires,
    bound <= bank <= empirical, at every rung by construction.

    Both use the same denominator as the structural quantities, the full
    location count, so the three curves share a measure and may share an axis.
    """
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    sel = [i for i, n in enumerate(rnames) if n in keep_residuals]
    Mloc = P.fsm_over_locations(Mstruct, fnames)[sel]
    sub = [rnames[i] for i in sel]
    order = [keep_residuals.index(n) for n in sub]
    loc_index = {l: i for i, l in enumerate(P.LOCATION_IDS)}

    trig_f = bank.residual_triggers(F_fault[:, cols], ALPHA)[:, order]
    cols_emp, cons_emp, det, leaks = {}, {}, {}, {}
    for loc in P.LOCATION_IDS:
        m = np.isin(y_fault, P.LOCATIONS[loc][0])
        if not m.any():
            continue
        c = (trig_f[m].mean(axis=0) > 0.5).astype(int)
        st = Mloc[:, loc_index[loc]]
        cols_emp[loc] = c
        cons_emp[loc] = (c & st).astype(int)
        det[loc] = float((trig_f[m].sum(1) > 0).mean())
        fired = [sub[i] for i in range(len(sub)) if c[i] and not st[i]]
        if fired:
            leaks[loc] = fired
    denom = float(len(P.LOCATION_IDS))
    emp_mass, groups, silent = _mass(cols_emp, denom)
    cons_mass, _, _ = _mass(cons_emp, denom)

    V = CF.Verifier(Mloc, sub, P.LOCATION_IDS)
    trig_c = bank.residual_triggers(F_cases[:, cols], ALPHA)[:, order]
    hit = amb = 0
    sizes = []
    for i, t in enumerate(truths):
        D, _ = V.candidates(trig_c[i])
        sizes.append(len(D))
        if t in D:
            hit += 1
        if len(D) > 1:
            amb += 1
    return {"n_residuals": len(keep_residuals),
            "n_locations_with_runs": len(cols_emp),
            "mean_detection": float(np.mean(list(det.values()))) if det else 0.0,
            "empirical_mass": emp_mass,
            "empirical_mass_consistent": cons_mass,
            "decoupling_violation": len(leaks) / float(len(cols_emp)),
            "leaks": {k: list(v) for k, v in leaks.items()},
            "silent": silent,
            "truth_in_admissible": hit / len(truths),
            "ambiguous_rate": amb / len(truths),
            "mean_D": float(np.mean(sizes))}


def main():
    Xfit, mfit = E2.load("nominal_fit")
    Xcal, mcal = E2.load("nominal_cal")
    Xflt, mflt = E2.load("fault")
    F_fit, _, _ = E2.features(Xfit, mfit, faulty=False)
    F_cal, _, _ = E2.features(Xcal, mcal, faulty=False)
    F_flt, y_flt, _ = E2.features(Xflt, mflt, faulty=True)
    cases, _ = CS.build_cases(4, 8, 0)
    F_cases = CS.case_features(cases)
    truths = [c["location"] for c in cases]

    all_sensors = list(tep.SENSORS.keys())
    rows = []
    print("%-38s %4s %5s %6s %6s %6s %6s %6s %6s"
          % ("instrumentation", "sens", "res", "bound", "bank", "emp",
             "emp-ok", "leak", "meanD"))
    for label, drop in E1.TEP_LADDER:
        removed = set()
        for g in drop:
            removed.update(tep.SENSOR_GROUPS[g])
        keep = [s for s in all_sensors if s not in removed]
        avail = [r for r in R.RESIDUAL_NAMES
                 if r in SM.available_residuals(keep)]
        th = theoretical(keep)
        rs = realised_structural(avail)
        if not avail:
            rows.append({"label": label, "n_sensors": len(keep),
                         "theoretical": th, "realised_structural": rs,
                         "empirical": None})
            print("%-38s %4d %5d %6.3f   (no residual survives)"
                  % (label[:38], len(keep), 0, th["indiscernible_mass"]))
            continue
        bank, cols = subset_bank(F_fit, F_cal, avail)
        em = empirical(bank, cols, avail, F_flt, y_flt, F_cases, truths)
        rows.append({"label": label, "dropped": drop, "n_sensors": len(keep),
                     "residuals": avail, "theoretical": th,
                     "realised_structural": rs, "empirical": em})
        print("%-38s %4d %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.2f"
              % (label[:38], len(keep), len(avail),
                 th["indiscernible_mass"], rs["indiscernible_mass"],
                 em["empirical_mass"], em["empirical_mass_consistent"],
                 em["decoupling_violation"], em["mean_D"]))
        for loc, fired in em["leaks"].items():
            print("      leak  %-20s fires %s" % (loc, ", ".join(fired)))

    with open(os.path.join(OUT, "E6_ablation.json"), "w") as fh:
        json.dump({"alpha": ALPHA, "rows": rows,
                   "n_cases": len(cases)}, fh, indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E6_ablation.json"))


if __name__ == "__main__":
    main()
