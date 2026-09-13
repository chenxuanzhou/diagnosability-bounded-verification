"""
E5: bound tightness.

R1 upper-bounds what any verifier built on a given sensor set can
distinguish.  An implemented verifier can only be coarser.  This measures the
shortfall against deviation severity and against measurement noise, and
reports two quantities that move in opposite directions.

  merge gap             the extra fraction of fault locations that leave no
      trace at all and are therefore merged with no-fault and with each
      other.  It shrinks as deviations grow: this is the quantitative
      content of "the bound is tight in the high signal limit".

  decoupling violation  the fraction of locations whose firing pattern lights
      a residual the structure proves is decoupled from them.  Under a
      correct model this is zero at every operating point.  It GROWS as
      deviations grow, because the leakage of a static residual bank on a
      moving plant only becomes visible once the signal is strong enough to
      push it past threshold.

Reporting both is the point.  A single monotone convergence curve would be a
less honest picture than two competing effects, and where they cross is where
this construction stops paying.

Comparability.  Extended magnitudes are generated for the full active fault
set (`sweep_full`), not only for the weak deviations, because plotting two
different fault populations on one axis reads as the method collapsing when
in fact the population changed.  The headline curve is additionally
restricted to the locations that have usable windows at EVERY level, so any
two points on it are comparable.  Runs that tripped the plant are dropped by
the same rule used everywhere else, and the attrition is reported.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import propositions as P                      # noqa: E402
from src.tep import tep_constants as C                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

DATA = os.path.join("data", "tep")
OUT = os.path.join("results", "E5")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA
SPEC = json.load(open(os.path.join(DATA, "spec.json")))
STEP = set(SPEC["step_faults"])

SWEEPS = {
    "magnitude": {"files": ["sweep_mag", "sweep_full"], "knob": "mag"},
    "stiction": {"files": ["sweep_stick"], "knob": "stiction"},
    "noise": {"files": ["sweep_noise"], "knob": "noise"},
    # the weak-deviation-only extension, kept as its own series so it is
    # never plotted against the full-population curve
    "magnitude_weak_only": {"files": ["sweep_big"], "knob": "mag"},
}


def level_of(m, knob):
    if knob == "noise":
        return m["noise"]
    if knob == "stiction":
        return m.get("stiction", 1.0)
    return m["fmag"] if m["fault"] in STEP else m["walk"]


def base_bank():
    Xfit, mfit = E2.load("nominal_fit")
    Xcal, mcal = E2.load("nominal_cal")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    return CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)


_NOISE_NOM = None


def bank_for_noise(level, fallback):
    """Recalibrate on nominal runs generated at this same noise level."""
    global _NOISE_NOM
    if _NOISE_NOM is None:
        X = np.load(os.path.join(DATA, "noise_nominal_X.npy"))
        meta = json.load(open(os.path.join(DATA,
                                           "noise_nominal_meta.json")))
        _NOISE_NOM = (X, meta)
    X, meta = _NOISE_NOM
    idx = [i for i, m in enumerate(meta) if m["noise"] == level]
    if not idx:
        return fallback, False
    F, _, _ = E2.features(X[idx], [meta[i] for i in idx], faulty=False)
    half = len(F) // 2
    if half < 100:
        return fallback, False
    return CF.ConformalBank().fit_scale(F[:half]).calibrate(F[half:]), True


def load_sweep(files):
    Xs, metas = [], []
    for f in files:
        Xs.append(np.load(os.path.join(DATA, "%s_X.npy" % f)))
        metas += json.load(open(os.path.join(DATA, "%s_meta.json" % f)))
    return np.concatenate(Xs, axis=0), metas


def profile(trig, labels):
    """Empirical signature and detection rate per location."""
    cols, det, n = {}, {}, {}
    for loc in P.LOCATION_IDS:
        sel = np.isin(labels, P.LOCATIONS[loc][0])
        if not sel.any():
            continue
        cols[loc] = (trig[sel].mean(axis=0) > 0.5).astype(int)
        det[loc] = float((trig[sel].sum(1) > 0).mean())
        n[loc] = int(sel.sum())
    return cols, det, n


def measures(cols, Mloc, population):
    """Merge gap and decoupling violation over a fixed set of locations."""
    loc_index = {l: i for i, l in enumerate(P.LOCATION_IDS)}
    pop = [l for l in population if l in cols]
    if not pop:
        return float("nan"), float("nan"), []
    silent = [l for l in pop if not cols[l].any()]
    viol = sum(1 for l in pop
               if np.any(cols[l] & (1 - Mloc[:, loc_index[l]])))
    return len(silent) / len(pop), viol / len(pop), silent


def run_sweep(spec, bank0, Mloc, order):
    X, meta = load_sweep(spec["files"])
    knob = spec["knob"]
    levels = sorted({level_of(m, knob) for m in meta
                     if m["fault"] in C.ACTIVE_IDV})
    per_level = {}
    for lv in levels:
        idx = [i for i, m in enumerate(meta)
               if m["fault"] in C.ACTIVE_IDV and level_of(m, knob) == lv]
        if not idx:
            continue
        sub, submeta = X[idx], [meta[i] for i in idx]
        F, y, _ = E2.features(sub, submeta, faulty=True)
        if len(F) == 0:
            continue
        bank, recal = (bank_for_noise(lv, bank0) if knob == "noise"
                       else (bank0, False))
        trig = bank.residual_triggers(F, ALPHA)[:, order]
        cols, det, n = profile(trig, y)
        per_level[lv] = {"cols": cols, "det": det, "n": n,
                         "n_windows": int(len(F)), "n_runs": len(idx),
                         "n_tripped": sum(1 for m in submeta
                                          if m["shutdown"]),
                         "recalibrated": bool(recal)}

    # locations usable at every level: the only ones a curve may compare
    common = set(P.LOCATION_IDS)
    for v in per_level.values():
        common &= set(v["cols"])
    common = sorted(common)

    rows = []
    for lv in sorted(per_level):
        v = per_level[lv]
        gap, viol, silent = measures(v["cols"], Mloc, common)
        gap_a, viol_a, _ = measures(v["cols"], Mloc, list(v["cols"]))
        rows.append({
            "level": float(lv), "n_windows": v["n_windows"],
            "n_runs": v["n_runs"], "n_tripped": v["n_tripped"],
            "recalibrated": v["recalibrated"],
            "n_locations_common": len(common),
            "n_locations_seen": len(v["cols"]),
            "merge_gap": gap, "decoupling_violation": viol,
            "merge_gap_all_seen": gap_a,
            "decoupling_violation_all_seen": viol_a,
            "mean_detection": (float(np.mean([v["det"][l] for l in common]))
                               if common else float("nan")),
            "silent": silent, "detection": v["det"],
            "n_per_location": v["n"]})
    return {"levels": rows, "common_population": common}


def main():
    bank0 = base_bank()
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]

    blocks = P.structural_classes()
    n_loc = len(P.LOCATION_IDS)
    theo = sum(len(b) for b in blocks if len(b) > 1) / float(n_loc)
    print("structural bound: %d classes over %d locations, "
          "indiscernible mass %.3f" % (len(blocks), n_loc, theo))
    print("  non-singleton classes: %s"
          % [b for b in blocks if len(b) > 1])

    out = {"theoretical_mass": theo, "structural_classes": blocks,
           "sweeps": {}}
    for name, spec in SWEEPS.items():
        res = run_sweep(spec, bank0, Mloc, order)
        out["sweeps"][name] = res
        print("")
        print("%s sweep, comparable over %d locations: %s"
              % (name, len(res["common_population"]),
                 ", ".join(l.replace("L_", "")
                           for l in res["common_population"])))
        print("  %-8s %7s %6s %9s %10s %11s  %s"
              % ("level", "windows", "trips", "detection", "merge gap",
                 "decoupling", "silent"))
        for r in res["levels"]:
            print("  %-8.3f %7d %6d %9.3f %10.3f %11.3f  %s"
                  % (r["level"], r["n_windows"], r["n_tripped"],
                     r["mean_detection"], r["merge_gap"],
                     r["decoupling_violation"],
                     ", ".join(s.replace("L_", "") for s in r["silent"])
                     or "none"))

    # per-location detection curves: the direct evidence that a location
    # leaves the silent set once its deviation is big enough
    weak = ["L_Dfeed_temp", "L_Cfeed_temp", "L_cond_cw_valve"]
    curves = {}
    for name in ("magnitude", "magnitude_weak_only", "stiction"):
        for r in out["sweeps"][name]["levels"]:
            for w in weak:
                if w in r["detection"]:
                    curves.setdefault(w, {})[
                        "%s@%g" % (name, r["level"])] = r["detection"][w]
    out["weak_detection_curves"] = curves
    print("")
    print("detection of the deviations that start silent")
    for w, v in curves.items():
        print("  %-16s %s" % (w.replace("L_", ""),
                              ", ".join("%s %.2f" % kv
                                        for kv in sorted(v.items()))))

    json.dump(out, open(os.path.join(OUT, "E5_bound_tightness.json"), "w"),
              indent=2)
    print("")
    print("wrote %s" % os.path.join(OUT, "E5_bound_tightness.json"))


if __name__ == "__main__":
    main()
