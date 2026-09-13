"""
E5: bound tightness.

R1 says the structural quotient upper-bounds what any verifier can
distinguish.  An implemented verifier can only be coarser, and the shortfall
is a signal-to-noise effect: a deviation too small to push a residual past
its calibrated threshold leaves no trace, so its location becomes
indistinguishable from every other location that also leaves no trace, and
from no-fault.

Two quantities are reported at each operating point.

  merge gap            how much coarser the verifier is than the bound,
      measured as the extra fraction of fault locations that are silent and
      therefore merged with no-fault.  This is non-negative by
      construction and should close as deviations grow.  It is the
      quantitative content of "the bound is tight in the high-SNR limit".

  spurious separation  how often two locations the structure proves to be
      indiscernible nevertheless produce different firing patterns.  Under a
      correct model this must be zero: a residual generator decoupled from a
      fault cannot respond to it.  Any non-zero value is model mismatch
      leaking into the residuals, not information, so it is reported as a
      defect rather than as performance.

The noise sweep recalibrates the conformal thresholds at each noise level,
because an operator calibrating a plant would calibrate against that plant's
own nominal data.  Keeping thresholds from a quieter plant would measure
mis-calibration rather than signal-to-noise.
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
    """Recalibrate on nominal runs generated at this same noise level.

    An operator calibrating a plant calibrates against that plant's own
    nominal data, so the thresholds must move with the noise floor.  Split
    conformal at alpha = 0.01 needs at least 99 calibration windows after
    the fit/calibration split, which is why a dedicated block of nominal
    runs was generated at each noise level.
    """
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
    sub, submeta = X[idx], [meta[i] for i in idx]
    F, _, _ = E2.features(sub, submeta, faulty=False)
    half = len(F) // 2
    if half < 100:
        return fallback, False
    return (CF.ConformalBank().fit_scale(F[:half]).calibrate(F[half:]),
            True)


def analyse(trig, labels, Mloc, struct_blocks):
    """Empirical signatures per location, and the two reported quantities."""
    cols = {}
    det = {}
    for loc in P.LOCATION_IDS:
        sel = np.isin(labels, P.LOCATIONS[loc][0])
        if not sel.any():
            continue
        cols[loc] = (trig[sel].mean(axis=0) > 0.5).astype(int)
        det[loc] = float((trig[sel].sum(1) > 0).mean())
    seen = list(cols)
    silent = [l for l in seen if not cols[l].any()]

    # merge gap: silent locations collapse onto no-fault and onto each other
    merge_gap = len(silent) / float(len(seen)) if seen else float("nan")

    # Decoupling violations.  Two locations in the same structural class
    # share a SENSITIVITY set, not a firing pattern: how many of the
    # residuals they can affect actually cross the threshold depends on how
    # severe each deviation is, and the benchmark's two members of a class
    # are not equally severe.  So comparing their firing patterns directly
    # would score a severity difference as a theory violation.
    #
    # What the theory does forbid is a residual firing for a fault the
    # structure decouples from it.  That is what is counted here: for each
    # location, whether its majority pattern lights a residual outside its
    # own structural column.
    pairs = tot = 0
    loc_index = {l: i for i, l in enumerate(P.LOCATION_IDS)}
    for loc, col in cols.items():
        tot += 1
        struct_col = Mloc[:, loc_index[loc]]
        if np.any(col & (1 - struct_col)):
            pairs += 1
    return {"n_locations_seen": len(seen), "silent": silent,
            "merge_gap": merge_gap,
            "mean_detection": float(np.mean(list(det.values()))) if det else 0.0,
            "detection": det,
            "decoupling_violation": (pairs / tot) if tot else float("nan"),
            "locations_checked": tot}


def main():
    fallback = base_bank()
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]

    struct_blocks = P.structural_classes()
    n_loc = len(P.LOCATION_IDS)
    theo = sum(len(b) for b in struct_blocks if len(b) > 1) / float(n_loc)
    print("structural bound: %d classes over %d locations, "
          "indiscernible mass %.3f" % (len(struct_blocks), n_loc, theo))
    print("  non-singleton classes: %s"
          % [b for b in struct_blocks if len(b) > 1])
    print("  under a correct model no residual may fire for a fault the "
          "structure decouples it from, at any operating point")

    out = {"theoretical_mass": theo, "structural_classes": struct_blocks,
           "sweeps": {}}

    for tag, knob in (("sweep_mag", "mag"), ("sweep_stick", "stiction"),
                      ("sweep_noise", "noise")):
        X = np.load(os.path.join(DATA, "%s_X.npy" % tag))
        meta = json.load(open(os.path.join(DATA, "%s_meta.json" % tag)))
        if tag == "sweep_mag":
            # extended magnitudes for the deviations that stay below the
            # detection floor at benchmark magnitude
            Xb = np.load(os.path.join(DATA, "sweep_big_X.npy"))
            mb = json.load(open(os.path.join(DATA, "sweep_big_meta.json")))
            X = np.concatenate([X, Xb], axis=0)
            meta = meta + mb
        levels = sorted({level_of(m, knob) for m in meta
                         if m["fault"] in C.ACTIVE_IDV})
        title = {"mag": "fault magnitude sweep",
                 "stiction": "valve stiction deadband sweep",
                 "noise": "measurement noise sweep"}[knob]
        print("")
        print(title)
        print("  %-8s %7s %10s %11s %11s %s"
              % ("level", "windows", "detection", "merge gap",
                 "decoupling", "silent locations"))
        rows = []
        for lv in levels:
            idx = [i for i, m in enumerate(meta)
                   if m["fault"] in C.ACTIVE_IDV and level_of(m, knob) == lv]
            if not idx:
                continue
            sub, submeta = X[idx], [meta[i] for i in idx]
            F, y, _ = E2.features(sub, submeta, faulty=True)
            if len(F) == 0:
                continue
            bank, recal = (bank_for_noise(lv, fallback)
                           if knob == "noise" else (fallback, False))
            trig = bank.residual_triggers(F, ALPHA)[:, order]
            r = analyse(trig, y, Mloc, struct_blocks)
            r.update({"level": float(lv), "n_windows": int(len(F)),
                      "recalibrated": bool(recal)})
            r["full_panel"] = r["n_locations_seen"] >= len(P.LOCATION_IDS) - 1
            rows.append(r)
            print("  %-8.3f %7d %10.3f %11.3f %11.3f  %s"
                  % (lv, len(F), r["mean_detection"], r["merge_gap"],
                     r["decoupling_violation"],
                     ", ".join(s.replace("L_", "") for s in r["silent"])
                     or "none"))
        out["sweeps"][tag] = rows
        if knob == "stiction":
            weak = ["L_reac_cw_valve", "L_cond_cw_valve", "L_other_valves"]
            print("")
            print("  detection rate of the valve deviations vs deadband")
            print("  %-8s %s" % ("level", "".join("%18s" % w.replace("L_", "")
                                                  for w in weak)))
            curve = []
            for r in rows:
                vals = [r["detection"].get(w) for w in weak]
                curve.append({"level": r["level"],
                              "detection": {w: v for w, v in zip(weak, vals)}})
                print("  %-8.3f %s"
                      % (r["level"],
                         "".join("%18s" % ("%.3f" % v if v is not None
                                           else "-") for v in vals)))
            out["stiction_detection_curve"] = curve
        if knob == "mag":
            # The extended levels carry only the four weak deviations, so a
            # sweep-wide mass is not comparable there.  What IS comparable is
            # each weak location's own detection rate against its magnitude:
            # that is the direct evidence that a location leaves the silent
            # set once its deviation is large enough, which is what closes
            # the merge gap.
            weak = ["L_Dfeed_temp", "L_Cfeed_temp", "L_cond_cw_valve"]
            print("")
            print("  detection rate of the deviations that start silent")
            print("  %-8s %s" % ("level", "".join("%18s" % w.replace("L_", "")
                                                  for w in weak)))
            curve = []
            for r in rows:
                vals = [r["detection"].get(w) for w in weak]
                curve.append({"level": r["level"],
                              "detection": {w: v for w, v in zip(weak, vals)}})
                print("  %-8.3f %s"
                      % (r["level"],
                         "".join("%18s" % ("%.3f" % v if v is not None
                                           else "-") for v in vals)))
            out["weak_detection_curve"] = curve

    with open(os.path.join(OUT, "E5_bound_tightness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E5_bound_tightness.json"))


if __name__ == "__main__":
    main()
