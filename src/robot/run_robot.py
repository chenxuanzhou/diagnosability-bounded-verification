"""
The voraus-AD arm: the same construction, on real recorded machine data.

The plan gives this system the reality-check role.  The Tennessee Eastman
model is where the bound is verified quantitatively, because only there can
fault magnitude, noise and instrumentation be swept.  What this arm has to
show is that the construction survives contact with a real machine: real
noise, real unmodelled dynamics, real anomalies staged by an operator rather
than injected into an equation.

Labels.  The dataset records an anomaly CATEGORY and a VARIANT.  For the
three categories that were staged axis by axis -- axis friction, axis weight
and motor commutation -- the variant identifies the axis, so the proposition
can be resolved to the axis the structural model talks about.  The collision
and workpiece categories are not axis-resolved and map to the external-torque
and payload propositions respectively.  Invalid start pose and wobbling
station are excluded, as recorded in the Gate A mapping.

Residual relations are fitted on nominal recordings only, on structurally
admissible inputs, and calibrated conformally on a disjoint block of nominal
recordings.  No anomalous recording is used for fitting, scaling,
thresholding or model selection.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.robot import robot_structural as RS                # noqa: E402
from src.robot import voraus_data as VD                     # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import residuals_robot as RR              # noqa: E402

OUT = os.path.join("results", "robot")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.01

# variant id -> axis, from voraus_ad.Variant in the official repository
AXIS_OF_VARIANT = {}
for _v, _a in zip(range(0, 14), [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6]):
    AXIS_OF_VARIANT[_v] = _a                      # AXIS_FRICTION
for _v, _a in zip(range(14, 32),
                  [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]):
    AXIS_OF_VARIANT[_v] = _a                      # AXIS_WEIGHT
for _v, _a in zip(range(62, 69), [1, 1, 2, 3, 4, 5, 6]):
    AXIS_OF_VARIANT[_v] = _a                      # MOTOR_COMMUTATION

EXCLUDED = {"INVALID_POSITION", "WOBBLING_STATION"}


def proposition(m):
    """Structural fault mode of one recording, or None if excluded."""
    c = m["category_name"]
    if not m["anomaly"] or c == "NORMAL_OPERATION":
        return "NF"
    if c in EXCLUDED:
        return None
    if c in ("AXIS_FRICTION", "AXIS_WEIGHT", "MOTOR_COMMUTATION"):
        ax = AXIS_OF_VARIANT.get(m["setting"])
        if ax is None:
            return None
        return {"AXIS_FRICTION": "friction%d",
                "AXIS_WEIGHT": "addedmass%d",
                "MOTOR_COMMUTATION": "commutation%d"}[c] % ax
    if c in ("COLLISION_FOAM", "COLLISION_CABLE", "COLLISION_CARTON",
             "ENTANGLED"):
        return "extforce"
    if c in ("MISS_CAN", "LOSE_CAN", "CAN_WEIGHT"):
        return "payload"
    return None


def build_fsm(props):
    """Residual by proposition incidence, from the structural supports.

    The collision categories are not axis-resolved, so the external-torque
    proposition is the union over axes: a collision fires whichever
    rigid-body row carries the contact.
    """
    M = np.zeros((len(RR.RESIDUAL_NAMES), len(props)), dtype=int)
    for r, rn in enumerate(RR.RESIDUAL_NAMES):
        sup = set(RR.FAULT_SUPPORT[rn])
        for j, p in enumerate(props):
            if p == "extforce":
                if any(s.startswith("extforce") for s in sup):
                    M[r, j] = 1
            elif p in sup:
                M[r, j] = 1
    return M


def main():
    arrs, meta, cols = VD.load()
    props = [proposition(m) for m in meta]
    keep = [i for i, p in enumerate(props) if p is not None]
    print("recordings usable: %d of %d (excluded categories dropped)"
          % (len(keep), len(meta)))

    norm = [i for i in keep if props[i] == "NF"]
    flt = [i for i in keep if props[i] != "NF"]
    rng = np.random.default_rng(0)
    norm = list(rng.permutation(norm))
    n_fit, n_cal = 300, 500
    fit_i = norm[:n_fit]
    cal_i = norm[n_fit:n_fit + n_cal]
    tst_i = norm[n_fit + n_cal:]
    print("nominal split: fit %d, calibrate %d, test %d ; anomalous %d"
          % (len(fit_i), len(cal_i), len(tst_i), len(flt)))

    model = RR.RobotResiduals(cols).fit([arrs[i] for i in fit_i])
    print("relations fitted")

    def feats(idx):
        return np.stack([model.window_features(arrs[i]) for i in idx])

    Ffit, Fcal, Ftst = feats(fit_i), feats(cal_i), feats(tst_i)
    Fflt = feats(flt)
    yflt = [props[i] for i in flt]
    print("features computed")

    bank = CF.ConformalBank(names=RR.RESIDUAL_NAMES)
    bank.disp_mask = np.ones(len(RR.RESIDUAL_NAMES), dtype=bool)
    bank.fit_scale(Ffit).calibrate(Fcal)
    cov = bank.coverage(Ftst, [0.20, 0.10, 0.05, 0.02, 0.01, 0.005])
    print("\ncalibration curve on held-out nominal recordings")
    for r in cov:
        print("  alpha %5.3f  nominal %.3f  empirical %.3f"
              % (r["alpha"], r["nominal_coverage"], r["empirical_coverage"]))

    prop_ids = sorted({p for p in props if p not in (None, "NF")})
    M = build_fsm(prop_ids)
    trig = bank.residual_triggers(Fflt, ALPHA)
    trig_n = bank.residual_triggers(Ftst, ALPHA)
    print("\nfalse alarm on held-out nominal recordings: %.3f"
          % float((trig_n.sum(1) > 0).mean()))

    V = CF.Verifier(M, RR.RESIDUAL_NAMES, prop_ids)
    rows = []
    print("\nper proposition")
    print("  %-16s %5s %7s %8s %8s  %s"
          % ("proposition", "n", "detect", "truth in D", "mean|D|", "top residuals"))
    for p in prop_ids:
        sel = np.array([y == p for y in yflt])
        if not sel.any():
            continue
        prof = trig[sel].mean(0)
        det = float((trig[sel].sum(1) > 0).mean())
        hits, sizes = [], []
        for t in trig[sel]:
            D, _ = V.candidates(t)
            hits.append(p in D)
            sizes.append(len(D))
        top = ", ".join("%s %.2f" % (RR.RESIDUAL_NAMES[k].replace("r_", ""),
                                     prof[k])
                        for k in np.argsort(-prof)[:3] if prof[k] > 0.1)
        rows.append({"proposition": p, "n": int(sel.sum()),
                     "detection": det, "truth_in_D": float(np.mean(hits)),
                     "mean_D": float(np.mean(sizes)),
                     "profile": prof.tolist()})
        print("  %-16s %5d %7.2f %8.2f %8.2f  %s"
              % (p, sel.sum(), det, np.mean(hits), np.mean(sizes), top))

    json.dump({"alpha": ALPHA, "coverage": cov,
               "false_alarm_nominal": float((trig_n.sum(1) > 0).mean()),
               "propositions": prop_ids, "rows": rows,
               "residuals": RR.RESIDUAL_NAMES,
               "split": {"fit": len(fit_i), "cal": len(cal_i),
                         "test": len(tst_i), "fault": len(flt)}},
              open(os.path.join(OUT, "robot_results.json"), "w"), indent=2)
    print("\nwrote %s" % os.path.join(OUT, "robot_results.json"))


if __name__ == "__main__":
    main()
