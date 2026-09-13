"""
E2: residual construction, conformal calibration, and empirical validation
of the fault signature matrix.

Deliverables:
  1. the calibration curve -- empirical coverage against the nominal level,
     on held-out nominal data;
  2. the per-feature false-alarm rate at the operating level;
  3. an empirical check of the structural fault signature matrix, per fault
     location: which residuals actually fire, and whether residuals the
     structure decouples from a location stay quiet;
  4. the detection rate per location, which is what the bound-tightness
     experiment later sweeps against fault magnitude.
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
from src.verifier import residuals_tep as R                 # noqa: E402

DATA = os.path.join("data", "tep")
OUT = os.path.join("results", "E2")
os.makedirs(OUT, exist_ok=True)

WIDTH = 60                    # 3 h window
IDV_ON = 160                  # fault onset sample
SETTLE = 20                   # samples of transient skipped after onset
ALPHA = 0.01                  # operating joint false-alarm level
ALPHAS = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002]


def load(tag):
    X = np.load(os.path.join(DATA, "%s_X.npy" % tag))
    meta = json.load(open(os.path.join(DATA, "%s_meta.json" % tag)))
    return X, meta


def _usable_end(meta_row, T):
    """Last sample index that is still valid.

    A run that tripped the plant is truncated one window before the trip:
    after a shutdown the integrator freezes and the data is meaningless.
    """
    stop = meta_row.get("shutdown") or 0
    return T if not stop else max(stop // 180 - 2, 0)


def features(X, meta, faulty, width=WIDTH, stride=None):
    T = X.shape[1]
    stride = stride or (60 if faulty else 30)
    first = IDV_ON + SETTLE if faulty else 60
    feats, labels, runs = [], [], []
    for i, run in enumerate(X):
        end = _usable_end(meta[i], T)
        use = [s for s in range(first, end - width + 1, stride)]
        if not use:
            continue
        feats.append(CF.window_matrix(run, use, width))
        labels += [meta[i]["fault"]] * len(use)
        runs += [i] * len(use)
    if not feats:
        return np.zeros((0, 2 * len(R.RESIDUAL_NAMES))), np.array([]), np.array([])
    return np.vstack(feats), np.array(labels), np.array(runs)


def main():
    Xfit, mfit = load("nominal_fit")
    Xcal, mcal = load("nominal_cal")
    Xtst, mtst = load("nominal_test")
    Xflt, mflt = load("fault")

    Ffit, _, _ = features(Xfit, mfit, faulty=False)
    Fcal, _, _ = features(Xcal, mcal, faulty=False)
    Ftst, _, _ = features(Xtst, mtst, faulty=False)
    Fflt, yflt, rflt = features(Xflt, mflt, faulty=True)
    print("windows: fit %d  cal %d  test %d  fault %d"
          % (len(Ffit), len(Fcal), len(Ftst), len(Fflt)))
    present = sorted(set(yflt.tolist()))
    print("faults with usable windows: %s" % present)
    missing = [k for k in C.ACTIVE_IDV if k not in present]
    if missing:
        print("faults with no usable window (plant tripped): %s" % missing)

    bank = CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)
    cov = bank.coverage(Ftst, ALPHAS)
    print("\ncalibration curve on held-out nominal data")
    print("  alpha   nominal   empirical")
    for r in cov:
        print("  %5.3f   %7.3f   %9.3f"
              % (r["alpha"], r["nominal_coverage"], r["empirical_coverage"]))

    far = bank.per_residual_far(Ftst, ALPHA)
    worst = sorted(far.items(), key=lambda kv: -kv[1])[:6]
    print("\nhighest per-feature false-alarm rates at alpha = %.3f: %s"
          % (ALPHA, ", ".join("%s %.3f" % kv for kv in worst)))

    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    order = [R.RESIDUAL_NAMES.index(n) for n in rnames]
    trig = bank.residual_triggers(Fflt, ALPHA)[:, order]
    Mloc = P.fsm_over_locations(Mstruct, fnames)

    # ---- per location: firing profile, detection rate, decoupling leakage
    rows = []
    print("\nper location: detection rate, and whether residuals the "
          "structure decouples stay quiet")
    print("  %-18s %6s %8s %8s  %s"
          % ("location", "det", "coupled", "leak", "top residuals"))
    for j, loc in enumerate(P.LOCATION_IDS):
        idvs = [k for k in P.LOCATIONS[loc][0] if k in present]
        if not idvs:
            rows.append({"location": loc, "detection_rate": None,
                         "note": "plant tripped, no usable window"})
            print("  %-18s %6s   (plant tripped in every run)" % (loc, "-"))
            continue
        sel = np.isin(yflt, idvs)
        prof = trig[sel].mean(axis=0)
        det = float((trig[sel].sum(1) > 0).mean())
        pred = Mloc[:, j]
        leak = float(prof[pred == 0].max()) if (pred == 0).any() else 0.0
        hit = float(prof[pred == 1].max()) if (pred == 1).any() else 0.0
        top = ", ".join("%s %.2f" % (rnames[i].replace("r_", ""), prof[i])
                        for i in np.argsort(-prof)[:3] if prof[i] > 0.05)
        rows.append({"location": loc, "idvs": idvs, "n_windows": int(sel.sum()),
                     "detection_rate": det, "max_coupled": hit,
                     "max_leak": leak, "profile": prof.tolist()})
        print("  %-18s %6.2f %8.2f %8.2f  %s" % (loc, det, hit, leak, top))

    # ---- how often the strict consistency rule finds no explanation
    V = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)
    rules = [V.candidates(t)[1] for t in trig]
    sizes = [len(V.candidates(t)[0]) for t in trig]
    print("\ncandidate-set behaviour over %d fault windows" % len(trig))
    print("  strict rule succeeded: %.3f ; nearest-signature fallback: %.3f"
          % (rules.count("strict") / len(rules),
             rules.count("nearest") / len(rules)))
    print("  |D(y)| mean %.2f, median %d, max %d"
          % (float(np.mean(sizes)), int(np.median(sizes)), int(np.max(sizes))))

    payload = {"alpha": ALPHA, "width_samples": WIDTH, "settle": SETTLE,
               "n_windows": {"fit": len(Ffit), "cal": len(Fcal),
                             "test": len(Ftst), "fault": len(Fflt)},
               "coverage": cov, "per_feature_far": far,
               "residuals": rnames, "locations": P.LOCATION_IDS,
               "fsm_locations": Mloc.tolist(),
               "per_location": rows,
               "fallback_rate": rules.count("nearest") / len(rules),
               "candidate_set_sizes": {"mean": float(np.mean(sizes)),
                                       "median": float(np.median(sizes)),
                                       "max": int(np.max(sizes))},
               "faults_without_windows": missing}
    with open(os.path.join(OUT, "E2_calibration.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    np.savez(os.path.join(OUT, "bank.npz"), centre=bank.centre,
             scale=bank.scale, cal_scores=bank.cal_scores,
             names=np.array(bank.names))
    print("\nwrote %s" % os.path.join(OUT, "E2_calibration.json"))


if __name__ == "__main__":
    main()
