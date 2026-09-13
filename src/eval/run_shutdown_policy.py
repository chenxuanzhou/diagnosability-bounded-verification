"""
How runs that tripped the plant are handled, and whether it matters.

Eighteen of the 160 benchmark-magnitude fault runs trip the plant: all eight
IDV(6) runs, plus one IDV(12) and one IDV(13).  After a trip the integrator
freezes and the data stops being a plant trajectory, so those samples cannot
be treated like any other.

The policy in force everywhere in this repository, implemented once in
`run_E2._usable_end`:

  a tripped run is truncated two samples before the trip, and any run left
  with no complete window after truncation is dropped from that analysis.

Truncating rather than discarding keeps the pre-trip behaviour, which is real
plant data and often the most informative part of the run.  Dropping the
remainder avoids mixing frozen samples into a window.

A policy choice that is not shown to be harmless is a place for a result to
hide, so this compares it against the two alternatives a reader would
suggest: discarding every tripped run outright, and keeping them untruncated.
If the headline numbers move under those, the choice matters and has to be
argued; if they do not, it does not.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "sensitivity")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA

POLICIES = {
    "truncate (in force)": "truncate",
    "discard tripped runs": "discard",
    "keep untruncated": "keep",
}


def features_under(X, meta, policy):
    """Fault-window features under one trip policy."""
    T = X.shape[1]
    feats, labels = [], []
    for i, m in enumerate(meta):
        tripped = bool(m.get("shutdown"))
        if policy == "discard" and tripped:
            continue
        if policy == "keep":
            end = T
        else:
            end = E2._usable_end(m, T)
        use = list(range(E2.IDV_ON + E2.SETTLE, end - E2.WIDTH + 1, 60))
        if not use:
            continue
        feats.append(CF.window_matrix(X[i], use, E2.WIDTH))
        labels += [m["fault"]] * len(use)
    if not feats:
        return np.zeros((0, 2 * len(CF.R.RESIDUAL_NAMES))), np.array([])
    return np.vstack(feats), np.array(labels)


def main():
    Xfit, mfit = E2.load("nominal_fit")
    Xcal, mcal = E2.load("nominal_cal")
    Xtst, mtst = E2.load("nominal_test")
    Xflt, mflt = E2.load("fault")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    Ftst, _, _ = E2.features(Xtst, mtst, faulty=False)
    bank = CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]

    tripped = sum(1 for m in mflt if m["shutdown"])
    from collections import Counter
    which = Counter(m["fault"] for m in mflt if m["shutdown"])
    print("fault runs: %d, of which tripped: %d" % (len(mflt), tripped))
    print("tripped runs by disturbance: %s" % dict(sorted(which.items())))
    print("nominal runs that tripped: %d"
          % sum(1 for m in mfit + mcal + mtst if m["shutdown"]))

    rows = []
    print("")
    print("%-22s %8s %10s %12s %12s"
          % ("policy", "windows", "detection", "merge gap", "decoupling"))
    for label, pol in POLICIES.items():
        F, y = features_under(Xflt, mflt, pol)
        if len(F) == 0:
            continue
        trig = bank.residual_triggers(F, ALPHA)[:, order]
        cols, det = {}, {}
        for loc in P.LOCATION_IDS:
            sel = np.isin(y, P.LOCATIONS[loc][0])
            if not sel.any():
                continue
            cols[loc] = (trig[sel].mean(axis=0) > 0.5).astype(int)
            det[loc] = float((trig[sel].sum(1) > 0).mean())
        loc_index = {l: i for i, l in enumerate(P.LOCATION_IDS)}
        silent = [l for l in cols if not cols[l].any()]
        viol = sum(1 for l in cols
                   if np.any(cols[l] & (1 - Mloc[:, loc_index[l]])))
        row = {"policy": label, "n_windows": int(len(F)),
               "n_locations": len(cols),
               "mean_detection": float(np.mean(list(det.values()))),
               "merge_gap": len(silent) / len(cols),
               "decoupling_violation": viol / len(cols),
               "silent": silent, "detection": det}
        rows.append(row)
        print("%-22s %8d %10.3f %12.3f %12.3f"
              % (label, len(F), row["mean_detection"], row["merge_gap"],
                 row["decoupling_violation"]))

    base = rows[0]
    print("")
    print("largest shift in any per-location detection rate, against the "
          "policy in force:")
    for r in rows[1:]:
        d = max(abs(r["detection"].get(k, 0.0) - v)
                for k, v in base["detection"].items())
        loc = max(base["detection"],
                  key=lambda k: abs(r["detection"].get(k, 0.0)
                                    - base["detection"][k]))
        print("  %-22s %.3f  (at %s)" % (r["policy"], d, loc))

    json.dump({"n_fault_runs": len(mflt), "n_tripped": tripped,
               "tripped_by_disturbance": {str(k): v
                                          for k, v in which.items()},
               "policy_in_force": "truncate two samples before the trip; "
                                  "drop runs with no complete window left",
               "rows": rows},
              open(os.path.join(OUT, "shutdown_policy.json"), "w"), indent=2)
    print("")
    print("wrote %s" % os.path.join(OUT, "shutdown_policy.json"))


if __name__ == "__main__":
    main()
