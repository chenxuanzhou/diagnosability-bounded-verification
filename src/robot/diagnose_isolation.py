"""
Why the robot residuals detect but do not isolate.

Observed: an axis-6 friction fault lights the rigid-body rows of axes 2 and
4, which the structural model says are decoupled from it, and `r_rbd_2`
fires under almost any deviation.  Two candidate causes, tested here in the
order that settles the cheaper one first.

TEST 1, fit error.  A fitted relation that explains 79 % of its target has a
21 % unexplained residue.  If a fault moves the target by less than that
residue, the relation cannot see the fault, and whatever it does at threshold
is noise rather than signal.  For each relation this compares the shift a
fault produces against the relation's own nominal spread, and separately
reports each relation's false-alarm rate on held-out nominal recordings.  A
relation whose nominal false-alarm rate is already high is simply a bad
relation and belongs out of the bank.

TEST 2, wrong fault location.  The structural model puts drivetrain friction
on the motor side, upstream of the joint torque sensor, so friction appears
in the motor-side relation and NOT in the rigid-body row.  If the injected
friction acts downstream of the sensor instead -- in the bearing rather than
the gearbox -- then it genuinely does enter the rigid-body row, and the
model's placement of it is wrong.  That variant is built here and its
predicted signatures are scored against what actually fires.

If TEST 2 wins, this is not a failure of the method.  It is the method
detecting an error in a commercial arm's structural model by signature
mismatch, which is the same mechanism Gate B used on the Tennessee Eastman
model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.robot import run_robot as RUN                      # noqa: E402
from src.robot import voraus_data as VD                     # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import residuals_robot as RR              # noqa: E402

OUT = os.path.join("results", "robot")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.01


def support_downstream():
    """Fault supports if friction acts downstream of the torque sensor.

    The joint torque sensor sits between the gearbox and the link.  Friction
    upstream of it (gearbox, motor bearings) never enters the rigid-body
    balance, because the sensor already reads the torque delivered to the
    link.  Friction downstream of it (link bearing) does: the link sees the
    measured torque minus that friction.  So the single edit is to add
    friction_i to the rigid-body row of axis i, and nowhere else -- friction
    is a joint-local torque and cannot reach another axis's row.
    """
    sup = {k: list(v) for k, v in RR.FAULT_SUPPORT.items()}
    for i in RR.AX:
        sup["r_rbd_%d" % i] = sup["r_rbd_%d" % i] + ["friction%d" % i]
    return sup


def support_only_downstream():
    """Friction ONLY downstream: it leaves the motor-side relation entirely.

    If the injected friction sits in the link bearing rather than in the
    gearbox, the motor-side balance never sees it -- the torque sensor
    already reads what was delivered -- and only the rigid-body row does.
    This variant is what makes the friction and commutation faults
    structurally separable, so it is a stronger claim than the "both" variant
    and is tested separately rather than assumed.
    """
    sup = {k: list(v) for k, v in RR.FAULT_SUPPORT.items()}
    for i in RR.AX:
        sup["r_mdyn_%d" % i] = [x for x in sup["r_mdyn_%d" % i]
                                if x != "friction%d" % i]
        sup["r_rbd_%d" % i] = sup["r_rbd_%d" % i] + ["friction%d" % i]
    return sup


def fsm_from_support(support, props):
    M = np.zeros((len(RR.RESIDUAL_NAMES), len(props)), dtype=int)
    for r, rn in enumerate(RR.RESIDUAL_NAMES):
        s = set(support[rn])
        for j, p in enumerate(props):
            if p == "extforce":
                if any(x.startswith("extforce") for x in s):
                    M[r, j] = 1
            elif p in s:
                M[r, j] = 1
    return M


def agreement(profiles, M, props):
    """How well a signature matrix explains what actually fired.

    Per location: of the residuals that fire in the majority of its windows,
    how many the signature says are coupled to it (precision), and of the
    residuals the signature says are coupled, how many actually fire
    (recall).  Reported as the mean over locations, plus the count of firing
    residuals the signature cannot explain.
    """
    prec, rec, unexplained = [], [], 0
    for j, p in enumerate(props):
        if p not in profiles:
            continue
        obs = (np.asarray(profiles[p]) > 0.5).astype(int)
        pred = M[:, j]
        if obs.sum():
            prec.append(float((obs & pred).sum() / obs.sum()))
            unexplained += int((obs & (1 - pred)).sum())
        if pred.sum():
            rec.append(float((obs & pred).sum() / pred.sum()))
    return {"precision": float(np.mean(prec)) if prec else float("nan"),
            "recall": float(np.mean(rec)) if rec else float("nan"),
            "unexplained_firings": unexplained}


def main():
    arrs, meta, cols = VD.load()
    props_all = [RUN.proposition(m) for m in meta]
    keep = [i for i, p in enumerate(props_all) if p is not None]
    norm = [i for i in keep if props_all[i] == "NF"]
    flt = [i for i in keep if props_all[i] != "NF"]
    rng = np.random.default_rng(0)
    norm = list(rng.permutation(norm))
    fit_i, cal_i, tst_i = norm[:300], norm[300:800], norm[800:]

    model = RR.RobotResiduals(cols).fit([arrs[i] for i in fit_i])
    print("relations fitted")

    def feats(idx):
        return np.stack([model.window_features(arrs[i]) for i in idx])

    Ffit, Fcal, Ftst = feats(fit_i), feats(cal_i), feats(tst_i)
    Fflt = feats(flt)
    yflt = [props_all[i] for i in flt]
    bank = CF.ConformalBank(names=RR.RESIDUAL_NAMES)
    bank.disp_mask = np.ones(len(RR.RESIDUAL_NAMES), dtype=bool)
    bank.fit_scale(Ffit).calibrate(Fcal)

    # ---------------------------------------------------------- TEST 1
    thr = bank.threshold(ALPHA)
    fire_nom = np.abs(bank.z(Ftst)) > thr
    m = len(RR.RESIDUAL_NAMES)
    per_res_fa = ((fire_nom[:, :m] | fire_nom[:, m:]).mean(axis=0))
    zf, zn = bank.z(Fflt), bank.z(Ftst)
    print("")
    print("TEST 1  relation quality on held-out NOMINAL recordings")
    print("  %-12s %10s %12s" % ("relation", "false alarm", "nominal |z| p95"))
    bad = []
    for k, name in enumerate(RR.RESIDUAL_NAMES):
        p95 = float(np.percentile(np.abs(zn[:, [k, m + k]]), 95))
        if per_res_fa[k] > 0.02 or name.startswith("r_rbd"):
            print("  %-12s %10.3f %12.2f" % (name, per_res_fa[k], p95))
        if per_res_fa[k] > 0.05:
            bad.append(name)
    print("  relations with a nominal false-alarm rate above 5 %%: %s"
          % (bad or "none"))

    # effect size: how far a fault moves each relation, in units of the
    # threshold it has to cross
    print("")
    print("TEST 1  effect size of each fault on its OWN motor-side relation")
    print("  %-14s %10s %10s" % ("fault", "own mdyn", "own rbd"))
    eff = {}
    for i in RR.AX:
        for fam in ("friction", "commutation"):
            p = "%s%d" % (fam, i)
            sel = np.array([y == p for y in yflt])
            if not sel.any():
                continue
            km = RR.RESIDUAL_NAMES.index("r_mdyn_%d" % i)
            kr = RR.RESIDUAL_NAMES.index("r_rbd_%d" % i)
            a = float(np.median(np.abs(zf[sel][:, [km, m + km]]).max(1)))
            b = float(np.median(np.abs(zf[sel][:, [kr, m + kr]]).max(1)))
            eff[p] = {"own_mdyn_over_threshold": a / thr,
                      "own_rbd_over_threshold": b / thr}
            print("  %-14s %9.2fx %9.2fx" % (p, a / thr, b / thr))

    # ---------------------------------------------------------- TEST 2
    props = sorted({p for p in props_all if p not in (None, "NF")})
    trig = bank.residual_triggers(Fflt, ALPHA)
    profiles = {}
    for p in props:
        sel = np.array([y == p for y in yflt])
        if sel.any():
            profiles[p] = trig[sel].mean(0).tolist()

    variants = {
        "friction upstream (as modelled)": RR.FAULT_SUPPORT,
        "friction on both sides": support_downstream(),
        "friction only downstream": support_only_downstream(),
    }
    mats = {k: fsm_from_support(v, props) for k, v in variants.items()}
    agrees = {k: agreement(profiles, v, props) for k, v in mats.items()}
    M_up, M_dn = mats["friction upstream (as modelled)"],         mats["friction on both sides"]
    a_up, a_dn = agrees["friction upstream (as modelled)"],         agrees["friction on both sides"]
    print("")
    print("TEST 2  does moving friction downstream of the torque sensor "
          "explain the firings better?")
    print("  %-34s %9s %9s %14s"
          % ("structural model", "precision", "recall", "unexplained"))
    for nm, a in agrees.items():
        print("  %-34s %9.3f %9.3f %14d"
              % (nm, a["precision"], a["recall"], a["unexplained_firings"]))

    # the same restricted to the friction faults, where the two differ
    fr = [p for p in props if p.startswith("friction")]
    sub = {k: v for k, v in profiles.items() if k in fr}
    agr_f = {k: agreement(sub, v, props) for k, v in mats.items()}
    print("  restricted to the friction faults, where the variants differ:")
    for nm, a in agr_f.items():
        print("    %-32s precision %.3f recall %.3f unexplained %d"
              % (nm, a["precision"], a["recall"],
                 a["unexplained_firings"]))
    # the commutation faults are the control: no variant changes them, so if
    # they are already explained the motor-side placement is right for them
    cm = [p for p in props if p.startswith("commutation")]
    a_cm = agreement({k: v for k, v in profiles.items() if k in cm},
                     M_up, props)
    print("  commutation faults under the unchanged model (control): "
          "precision %.3f recall %.3f" % (a_cm["precision"], a_cm["recall"]))
    a_up_f = agr_f["friction upstream (as modelled)"]
    a_dn_f = agr_f["friction on both sides"]

    json.dump({"alpha": ALPHA, "threshold": thr,
               "per_residual_false_alarm":
                   dict(zip(RR.RESIDUAL_NAMES, per_res_fa.tolist())),
               "high_false_alarm_relations": bad,
               "effect_sizes": eff,
               "profiles": profiles,
               "agreement_all_faults": agrees,
               "agreement_friction_faults": agr_f,
               "agreement_commutation_control": a_cm},
              open(os.path.join(OUT, "isolation_diagnosis.json"), "w"),
              indent=2)
    print("")
    print("wrote %s" % os.path.join(OUT, "isolation_diagnosis.json"))


if __name__ == "__main__":
    main()
