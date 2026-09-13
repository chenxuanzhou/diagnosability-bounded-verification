"""
E9: what happens when the structural model is wrong.

The whole construction rests on a model of the plant, and the obvious
objection is that the model is never right. This measures how the guarantee
degrades under three kinds of being wrong, chosen because they fail in
different ways:

  parameter error      every physical constant the residuals use is
      perturbed by a relative error.  The relations still have the right
      form and the right couplings, they are just quantitatively off.  This
      should cost calibration first: the residuals acquire a bias, the
      nominal distribution shifts, and the false-alarm rate rises above the
      level the conformal layer promised.

  unmodelled dynamics  the inventory and stored-energy accumulation terms
      are removed, so every vessel balance reverts to a steady-state
      relation on a plant that is not at steady state.  This is the classic
      omission and it should cost decoupling, not just calibration.

  wrong coupling       the condenser heat-transfer coefficient is assumed
      independent of the process flow, which is precisely the error the
      first version of this model made and which Gate B caught.  The
      residuals still work; the FAULT SIGNATURE is wrong, so the verifier
      confidently rules out the truth.

The third is the dangerous one and the point of separating them: a
calibration failure shows up as noise, while a signature failure shows up as
a confident wrong answer.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import cases as CS                            # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.tep import tep_constants as C                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import residuals_tep as R                 # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "E9")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA

# every constant the residuals read, by name in tep_constants
PERTURBABLE = ["XMW", "AVP", "BVP", "CVP", "AD", "BD", "CD", "AH", "BH",
               "CH", "AG", "BG", "CG", "AV", "HWR", "HWS", "VTR", "VTS",
               "VTC", "VTV", "CPFLMX", "X4A_NOM", "X4B_NOM", "X4C_NOM",
               "TST1_NOM", "TST2_NOM", "TST3_NOM", "TST4_NOM", "TCWR_NOM",
               "TCWS_NOM", "TCV_NOM", "HTR1", "HTR2", "RG"]


class Perturbation:
    """Temporarily corrupt the verifier's constants, never the plant's."""

    def __init__(self, eps, seed=0):
        self.eps, self.seed = eps, seed
        self.saved = {}

    def __enter__(self):
        rng = np.random.default_rng(self.seed)
        for name in PERTURBABLE:
            v = getattr(C, name)
            self.saved[name] = copy.deepcopy(v)
            if isinstance(v, np.ndarray):
                f = 1.0 + self.eps * rng.standard_normal(v.shape)
                setattr(C, name, v * f)
            else:
                setattr(C, name, v * (1.0 + self.eps * rng.standard_normal()))
        # VRNG is a dict of valve ranges
        self.saved["VRNG"] = dict(C.VRNG)
        C.VRNG = {k: v * (1.0 + self.eps * rng.standard_normal())
                  for k, v in C.VRNG.items()}
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            setattr(C, k, v)


def evaluate(tag, Mloc_used, Ffit, Fcal, Ftst, Fflt, yflt, Fcase, truths,
             stale_bank=None):
    """Coverage, detection, decoupling and diagnosis under one condition.

    stale_bank models drift: the verifier was calibrated on the correct
    model and the constants moved afterwards, so the calibration is never
    refreshed.  Without it the calibration is redone under the perturbed
    constants, which is the case where a static bias is simply absorbed.
    """
    bank = stale_bank or CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)
    rnames = [n for n in R.RESIDUAL_NAMES]
    cov = bank.coverage(Ftst, [ALPHA])[0]
    trig = bank.residual_triggers(Fflt, ALPHA)
    det, viol = {}, 0
    loc_index = {l: i for i, l in enumerate(P.LOCATION_IDS)}
    seen = 0
    for loc in P.LOCATION_IDS:
        sel = np.isin(yflt, P.LOCATIONS[loc][0])
        if not sel.any():
            continue
        seen += 1
        det[loc] = float((trig[sel].sum(1) > 0).mean())
        col = (trig[sel].mean(axis=0) > 0.5).astype(int)
        if np.any(col & (1 - Mloc_used[:, loc_index[loc]])):
            viol += 1
    V = CF.Verifier(Mloc_used, rnames, P.LOCATION_IDS)
    tc = bank.residual_triggers(Fcase, ALPHA)
    hit = sum(1 for i, t in enumerate(truths)
              if t in V.candidates(tc[i])[0])
    sizes = [len(V.candidates(tc[i])[0]) for i in range(len(truths))]
    return {"condition": tag,
            "empirical_coverage": cov["empirical_coverage"],
            "coverage_shortfall": (1.0 - ALPHA) - cov["empirical_coverage"],
            "mean_detection": float(np.mean(list(det.values()))),
            "decoupling_violation": viol / max(seen, 1),
            "truth_in_admissible": hit / len(truths),
            "mean_D": float(np.mean(sizes))}


def features(no_accum=False):
    """Window features for every split, optionally without accumulation."""
    R.NO_ACCUMULATION = no_accum
    Xfit, mfit = E2.load("nominal_fit")
    Xcal, mcal = E2.load("nominal_cal")
    Xtst, mtst = E2.load("nominal_test")
    Xflt, mflt = E2.load("fault")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    Ftst, _, _ = E2.features(Xtst, mtst, faulty=False)
    Fflt, yflt, _ = E2.features(Xflt, mflt, faulty=True)
    cases, _ = CS.build_cases(4, 8, 0)
    Fcase = CS.case_features(cases)
    R.NO_ACCUMULATION = False
    return Ffit, Fcal, Ftst, Fflt, yflt, Fcase, [c["location"] for c in cases]


def main():
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    idx = [R.RESIDUAL_NAMES.index(n) for n in rnames]
    inv = np.argsort(idx)
    Mfull = np.zeros((len(R.RESIDUAL_NAMES), Mloc.shape[1]), dtype=int)
    Mfull[idx] = Mloc

    rows = []
    base = features()
    rows.append(evaluate("correct model", Mfull, *base))

    base_bank = CF.ConformalBank().fit_scale(base[0]).calibrate(base[1])
    for eps in (0.01, 0.02, 0.05, 0.10, 0.20):
        with Perturbation(eps, seed=11):
            f = features()
        rows.append(evaluate("parameter error %.0f%%, recalibrated"
                             % (100 * eps), Mfull, *f))
        rows.append(evaluate("parameter error %.0f%%, stale calibration"
                             % (100 * eps), Mfull, *f,
                             stale_bank=base_bank))

    f = features(no_accum=True)
    rows.append(evaluate("accumulation terms removed", Mfull, *f))

    # wrong coupling: the first model's mistake, the condenser coolant valve
    # assumed decoupled from the condenser coolant relation
    # The signature error has to be placed on a fault that is actually
    # detectable, otherwise it changes nothing: the reactor coolant valve
    # fires its residual on every window, so denying that coupling makes the
    # verifier rule the truth out rather than merely widen the set.
    Mwrong = Mfull.copy()
    j = P.LOCATION_IDS.index("L_reac_cw_valve")
    Mwrong[R.RESIDUAL_NAMES.index("r_cwR"), j] = 0
    rows.append(evaluate("wrong signature, reactor coolant valve", Mwrong,
                         *base))

    print("%-36s %9s %9s %10s %9s %6s"
          % ("condition", "coverage", "detection", "decoupling",
             "truth in D", "|D|"))
    for r in rows:
        print("%-36s %9.3f %9.3f %10.3f %9.3f %6.2f"
              % (r["condition"], r["empirical_coverage"],
                 r["mean_detection"], r["decoupling_violation"],
                 r["truth_in_admissible"], r["mean_D"]))
    json.dump({"alpha": ALPHA, "rows": rows},
              open(os.path.join(OUT, "E9_mismatch.json"), "w"), indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E9_mismatch.json"))


if __name__ == "__main__":
    main()
