"""
Fault signature matrix of the implemented residual bank.

The signature of a residual is not asserted by hand.  Each residual is
declared in terms of the intermediate quantities it consumes, each
intermediate quantity is declared in terms of the structural equation that
produces it and the quantities that equation needs, and the signature is
the transitive closure over that graph, read against the fault incidence of
the structural model.  If the structural model and the implementation ever
disagree about where a fault lives, this file breaks rather than quietly
reporting the wrong signature.

The realised partition of the fault set by identical FSM columns is then
compared with the structural upper bound from E1.  The difference between
the two is the bound-tightness gap the plan asks for: the verifier can
never do better than the bound, and any shortfall is a deficiency of the
constructed residual bank, not of the theory.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.tep import tep_structural as tep                   # noqa: E402
from src.verifier import residuals_tep as R                 # noqa: E402

# quantity -> (equations that produce it, quantities it consumes)
# Measured quantities produce themselves through their measurement equation,
# which carries no fault, so they are simply roots.
DEPS = {
    # ---- measured roots
    "FTM1": ([], []), "FTM2": ([], []), "FTM3": ([], []), "FTM4": ([], []),
    "FTM6": ([], []), "FTM9": ([], []), "FTM10": ([], []),
    "PTR": ([], []), "PTS": ([], []), "PTV": ([], []),
    "TCR": ([], []), "TCS": ([], []), "TCC": ([], []),
    "TWR": ([], []), "TWS": ([], []), "VLR": ([], []),
    "QUC_meas": ([], []), "CPDH": ([], []),
    "XVV": ([], []), "XVS": ([], []), "XLC": ([], []),
    # ---- valve positions.  Valves 5, 7, 8 and 9 share the stiction fault
    #      through the intermediate variable dev19, so their positions
    #      depend on the equation that defines it.
    "DEV19": (["e_dev19"], []),
    **{"VPOS%d" % j: (["e_vpos%d" % j],
                      ["DEV19"] if j in (5, 7, 8, 9) else [])
       for j in range(1, 13)},
    # ---- separator inversion
    "XLS": (["e_PPS%d" % i for i in range(4, 9)]
            + ["e_XVS%d" % i for i in range(1, 9)], ["XVS", "PTS", "TCS"]),
    "DLS": (["e_DLS"], ["XLS", "TCS"]),
    "DLC": (["e_DLC"], ["XLC", "TCC"]),
    "FTM11": ([], ["DLS"]),
    "FTM13": ([], ["DLC"]),
    "FTM8": (["e_balS%d" % i for i in range(1, 9)]
             + ["e_FCM%d_9" % i for i in range(1, 9)]
             + ["e_FCM%d_10" % i for i in range(1, 9)]
             + ["e_FCM%d_11" % i for i in range(1, 9)],
             ["FTM9", "FTM10", "FTM11", "XVS", "XLS"]),
    "XST8": (["e_balS%d" % i for i in range(1, 9)]
             + ["e_FCM%d_8" % i for i in range(1, 9)],
             ["FTM8", "XVS", "XLS", "FTM9", "FTM10", "FTM11"]),
    "XMWS8": ([], ["XST8"]),
    "XMWS9": ([], ["XVS"]),
    "XMWS6": ([], ["XVV"]),
    # ---- stripper inversion: needs the stream-4 composition, which is
    #      exactly where IDV1/2/8 act
    # The three stream-4 mole fractions are tracked separately: IDV(1)
    # moves only x_A (and x_C by difference), IDV(2) moves x_A and x_B in a
    # fixed ratio, IDV(8) walks x_A and x_B independently (teprob.f
    # l.407-410).  Lumping them would hide exactly the distinctions the
    # element balances are able to make.
    "X4A": (["e_XST1_4", "e_dev2", "e_dev8"], []),
    "X4B": (["e_XST2_4", "e_dev2", "e_dev8"], []),
    "X4C": (["e_XST3_4"], ["X4A", "X4B"]),
    "X4": ([], ["X4A", "X4B", "X4C"]),
    "FTM5": (["e_FTM5"] + ["e_FIN%d" % i for i in range(1, 9)]
             + ["e_FCM%d_5" % i for i in range(1, 9)]
             + ["e_FCM%d_4" % i for i in range(1, 4)]
             + ["e_SFR%d" % i for i in range(4, 9)]
             + ["e_VOVRL", "e_TMPFAC"],
             ["X4", "FTM4", "XLS", "FTM11", "XLC", "FTM13"]),
    "XST5": (["e_XST%d_5" % i for i in range(1, 9)],
             ["FTM5", "X4", "FTM4", "XLS", "FTM11", "XLC", "FTM13"]),
    # ---- reaction rates from the reactor component balance
    "CRXR": (["e_balR%d" % i for i in range(1, 9)]
             + ["e_FCM%d_6" % i for i in range(1, 9)],
             ["FTM8", "XST8", "FTM6", "XVV"]),
    "RR": (["e_CRXR1", "e_CRXR3", "e_CRXR4", "e_CRXR5", "e_CRXR6",
            "e_CRXR7", "e_CRXR8"], ["CRXR"]),
    "RH": (["e_RH"], ["RR"]),
    # ---- heat transfer
    "AGSP": (["e_AGSP"], ["VPOS12"]),
    "UAR": (["e_UAR"], ["VLR", "AGSP"]),
    "QUR": (["e_QUR"], ["UAR", "TWR", "TCR"]),
    "UAS": (["e_UAS"], ["FTM8"]),
    "QUS": (["e_QUS"], ["UAS", "TWS", "TCR"]),
    "FWR": (["e_FWR"], ["VPOS10"]),
    "FWS": (["e_FWS"], ["VPOS11"]),
    "TCWR": (["e_TCWR"], []),
    "TCWS": (["e_TCWS"], []),
    "UAC": (["e_UAC"], ["VPOS9"]),
    # ---- enthalpies
    "HST1": (["e_HST1", "e_TST1"], []),
    "HST2": (["e_HST2"], []),
    "HST3": (["e_HST3"], []),
    "HST4": (["e_HST4", "e_TST4"], ["X4"]),
    "HST5": (["e_HST5"], ["XST5", "TCC"]),
    "HST8": (["e_HST8"], ["XST8", "TCR"]),
    "HST9": (["e_HST9", "e_HST9b"], ["XVS", "TCS", "CPDH", "FTM9"]),
    "HST11": (["e_HST11"], ["XLS", "TCS"]),
    "HST13": (["e_HST13"], ["XLC", "TCC"]),
}

# residual -> (equations it closes, quantities it consumes)
RESIDUALS = {
    "r_v1_Dfeed":   (["e_FTM1"], ["FTM1", "VPOS1"]),
    "r_v2_Efeed":   (["e_FTM2"], ["FTM2", "VPOS2"]),
    "r_v3_Afeed":   (["e_FTM3"], ["FTM3", "VPOS3"]),
    "r_v4_ACfeed":  (["e_FTM4"], ["FTM4", "VPOS4"]),
    "r_v5_recycle": (["e_FTM9"], ["FTM9", "VPOS5", "XMWS9", "PTV", "PTS"]),
    "r_v6_purge":   (["e_FTM10"], ["FTM10", "VPOS6", "XMWS9", "PTS"]),
    "r_v7_sepflow": (["e_FTM11"], ["FTM11", "VPOS7"]),
    "r_v8_product": (["e_FTM13"], ["FTM13", "VPOS8"]),
    "r_steam":      (["e_QUC"], ["QUC_meas", "UAC", "TCC"]),
    "r_cpdh":       (["e_CPDH"], ["CPDH", "PTV", "PTS", "TCS", "XMWS9"]),
    "r_f6_reacfeed": (["e_FTM6"], ["FTM6", "PTV", "PTR", "XMWS6"]),
    "r_f8_reacout": (["e_FTM8"], ["FTM8", "PTR", "PTS", "XMWS8"]),
    "r_cwR":        (["e_TWR"], ["FWR", "TCWR", "TWR", "QUR"]),
    "r_cwS":        (["e_TWS"], ["FWS", "TCWS", "TWS", "QUS"]),
    "r_balA":       ([], ["X4A", "FTM3", "FTM4", "FTM10", "FTM13", "XVS",
                          "XLC", "FTM2"]),
    "r_balB":       ([], ["X4B", "FTM1", "FTM3", "FTM4", "FTM10", "XVS"]),
    "r_balC":       ([], ["X4C", "FTM4", "FTM10", "FTM13", "XVS", "XLC"]),
    "r_balD":       ([], ["FTM1", "FTM2", "FTM10", "FTM13", "XVS", "XLC"]),
    "r_dec2":       ([], ["X4A", "X4B", "FTM1", "FTM2", "FTM3", "FTM4",
                          "FTM10", "FTM13", "XVS", "XLC"]),
    "r_energy":     (["e_balV_E", "e_balR_E"],
                     ["HST1", "HST2", "HST3", "HST5", "HST9", "HST8",
                      "FTM1", "FTM2", "FTM3", "FTM5", "FTM9", "FTM8",
                      "RH", "QUR"]),
    "r_sep_En":     (["e_balS_E"],
                     ["HST8", "FTM8", "HST9", "FTM9", "FTM10", "HST11",
                      "FTM11", "QUS", "XLS", "XVS", "TCS"]),
    "r_strip_En":   (["e_balC_E"], ["HST4", "HST11", "HST5", "HST13",
                                    "FTM4", "FTM11", "FTM5", "FTM13",
                                    "QUC_meas"]),
    "r_kin1":       (["e_RR1", "e_dev13"], ["RR", "TCR", "XST8", "PTR", "VLR"]),
    "r_kin2":       (["e_RR2", "e_dev13"], ["RR", "TCR", "XST8", "PTR", "VLR"]),
    "r_kin3":       (["e_RR3"], ["RR", "TCR", "XST8", "PTR", "VLR"]),
    "r_kin4":       (["e_RR4"], ["RR", "TCR", "XST8", "PTR", "VLR"]),
}

# r_dec2 is a decoupling combination, not an independent balance: it is the
# linear combination of the A and B element balances that annihilates the
# direction in which IDV(2) moves the stream-4 composition (dx_A/dx_B =
# -2.43719e-3 / 5.0e-3, taken from teprob.f l.407-409, i.e. from the model,
# not from data).  It therefore stays quiet under IDV(2) while still firing
# under IDV(1) and IDV(8), which is what separates those three faults.
# Structurally it lives on the same equations as the balances it combines,
# so its signature is computed the same way and then IDV2 is removed.
DECOUPLED = {"r_dec2": ["fIDV2"]}


def _closure(eqs, quants, seen=None):
    if seen is None:
        seen = set()
    out = set(eqs)
    for q in quants:
        if q in seen:
            continue
        seen.add(q)
        e, sub = DEPS[q]
        out |= _closure(e, sub, seen)
    return out


def build_fsm():
    """Return (residual names, fault names, FSM array, equation support)."""
    model = tep.build_model()
    eqindex = {nm: i for i, nm in enumerate(model.eqnames)}
    F = np.asarray(model.F.todense() if hasattr(model.F, "todense")
                   else model.F)
    fnames = list(model.f)

    rnames = list(RESIDUALS.keys())
    fsm = np.zeros((len(rnames), len(fnames)), dtype=int)
    support = {}
    for i, rn in enumerate(rnames):
        eqs, quants = RESIDUALS[rn]
        sup = _closure(eqs, quants)
        missing = [e for e in sup if e not in eqindex]
        if missing:
            raise KeyError("%s: equations not in the structural model: %s"
                           % (rn, missing))
        support[rn] = sorted(sup)
        rows = [eqindex[e] for e in sup]
        fsm[i] = (F[rows].sum(0) > 0).astype(int)
        for f in DECOUPLED.get(rn, []):
            fsm[i, fnames.index(f)] = 0
    return rnames, fnames, fsm, support


def partition_from_fsm(fsm, fnames, restrict=None):
    """Group faults with identical FSM columns."""
    keep = [i for i, f in enumerate(fnames)
            if restrict is None or f[1:] in restrict]
    groups = {}
    for i in keep:
        groups.setdefault(tuple(fsm[:, i]), []).append(fnames[i][1:])
    return sorted(groups.values(), key=lambda b: b[0])
