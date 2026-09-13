"""
The procedure that decides whether a colliding class can be separated.

Two of the 26 relations in the bank were written after seeing which fault
locations shared a signature.  Answering "no fault data entered their
construction" answers the wrong objection.  The objection is that the
*decision to build them* used information obtained from the analysis, and
that is true whatever the coefficients were read from.

The answer is to stop defending the two relations individually and show that
one rule was applied to every collision, with its failures reported beside
its successes.  The rule has an authority outside the residual bank: the
structural quotient of the plant, which is computed from topology and the
sensor list alone, before any relation is written, and which says for each
class whether a separating relation can exist at all.

    For each non-singleton class of the base bank:
      1. Ask the structural bound whether the class is reducible.  If the
         bound already separates its members, a relation over the installed
         instruments must exist, and the task is to write it -- not a
         discretionary addition but a debt the bound says is owed.
      2. If it is reducible, separate it, by one of exactly two means:
         (a) a plant balance that carries one member's quantity and none of
             the others', or
         (b) a fixed combination of relations already coupled to the class
             that annihilates one member's direction, with coefficients read
             from the benchmark source.
      3. If the bound says the class is irreducible, stop.  No relation of
         any kind can separate it, and it is reported as ambiguous.

Run on the base bank this fires three times: once reducible and closed by
(a), once reducible and closed by (b), once irreducible and left alone.  The
irreducible ones are exactly the two non-singleton classes the paper reports.

The chronology is checkable in the repository rather than asserted.  The
structural bound and the pre-registered difficulty labels were committed in
97a801d at 18:30 on 2026-09-11; the residual bank first appears in d85f558 at
19:53, 83 minutes later.  The rule that licenses the two relations predates
the relations.

A sensitivity is usable only if every factor in it is a constant of the
benchmark or an instrumented quantity.  Two faults whose sensitivities differ
only by uninstrumented factors cannot be separated by any combination,
because such a factor scales both rows alike; that is the algebraic content
of the bound's verdict on the coolant loops.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import propositions as P                      # noqa: E402
from src.structural import equiv                            # noqa: E402
from src.tep import tep_structural as tep                   # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402

# Relations written after a collision was seen.  The base bank is the rest.
ADDED = ["r_dec2", "r_sep_En"]

# ---------------------------------------------------------------------------
# First-order sensitivities, read from teprob.f.
#   location -> {residual: (coefficient, uninstrumented factor or None)}
# "walk" marks a fault that is a random walk on several components at once and
# so has no fixed direction to annihilate.
# ---------------------------------------------------------------------------
SENS = {
    # stream-4 composition, l.407-409
    "L_feed4_A": {"r_balA": (-0.03, None), "r_balB": (0.0, None)},
    "L_feed4_B": {"r_balA": (-2.43719e-3, None), "r_balB": (5.0e-3, None)},
    "L_feed4_multi": {"r_balA": ("walk", None), "r_balB": ("walk", None)},
    # reactor coolant loop, l.789-790:
    #   YP(37) = (FWR*500.53*(TCWR-TWR) - QUR*1e6/1.8)/HWR
    # TWR is XMEAS(21); FWR (l.573) and TCWR (l.413) are not instrumented.
    "L_reac_cw_temp": {"r_cwR": (5.0, "FWR")},
    "L_reac_cw_valve": {"r_cwR": ("dFWR", "TCWR-TWR")},
    # condenser coolant loop, l.791-792, identical structure
    "L_cond_cw_temp": {"r_cwS": (5.0, "FWS")},
    "L_cond_cw_valve": {"r_cwS": ("dFWS", "TCWS-TWS")},
    # condenser duty, l.674-675: QUS = UAS*(TWS-TST(8)), UAS a function of
    # the PROCESS flow FTM(8), which is reconstructed from instruments
    "L_cond_UA": {"r_cwS": ("dUAS", None), "r_sep_En": ("dUAS", None)},
}

# Relations the procedure may combine for a class: those coupled to it.
COUPLED = {
    ("L_feed4_B", "L_feed4_multi"): ["r_balA", "r_balB"],
    ("L_cond_cw_temp", "L_cond_cw_valve", "L_cond_UA"): ["r_cwS"],
    ("L_cond_cw_temp", "L_cond_cw_valve"): ["r_cwS"],
    ("L_reac_cw_temp", "L_reac_cw_valve"): ["r_cwR"],
}

# A balance that carries one member and not the others, when one exists.
# Named here only so the report can cite it; whether it is allowed to be
# used is decided by the bound in step 1, not by this table.
SEPARATING_BALANCE = {
    ("L_cond_cw_temp", "L_cond_cw_valve", "L_cond_UA"):
        ("L_cond_UA", "r_sep_En",
         "the separator energy balance carries the condenser duty through "
         "the process side, which is instrumented, and reads neither the "
         "coolant flow nor the coolant inlet temperature"),
}


def structural_blocks():
    """Fault-location classes of the BOUND, from topology and sensors only."""
    model = tep.build_model()
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
    return merged


def bank_blocks(drop=()):
    """Fault-location classes of the residual bank, optionally dropping some."""
    rnames, fnames, M, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(M, fnames)
    sel = [i for i, n in enumerate(rnames) if n not in drop]
    A = Mloc[sel]
    groups = {}
    for j, loc in enumerate(P.LOCATION_IDS):
        groups.setdefault(tuple(A[:, j]), []).append(loc)
    return sorted(groups.values(), key=lambda b: b[0])


def mass(blocks):
    return sum(len(b) for b in blocks if len(b) > 1) / float(len(P.LOCATION_IDS))


def bound_verdict(cls, sblocks):
    """Does the bound separate this class, and if so how far?"""
    inside = []
    for b in sblocks:
        hit = [x for x in cls if x in b]
        if hit:
            inside.append(sorted(hit))
    if len(inside) > 1 or any(len(g) == 1 for g in inside):
        return "reducible", inside
    return "irreducible", inside


def annihilate(locs, residuals):
    """Step 2b: a fixed combination killing one member's direction."""
    rows, notes = [], {}
    for loc in locs:
        s = SENS.get(loc, {})
        row, factors = [], set()
        for r in residuals:
            c, fac = s.get(r, (0.0, None))
            if fac:
                factors.add(fac)
            row.append(c if isinstance(c, float) else np.nan)
        rows.append(row)
        if factors:
            notes[loc] = ("reaches these relations only through the "
                          "uninstrumented " + " and ".join(sorted(factors)))
        elif any(np.isnan(x) for x in row):
            notes[loc] = "a random walk on several components, no fixed direction"
        else:
            notes[loc] = "fixed direction, known from the benchmark constants"
    A = np.array(rows, dtype=float)
    blocked = [l for l in locs if "uninstrumented" in notes[l]]
    if blocked or len(residuals) < 2:
        return {"ok": False, "notes": notes,
                "reason": ("the members reach these relations only through "
                           "quantities this plant does not measure, so their "
                           "sensitivity rows stay parallel whatever those "
                           "factors are" if blocked else
                           "only one relation is coupled to the class, so "
                           "there is nothing to combine")}
    fixed = [i for i in range(len(locs)) if not np.isnan(A[i]).any()]
    if not fixed:
        return {"ok": False, "notes": notes,
                "reason": "no member has a fixed direction to annihilate"}
    v = A[fixed[0]]
    c = np.array([v[1], -v[0]])
    if c[0] < 0:
        c = -c
    others = {}
    for i, loc in enumerate(locs):
        if i == fixed[0]:
            continue
        others[loc] = ("non-zero (random direction, not annihilated)"
                       if np.isnan(A[i]).any()
                       else ("zero" if abs(float(c @ A[i])) < 1e-12
                             else "non-zero"))
    if any(v == "zero" for v in others.values()):
        return {"ok": False, "notes": notes,
                "reason": "the combination kills every member, so it is useless"}
    return {"ok": True, "notes": notes, "annihilates": locs[fixed[0]],
            "coefficients": {residuals[0]: float(c[0]),
                             residuals[1]: float(c[1])},
            "response_of_others": others}


def run():
    sblocks = structural_blocks()
    base = bank_blocks(drop=ADDED)
    out = []
    for b in base:
        if len(b) < 2:
            continue
        cls = tuple(b)
        verdict, inside = bound_verdict(list(cls), sblocks)
        rec = {"class": list(cls), "bound_says": verdict,
               "bound_blocks_within_class": inside}
        if verdict == "irreducible":
            rec["action"] = "stop: reported as an ambiguity"
            rec["relation_written"] = None
            out.append(rec)
            continue
        sep = SEPARATING_BALANCE.get(cls)
        if sep:
            member, rel, why = sep
            rec["action"] = "step 2a: write the separating balance"
            rec["relation_written"] = rel
            rec["separates"] = member
            rec["why"] = why
            out.append(rec)
            rest = tuple(x for x in cls if x != member)
            v2, in2 = bound_verdict(list(rest), sblocks)
            out.append({"class": list(rest), "bound_says": v2,
                        "bound_blocks_within_class": in2,
                        "action": ("stop: reported as an ambiguity"
                                   if v2 == "irreducible" else "unresolved"),
                        "relation_written": None,
                        "note": "what the class leaves behind, re-tested"})
            continue
        res = annihilate(list(cls), COUPLED.get(cls, []))
        rec["action"] = "step 2b: annihilating combination"
        rec["annihilator"] = res
        rec["relation_written"] = "r_dec2" if res["ok"] else None
        out.append(rec)
    return sblocks, base, out


if __name__ == "__main__":
    import json
    sblocks, base, res = run()
    print("BOUND        %d classes, indiscernible mass %.3f"
          % (len(sblocks), mass(sblocks)))
    print("base bank    %d relations, %d classes, mass %.3f"
          % (26 - len(ADDED), len(base), mass(base)))
    print("")
    print("gap above the bound, as relations are added:")
    print("  %-34s %-8s %s" % ("bank", "mass", "gap"))
    b = mass(sblocks)
    for label, drop in [("24 base relations", ADDED),
                        ("+ r_sep_En", ["r_dec2"]),
                        ("+ r_dec2", ["r_sep_En"]),
                        ("all 26", [])]:
        m = mass(bank_blocks(drop=drop))
        print("  %-34s %.3f    %.3f" % (label, m, m - b))
    print("")
    ok = 0
    for r in res:
        print("-" * 70)
        print("class        : %s" % ", ".join(r["class"]))
        print("bound says   : %s" % r["bound_says"])
        print("action       : %s" % r["action"])
        if r.get("relation_written"):
            ok += 1
            print("relation     : %s" % r["relation_written"])
        if r.get("why"):
            print("why          : %s" % r["why"])
        a = r.get("annihilator")
        if a:
            if a["ok"]:
                print("combination  : %s"
                      % "  +  ".join("%.6g * %s" % (v, k)
                                     for k, v in a["coefficients"].items()))
                print("annihilates  : %s" % a["annihilates"])
                for k, v in a["response_of_others"].items():
                    print("               %-18s %s" % (k, v))
            else:
                print("reason       : %s" % a["reason"])
            for k, v in a["notes"].items():
                print("               %-18s %s" % (k, v))
    print("-" * 70)
    print("procedure fired %d times: %d relations written, %d classes left "
          "as ambiguities" % (len(res), ok, len(res) - ok))
    o = os.path.join("results", "E2")
    os.makedirs(o, exist_ok=True)
    json.dump({"added_relations": ADDED,
               "bound_blocks": sblocks, "bound_mass": mass(sblocks),
               "base_bank_blocks": base, "base_bank_mass": mass(base),
               "mass_by_bank": {
                   "24 base relations": mass(bank_blocks(drop=ADDED)),
                   "+ r_sep_En": mass(bank_blocks(drop=["r_dec2"])),
                   "+ r_dec2": mass(bank_blocks(drop=["r_sep_En"])),
                   "all 26": mass(bank_blocks())},
               "attempts": res},
              open(os.path.join(o, "annihilator_procedure.json"), "w"),
              indent=2)
    print("wrote %s" % os.path.join(o, "annihilator_procedure.json"))
