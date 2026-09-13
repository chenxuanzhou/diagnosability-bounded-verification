"""
E8: is the language model necessary, and where.  Within-subject design.

The design does not ask whether an LLM is good at diagnosis.  It asks a
question whose answer is already constrained by a proved fact.  R1 says the
verifier is blind inside an indiscernibility class, so conditional on the
admissible set having more than one member, a method reading only the sensor
channels cannot beat 1/|D| there.  That is a theorem, not an empirical
finding, which is what makes the comparison hard to argue with: any method
that beats it must be reading something the sensors never carried.

Why within-subject.  An earlier version assigned each case ONE information
level, so the three levels were three different case sets.  That made the
sensor-only control appear to vary with the level of a log it never saw,
which is impossible and was purely a case-difficulty artefact.  It also
weakened the main claim, because a reviewer could attribute the graded gain
to the cases rather than to the text.

Here every case is run under every condition:

  A  verifier alone            a uniform pick inside the admissible set.
                               Expected score 1/|D| by construction.
  B  verifier + agent, no text the agent sees the window and the admissible
                               set.  Run ONCE per case: the level does not
                               exist for it, so its per-level numbers are
                               identical by construction, which is the point.
  C  verifier + agent + log     run THREE times per case, once with an
                               irrelevant log, once with a weakly relevant
                               one, once with a log that settles it.

Any gradient in C across levels is therefore attributable to the text alone,
because the cases, the window, the admissible set, the model and the seed are
all held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.agents import baselines as B                       # noqa: E402
from src.agents import llm as LLM                           # noqa: E402
from src.eval import cases as CS                            # noqa: E402
from src.eval import oplogs as OL                           # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "E8")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA

AMBIGUOUS = ["L_reac_cw_temp", "L_reac_cw_valve",
             "L_cond_cw_temp", "L_cond_cw_valve"]

PROMPT = (
    "You are supporting an operator on the Tennessee Eastman plant.\n\n"
    "A physical consistency check has already been run on the plant's "
    "redundancy relations. It has narrowed the cause to the following "
    "candidates and can go no further: on this instrumentation the "
    "candidates produce the same residual pattern, so no further "
    "measurement on the installed sensors can separate them.\n\n"
    "Candidates:\n{cands}\n\n"
    "Choose the single most likely candidate.{extra}\n\n"
    'Answer with one JSON object and nothing else:\n'
    '{{"fault": "<one candidate id>", "confidence": <0 to 1>, '
    '"reason": "<one sentence>"}}'
)


def build_cases(n_per_location=20, seed=7):
    """Windows from the ambiguous locations where the verifier is stuck."""
    rng = np.random.default_rng(seed)
    Xf, mf = CS._load("fault")
    Xfit, _ = CS._load("nominal_fit")
    pres = CS.Presenter(Xfit)

    Xa, ma = CS._load("nominal_fit")
    Xb, mb = CS._load("nominal_cal")
    A, _, _ = E2.features(Xa, ma, faulty=False)
    Bc, _, _ = E2.features(Xb, mb, faulty=False)
    bank = CF.ConformalBank().fit_scale(A).calibrate(Bc)
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    verifier = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]

    T = Xf.shape[1]
    picked = []
    for loc in AMBIGUOUS:
        idvs = P.LOCATIONS[loc][0]
        pool = []
        for i, m in enumerate(mf):
            if m["fault"] not in idvs:
                continue
            end = E2._usable_end(m, T)
            pool += [(i, s) for s in
                     range(E2.IDV_ON + E2.SETTLE, end - E2.WIDTH + 1, 30)]
        rng.shuffle(pool)
        got = 0
        for i, s in pool:
            if got >= n_per_location:
                break
            W = Xf[i, s:s + E2.WIDTH]
            F = CF.window_matrix(W, [0], E2.WIDTH)
            trig = bank.residual_triggers(F, ALPHA)[0, order]
            D, rule = verifier.candidates(trig)
            if len(D) > 1 and loc in D:
                picked.append({"id": "%s_r%d_s%d" % (loc, i, s),
                               "location": loc, "idv": int(mf[i]["fault"]),
                               "run": int(i), "start": int(s), "window": W,
                               "candidates": D, "rule": rule})
                got += 1
    return picked, pres


def ask(case, pres, model, seed, log_text, tag):
    cands = "\n".join("%s: %s" % (d, P.LOCATIONS[d][1] if d in P.LOCATIONS
                                  else "the plant is operating normally")
                      for d in case["candidates"])
    extra = ("\n\nOperations shift log for the last 24 hours:\n" + log_text
             if log_text else "")
    msgs = [{"role": "system",
             "content": PROMPT.format(cands=cands, extra=extra)},
            {"role": "user", "content": "Plant data window:\n"
                                        + pres.text(case["window"])}]
    kw = {}
    if LLM.MODELS[model][0] == "ollama":
        kw = {"temperature": 0.0, "seed": seed}
    txt, meta = LLM.chat(model, msgs, max_tokens=700, log_tag=tag, **kw)
    ans, _ = B.parse_answer(txt)
    if ans not in case["candidates"]:
        ans = None
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kimi-k3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-location", type=int, default=20)
    ap.add_argument("--tag", default="within")
    args = ap.parse_args()

    cases, pres = build_cases(args.per_location)
    if not cases:
        print("no ambiguous cases found; E8 cannot run")
        return

    # every case gets one log of each information level, drawn from the same
    # generator with a per-case seed, so the three logs of a case differ only
    # in how much they say
    logs = {lv: [OL.make_log(c["location"], lv, 900 + i)
                 for i, c in enumerate(cases)]
            for lv in OL.LEVELS}

    chance = float(np.mean([1.0 / len(c["candidates"]) for c in cases]))
    print("E8 within-subject: %d cases" % len(cases))
    print("  by location: %s" % dict(Counter(c["location"] for c in cases)))
    print("  |D| sizes  : %s" % dict(Counter(len(c["candidates"])
                                             for c in cases)))
    print("  condition A, verifier alone, expected accuracy = %.3f" % chance)
    print("  every case is run under condition B once and under condition C "
          "at each of the %d information levels" % len(OL.LEVELS))

    json.dump({"design": "within-subject", "n_cases": len(cases),
               "levels": list(OL.LEVELS), "chance": chance, "logs": logs,
               "cases": [{k: v for k, v in c.items() if k != "window"}
                         for c in cases]},
              open(os.path.join(OUT, "protocol_%s.json" % args.tag), "w"),
              indent=2)

    rows, raw = [], {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        w = 1 if LLM.MODELS[model][0] == "ollama" else 3
        for sd in range(args.seeds):
            conds = [("B", None)] + [("C", lv) for lv in OL.LEVELS]
            for cond, lv in conds:
                t0 = time.time()

                def one(i):
                    try:
                        return ask(cases[i], pres, model, sd,
                                   logs[lv][i] if lv else "",
                                   "E8/%s/%s/%s" % (cond, lv or "none",
                                                    cases[i]["id"]))
                    except Exception as e:              # noqa: BLE001
                        return None

                with ThreadPoolExecutor(max_workers=w) as ex:
                    ans = list(ex.map(one, range(len(cases))))
                ok = [1.0 if a == cases[i]["location"] else 0.0
                      for i, a in enumerate(ans)]
                row = {"model": model, "condition": cond, "level": lv,
                       "seed": sd, "accuracy": float(np.mean(ok)),
                       "answered": float(np.mean([a is not None
                                                  for a in ans])),
                       "n": len(cases),
                       "wall_s": round(time.time() - t0, 1)}
                rows.append(row)
                raw["%s|%s|%s|%d" % (model, cond, lv or "none", sd)] = [
                    {"case": cases[i]["id"], "truth": cases[i]["location"],
                     "answer": a} for i, a in enumerate(ans)]
                print("%-12s cond %s %-10s seed %d  acc %.3f  %.0fs"
                      % (model, cond, lv or "-", sd, row["accuracy"],
                         row["wall_s"]), flush=True)
                json.dump({"rows": rows, "raw": raw, "chance": chance,
                           "design": "within-subject",
                           "n_cases": len(cases)},
                          open(os.path.join(OUT, "E8_%s.json" % args.tag),
                               "w"), indent=2)

    # ---- summary with a paired comparison, which the design now allows
    print("")
    print("%-12s %-10s %8s  %s" % ("model", "condition", "accuracy",
                                   "vs condition B, paired over cases"))
    for model in sorted({r["model"] for r in rows}):
        base = [r for r in rows if r["model"] == model
                and r["condition"] == "B"]
        bacc = float(np.mean([r["accuracy"] for r in base]))
        print("%-12s %-10s %8.3f" % (model, "B (no log)", bacc))
        for lv in OL.LEVELS:
            sel = [r for r in rows if r["model"] == model
                   and r["condition"] == "C" and r["level"] == lv]
            if not sel:
                continue
            acc = float(np.mean([r["accuracy"] for r in sel]))
            print("%-12s %-10s %8.3f  %+.3f"
                  % (model, "C " + lv, acc, acc - bacc))
    print("")
    print("wrote %s" % os.path.join(OUT, "E8_%s.json" % args.tag))


if __name__ == "__main__":
    main()
