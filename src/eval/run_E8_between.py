"""
E8: is the language model necessary, and where.

The design does not ask whether an LLM is good at diagnosis.  It asks a
question whose answer is already constrained by a proved fact.  R1 says the
verifier is blind inside an indiscernibility class, so conditional on the
admissible set having more than one member, a method that reads only the
sensor channels cannot beat 1/|D| there.  That is a theorem, not an
empirical finding, which is what makes this comparison hard to argue with:
any method that does beat it must be reading something the sensors never
carried.

Three conditions, all restricted to cases where |D(y)| > 1 and the truth is
inside D:

  A  verifier alone            a uniform pick inside the admissible set.
                               Its expected score is 1/|D| by construction.
  B  verifier + LLM, no text   the model sees the window and the admissible
                               set.  Everything it has came through a
                               sensor, so the theory predicts chance.
  C  verifier + LLM + log      the model also sees a synthetic shift log
                               whose information content was fixed in
                               advance.

Condition B is the control that matters.  Without it, a gain in C could be
explained by the model reading the sensor data more cleverly; with it, any
gain in C is attributable to the text.

Results are reported stratified by the log's information level, because a
gain that appears only on the strongly informative logs is evidence for the
mechanism, whereas a uniform gain would suggest something else is going on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

# the locations that sit in a non-singleton structural class
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


def build_e8_cases(n_per_location=10, seed=7):
    """Windows from the ambiguous locations where the verifier is stuck."""
    rng = np.random.default_rng(seed)
    Xf, mf = CS._load("fault")
    Xfit, _ = CS._load("nominal_fit")
    pres = CS.Presenter(Xfit)

    Ffit, mfit2 = CS._load("nominal_fit")
    Fcal, mcal = CS._load("nominal_cal")
    A, _, _ = E2.features(Ffit, mfit2, faulty=False)
    Bc, _, _ = E2.features(Fcal, mcal, faulty=False)
    bank = CF.ConformalBank().fit_scale(A).calibrate(Bc)
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    verifier = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)

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
                     range(E2.IDV_ON + E2.SETTLE, end - E2.WIDTH + 1, 60)]
        rng.shuffle(pool)
        got = 0
        for i, s in pool:
            W = Xf[i, s:s + E2.WIDTH]
            F = CF.window_matrix(W, [0], E2.WIDTH)
            order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]
            trig = bank.residual_triggers(F, ALPHA)[0, order]
            D, rule = verifier.candidates(trig)
            if len(D) > 1 and loc in D:
                picked.append({"id": "%s_r%d_s%d" % (loc, i, s),
                               "location": loc, "idv": int(mf[i]["fault"]),
                               "run": int(i), "start": int(s), "window": W,
                               "candidates": D, "rule": rule})
                got += 1
            if got >= n_per_location:
                break
    return picked, pres


def ask(case, pres, model, seed, with_log, level, log_text):
    cands = "\n".join("%s: %s" % (d, P.LOCATIONS[d][1] if d in P.LOCATIONS
                                  else "no fault")
                      for d in case["candidates"])
    extra = ""
    if with_log:
        extra = ("\n\nOperations shift log for the last 24 hours:\n"
                 + log_text)
    body = PROMPT.format(cands=cands, extra=extra)
    msgs = [{"role": "system", "content": body},
            {"role": "user", "content": "Plant data window:\n"
                                        + pres.text(case["window"])}]
    kw = {}
    if LLM.MODELS[model][0] == "ollama":
        kw = {"temperature": 0.0, "seed": seed}
    txt, meta = LLM.chat(model, msgs, max_tokens=700,
                         log_tag="E8/%s/%s" % ("log" if with_log else "nolog",
                                               case["id"]), **kw)
    ans, _ = B.parse_answer(txt)
    if ans not in case["candidates"]:
        ans = None
    return {"answer": ans, "raw": txt, "meta": meta, "level": level}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kimi-k3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-location", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    cases, pres = build_e8_cases(args.per_location)
    if not cases:
        print("no ambiguous cases found; E8 cannot run")
        return
    levels = OL.assign_levels(len(cases))
    logs = [OL.make_log(c["location"], lv, 900 + i)
            for i, (c, lv) in enumerate(zip(cases, levels))]
    print("E8 cases: %d" % len(cases))
    from collections import Counter
    print("  by location: %s" % dict(Counter(c["location"] for c in cases)))
    print("  |D| sizes  : %s" % dict(Counter(len(c["candidates"])
                                             for c in cases)))
    print("  info levels: %s" % dict(Counter(levels)))

    chance = float(np.mean([1.0 / len(c["candidates"]) for c in cases]))
    print("  condition A, verifier alone, expected accuracy = %.3f" % chance)

    protocol = {"mix": OL.MIX, "levels": levels, "n_cases": len(cases),
                "chance": chance, "logs": logs,
                "cases": [{k: v for k, v in c.items() if k != "window"}
                          for c in cases]}
    json.dump(protocol, open(os.path.join(OUT, "protocol.json"), "w"),
              indent=2)

    rows, raw = [], {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for with_log in (False, True):
            for sd in range(args.seeds):
                t0 = time.time()
                w = 1 if LLM.MODELS[model][0] == "ollama" else 2

                def one(i):
                    try:
                        return ask(cases[i], pres, model, sd, with_log,
                                   levels[i], logs[i])
                    except Exception as e:                  # noqa: BLE001
                        return {"answer": None, "error": repr(e)[:160],
                                "level": levels[i]}

                with ThreadPoolExecutor(max_workers=w) as ex:
                    recs = list(ex.map(one, range(len(cases))))
                ok = [r["answer"] == cases[i]["location"]
                      for i, r in enumerate(recs)]
                by_level = {}
                for lv in OL.LEVELS:
                    sel = [o for o, l in zip(ok, levels) if l == lv]
                    by_level[lv] = float(np.mean(sel)) if sel else float("nan")
                row = {"model": model, "condition": "C" if with_log else "B",
                       "seed": sd, "accuracy": float(np.mean(ok)),
                       "by_level": by_level,
                       "answered": float(np.mean([r["answer"] is not None
                                                  for r in recs])),
                       "wall_s": round(time.time() - t0, 1)}
                rows.append(row)
                raw["%s|%s|%d" % (model, row["condition"], sd)] = [
                    {"case": cases[i]["id"], "truth": cases[i]["location"],
                     "answer": r.get("answer"), "level": r.get("level")}
                    for i, r in enumerate(recs)]
                print("%-12s cond %s seed %d  acc %.3f  (irrelevant %.2f, "
                      "weak %.2f, strong %.2f)  %.0fs"
                      % (model, row["condition"], sd, row["accuracy"],
                         by_level["irrelevant"], by_level["weak"],
                         by_level["strong"], row["wall_s"]), flush=True)
                json.dump({"rows": rows, "raw": raw, "chance": chance,
                           "protocol": {k: v for k, v in protocol.items()
                                        if k != "cases"}},
                          open(os.path.join(OUT, "E8_%s.json" % args.tag),
                               "w"), indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E8_%s.json" % args.tag))


if __name__ == "__main__":
    main()
