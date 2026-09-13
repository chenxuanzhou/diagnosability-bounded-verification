"""
E3 and E4: the baseline ladder on the Tennessee Eastman plant.

E3 is the unverified part of the ladder (B0 to B4) and exists to show the
problem is real.  E4 adds the structural verification layer (B5) and the
two reference points that bracket it: the verifier alone (B6) and the
oracle inside the admissible set (B7).

Metrics, all defined over the same case set:

  FAR_phys     fraction of ALL cases where the pipeline emitted a
               proposition that is inconsistent with the data, meaning it
               falls outside the indiscernibility class of the true fault.
               This is the headline number.
  exact        fraction of cases answered with the exact true location
  class        fraction of cases answered inside the true class.  Under the
               bound this is the honest accuracy measure; exact match is
               reported only for reference, because the bound forbids
               resolving inside a class.
  escalation   fraction of cases routed to a human
  answered     fraction of cases where the pipeline committed to anything
  |D|          size of the admissible set

Every LLM call is logged raw, with model id, call date, token usage and
latency.
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
from src.agents import detector as DET                      # noqa: E402
from src.agents import llm as LLM                           # noqa: E402
from src.agents import catalogue as CAT                     # noqa: E402
from src.eval import cases as CS                            # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "E4")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA


def setup(n_per_location=4, n_nf=8, case_seed=0):
    cases, pres = CS.build_cases(n_per_location, n_nf, case_seed)
    Xfit, mfit = CS._load("nominal_fit")
    Xcal, mcal = CS._load("nominal_cal")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    bank = CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)

    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    verifier = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)

    F = CS.case_features(cases)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]
    trig = bank.residual_triggers(F, ALPHA)[:, order]

    det = DET.PCADetector().fit(Xfit, CAT.CHANNELS)
    starts = list(range(60, Xcal.shape[1] - E2.WIDTH + 1, 60))
    det.calibrate(Xcal, E2.WIDTH, starts, ALPHA)
    return cases, pres, bank, verifier, trig, det


def run_config(name, cases, pres, verifier, trig, det, model, seed,
               workers=3, logs=None):
    """Run one baseline over every case."""
    rng = np.random.default_rng(1234 + seed)
    logs = logs or [""] * len(cases)

    def one(i):
        c = cases[i]
        t0 = time.time()
        try:
            return _one(i, c, t0)
        except Exception as e:                              # noqa: BLE001
            # a single case that the endpoint refuses must not cost the
            # whole grid; it is recorded as unanswered and counted
            return {"case": c["id"], "truth": c["location"], "answer": None,
                    "error": repr(e)[:200], "wall_s": time.time() - t0,
                    "n_candidates": 0}

    def _one(i, c, t0):
        if name == "B0":
            r = B.b0_bare(c, pres, model, seed)
        elif name == "B1":
            r = B.b1_rag(c, pres, model, seed)
        elif name == "B2":
            r = B.b2_self_consistency(c, pres, model, seed)
        elif name == "B3":
            r = B.b3_tools(c, pres, model, seed)
        elif name == "B4":
            r = B.b4_detector(c, pres, model, seed, det)
        elif name == "B5":
            r = B.b5_verified(c, pres, model, seed, verifier, trig[i])
        elif name == "B5L":
            r = B.b5_verified_with_log(c, pres, model, seed, verifier,
                                       trig[i], logs[i])
        elif name == "B6":
            r = B.b6_residual_only(c, verifier, trig[i], rng)
        elif name == "B7":
            r = B.b7_oracle(c, verifier, trig[i])
        else:
            raise ValueError(name)
        r["case"] = c["id"]
        r["truth"] = c["location"]
        r["wall_s"] = time.time() - t0
        r["n_candidates"] = len(r.get("candidates", []) or [])
        return r

    if name in ("B6", "B7"):
        return [one(i) for i in range(len(cases))]
    w = 1 if LLM.MODELS[model][0] == "ollama" else workers
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(one, range(len(cases))))


def score(records, cmap):
    n = len(records)
    far = exact = cls = esc = ans = esc_ok = 0
    dsizes, lat = [], []
    for r in records:
        t = r["truth"]
        a = r.get("answer")
        dec = r.get("decision")
        if dec == "escalate":
            esc += 1
            # an escalation is useful to the operator when the admissible
            # set it hands over actually contains the truth
            if t in (r.get("candidates") or []):
                esc_ok += 1
        if a is not None:
            ans += 1
            if a == t:
                exact += 1
            if a in cmap.get(t, {t}):
                cls += 1
            else:
                far += 1
        if r.get("n_candidates"):
            dsizes.append(r["n_candidates"])
        lat.append(r.get("meta", {}).get("latency_s")
                   if isinstance(r.get("meta"), dict) else None)
    lat = [x for x in lat if x]
    return {
        "n": n,
        "FAR_phys": far / n,
        "exact_acc_all": exact / n,
        "class_acc_all": cls / n,
        "class_acc_answered": (cls / ans) if ans else float("nan"),
        "answered_rate": ans / n,
        "escalation_rate": esc / n,
        "useful_escalation_rate": esc_ok / n,
        # end-to-end: the operator is served correctly either by a correct
        # answer or by an escalation whose admissible set contains the truth
        "end_to_end_correct": (cls + esc_ok) / n,
        "mean_D": float(np.mean(dsizes)) if dsizes else float("nan"),
        "median_latency_s": float(np.median(lat)) if lat else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="kimi-k3")
    ap.add_argument("--baselines", default="B0,B5,B6,B7")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--per-location", type=int, default=4)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--budget-cny", type=float, default=40.0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cases, pres, bank, verifier, trig, det = setup(args.per_location, args.nf)
    cmap = P.class_map()
    # operations text for every case, at pre-declared information levels;
    # only B5L reads it
    from src.eval import oplogs as OL
    levels = OL.assign_levels(len(cases))
    logs = [OL.make_log(c["location"], lv, 900 + i)
            for i, (c, lv) in enumerate(zip(cases, levels))]
    print("cases %d ; residual triggers fired on %d of them"
          % (len(cases), int((trig.sum(1) > 0).sum())))

    b0 = LLM.balance_cny()
    print("moonshot balance before: %s" % b0)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    all_rows, raw, done = [], {}, set()
    ckpt = os.path.join(OUT, "E4_%s.json" % args.tag)
    if args.resume and os.path.exists(ckpt):
        prev = json.load(open(ckpt))
        all_rows = prev.get("rows", [])
        raw = prev.get("raw", {})
        done = set(raw)
        print("resuming, %d configs already complete" % len(done))

    def checkpoint():
        with open(ckpt, "w") as fh:
            json.dump({"rows": all_rows, "raw": raw, "alpha": ALPHA,
                       "n_cases": len(cases),
                       "classes": P.structural_classes(),
                   "log_levels": levels,
                       "usage": LLM.USAGE.summary(),
                       "balance_before": b0,
                       "balance_after": LLM.balance_cny()}, fh, indent=2)
    for bl in baselines:
        mods = ["-"] if bl in ("B6", "B7") else models
        for md in mods:
            for sd in range(args.seeds):
                key = "%s|%s|%d" % (bl, md, sd)
                if key in done:
                    print("%-3s %-14s seed %d  (already done, skipped)"
                          % (bl, md, sd), flush=True)
                    continue
                t0 = time.time()
                recs = run_config(bl, cases, pres, verifier, trig, det,
                                  md if md != "-" else models[0], sd,
                                  logs=logs)
                m = score(recs, cmap)
                m.update({"baseline": bl, "model": md, "seed": sd,
                          "wall_s": round(time.time() - t0, 1)})
                all_rows.append(m)
                raw[key] = [
                    {k: v for k, v in r.items()
                     if k in ("case", "truth", "answer", "decision",
                              "candidates", "rule", "n_candidates",
                              "wall_s")} for r in recs]
                print("%-3s %-14s seed %d  FAR %.3f  class %.3f  exact %.3f "
                      " esc %.3f  ans %.3f  %.0fs"
                      % (bl, md, sd, m["FAR_phys"], m["class_acc_all"],
                         m["exact_acc_all"], m["escalation_rate"],
                         m["answered_rate"], m["wall_s"]), flush=True)
                checkpoint()
                spent = LLM.USAGE.cost_cny()
                if spent > args.budget_cny:
                    print("budget cap reached (%.2f CNY), stopping" % spent)
                    break

    b1 = LLM.balance_cny()
    print("\nmoonshot balance after: %s  (est. spend %.3f CNY)"
          % (b1, LLM.USAGE.cost_cny()))
    print(json.dumps(LLM.USAGE.summary()["by_model"], indent=2))

    checkpoint()
    print("wrote %s" % ckpt)


if __name__ == "__main__":
    main()
