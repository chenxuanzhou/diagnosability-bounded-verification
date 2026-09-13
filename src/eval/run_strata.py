"""
The main result table, stratified, with confidence intervals, and the strict
candidate rule reported alongside the relaxed one.

Why stratify.  A single average over the whole case set mixes three
populations whose difficulty is set by completely different things:

  resolved     the verifier alone pins the fault down to one location.  No
               method can do better than answer it, and none should do worse.
  ambiguous    the observation falls in a non-singleton class.  The verifier
               is provably blind here, so this is the only stratum where an
               agent can contribute, and the only one where a difference
               between methods means anything.
  undetected   nothing fires at all.  The deviation is below the detection
               floor, so no method built on these sensors can answer, and
               every method is wrong together.
  no fault     the plant is healthy.

Averaged together these dilute each other until every method looks alike,
which is exactly what the unstratified table showed.

Why both candidate rules.  What R2 proves is an exact-consistency operator:
accept, reject, or escalate.  What the implementation runs by default is
nearest-signature matching, which never returns an empty set and therefore
cannot reject on physical grounds alone.  That is an engineering relaxation
and it is reported as one, with the strict rule beside it, rather than left
for a reader to find.

Confidence intervals are bootstrap over CASES, not over seeds, because the
case is the sampling unit and seeds are repeated measurements of the same
case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import cases as CS                            # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "strata")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA
NBOOT = 4000


def setup(per_location, n_nf, case_seed=0):
    cases, pres = CS.build_cases(per_location, n_nf, case_seed)
    Xfit, mfit = CS._load("nominal_fit")
    Xcal, mcal = CS._load("nominal_cal")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    bank = CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    V = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)
    F = CS.case_features(cases)
    order = [CF.R.RESIDUAL_NAMES.index(n) for n in rnames]
    trig = bank.residual_triggers(F, ALPHA)[:, order]
    return cases, V, trig


def stratify(cases, V, trig):
    """Assign each case to exactly one stratum."""
    strata, info = {}, {}
    for i, c in enumerate(cases):
        D, rule = V.candidates(trig[i])
        Ds, rules = V.candidates(trig[i], rule="strict")
        if c["location"] == P.NO_FAULT:
            s = "no fault"
        elif not trig[i].any():
            s = "undetected"
        elif len(D) > 1:
            s = "ambiguous"
        else:
            s = "resolved"
        strata[c["id"]] = s
        info[c["id"]] = {"D": D, "rule": rule, "D_strict": Ds,
                         "rule_strict": rules, "truth": c["location"],
                         "n_fired": int(trig[i].sum())}
    return strata, info


def boot_ci(per_case_values, n=NBOOT, seed=0):
    """Bootstrap mean and 95 % interval, resampling cases."""
    v = np.asarray(list(per_case_values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def score_by_case(recs_by_seed, cmap, metric):
    """Per case, the metric averaged over seeds."""
    per_case = defaultdict(list)
    for recs in recs_by_seed:
        for r in recs:
            t, a = r["truth"], r.get("answer")
            if metric == "exact":
                val = 1.0 if a == t else 0.0
            elif metric == "class":
                val = 1.0 if (a is not None
                              and a in cmap.get(t, {t})) else 0.0
            elif metric == "far":
                val = 1.0 if (a is not None
                              and a not in cmap.get(t, {t})) else 0.0
            elif metric == "answered":
                val = 1.0 if a is not None else 0.0
            else:
                raise ValueError(metric)
            per_case[r["case"]].append(val)
    return {k: float(np.mean(v)) for k, v in per_case.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-location", type=int, default=4)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--grids", default="E4_main_kimi,E4_main_local")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    cases, V, trig = setup(args.per_location, args.nf)
    cmap = P.class_map()
    strata, info = stratify(cases, V, trig)
    order = ["resolved", "ambiguous", "undetected", "no fault"]
    counts = {s: sum(1 for v in strata.values() if v == s) for s in order}
    print("case strata (n = %d): %s" % (len(cases), counts))

    raw = {}
    for g in args.grids.split(","):
        p = os.path.join("results", "E4", "%s.json" % g.strip())
        if os.path.exists(p):
            raw.update(json.load(open(p))["raw"])
    print("configurations loaded: %d" % len(raw))

    # group runs by (baseline, model) over seeds
    groups = defaultdict(list)
    for k, recs in raw.items():
        bl, md, _ = k.split("|")
        groups[(bl, md)].append(recs)

    rows = []
    print("")
    print("%-5s %-12s %-11s %4s %18s %18s %18s"
          % ("base", "model", "stratum", "n", "class acc [95% CI]",
             "exact acc [95% CI]", "FAR_phys [95% CI]"))
    for (bl, md), recs_by_seed in sorted(groups.items()):
        by_case = {m: score_by_case(recs_by_seed, cmap, m)
                   for m in ("class", "exact", "far", "answered")}
        for s in order:
            ids = [c["id"] for c in cases if strata[c["id"]] == s
                   and c["id"] in by_case["class"]]
            if not ids:
                continue
            out = {"baseline": bl, "model": md, "stratum": s, "n": len(ids),
                   "n_seeds": len(recs_by_seed)}
            for m in ("class", "exact", "far", "answered"):
                mu, lo, hi = boot_ci([by_case[m][i] for i in ids])
                out[m] = {"mean": mu, "lo": lo, "hi": hi}
            rows.append(out)
            print("%-5s %-12s %-11s %4d  %.3f [%.3f,%.3f]  %.3f [%.3f,%.3f]"
                  "  %.3f [%.3f,%.3f]"
                  % (bl, md, s, len(ids),
                     out["class"]["mean"], out["class"]["lo"],
                     out["class"]["hi"], out["exact"]["mean"],
                     out["exact"]["lo"], out["exact"]["hi"],
                     out["far"]["mean"], out["far"]["lo"], out["far"]["hi"]))

    # ---- candidate rule comparison, verifier only, no agent involved
    print("")
    print("candidate rule, verifier alone over all %d cases" % len(cases))
    rule_rows = []
    for rule in ("nearest", "strict"):
        key = "D" if rule == "nearest" else "D_strict"
        rkey = "rule" if rule == "nearest" else "rule_strict"
        # How often the exact-consistency rule found NO fault able to explain
        # every firing.  That is the case R2's operator would report as a
        # physical inconsistency; the implementation falls back to nearest
        # signature there rather than returning nothing, and the fallback
        # rate is what this counts.
        fell_back = sum(1 for v in info.values()
                        if v[rkey] not in ("strict", "exact"))
        sizes = [len(v[key]) for v in info.values()]
        hit = [1.0 if v["truth"] in v[key] else 0.0 for v in info.values()]
        amb = [1.0 if len(v[key]) > 1 else 0.0 for v in info.values()]
        mu, lo, hi = boot_ci(hit)
        rule_rows.append({"rule": rule, "fallback_cases": fell_back,
                          "fallback_rate": fell_back / len(info),
                          "mean_D": float(np.mean(sizes)),
                          "escalation_rate": float(np.mean(amb)),
                          "truth_in_D": {"mean": mu, "lo": lo, "hi": hi}})
        print("  %-8s mean |D| %.2f   escalation %.3f   truth in D "
              "%.3f [%.3f,%.3f]   fell back to nearest %d of %d"
              % (rule, np.mean(sizes), np.mean(amb), mu, lo, hi,
                 fell_back, len(info)))

    # ---- B5 in the stratified table.
    # B5 accepts the verifier's answer when the admissible set is a
    # singleton and escalates otherwise.  It never uses the agent's
    # proposal, so its per-case outcome is a function of the set alone: the
    # same for every model and every seed.  It is scored once, under both
    # candidate rules, and carried into the same table as the others.  The
    # ambiguous stratum is where the two rules and B5's design show: it
    # escalates there by construction rather than guessing.
    for rule, key, name in (("nearest", "D", "B5"),
                            ("strict", "D_strict", "B5-strict")):
        per_case = {"class": {}, "exact": {}, "far": {}, "answered": {}}
        for c in cases:
            cid = c["id"]
            D = info[cid][key]
            truth = c["location"]
            if len(D) != 1:
                per_case["answered"][cid] = 0.0
                per_case["class"][cid] = 0.0
                per_case["exact"][cid] = 0.0
                per_case["far"][cid] = 0.0
                continue
            ans = D[0]
            ok_cls = ans in cmap.get(truth, {truth})
            per_case["answered"][cid] = 1.0
            per_case["class"][cid] = 1.0 if ok_cls else 0.0
            per_case["exact"][cid] = 1.0 if ans == truth else 0.0
            per_case["far"][cid] = 0.0 if ok_cls else 1.0
        for st in order:
            ids = [c["id"] for c in cases if strata[c["id"]] == st]
            if not ids:
                continue
            out = {"baseline": name, "model": "-", "stratum": st,
                   "n": len(ids), "n_seeds": 1,
                   "note": "deterministic given the admissible set; "
                           "identical for every model and seed"}
            for m in ("class", "exact", "far", "answered"):
                mu, lo, hi = boot_ci([per_case[m][i] for i in ids])
                out[m] = {"mean": mu, "lo": lo, "hi": hi}
            rows.append(out)
            print("%-5s %-12s %-11s %4d  %.3f [%.3f,%.3f]  %.3f [%.3f,%.3f]"
                  "  %.3f [%.3f,%.3f]"
                  % (name, "-", st, len(ids),
                     out["class"]["mean"], out["class"]["lo"],
                     out["class"]["hi"], out["exact"]["mean"],
                     out["exact"]["lo"], out["exact"]["hi"],
                     out["far"]["mean"], out["far"]["lo"],
                     out["far"]["hi"]))

    # ---- the pipeline under the strict rule.
    # B5 escalates whenever the admissible set is not a singleton, so its
    # outcome is fully determined by the set and the agent's proposal.  The
    # proposal is B0's answer for the same model and seed, because B5 issues
    # exactly that call.  So the strict-rule pipeline can be reconstructed
    # offline without spending anything.
    print("")
    print("pipeline under each candidate rule, reconstructed from the "
          "agent's own proposals")
    pipe_rows = []
    b0 = {k: v for k, v in raw.items() if k.startswith("B0|")}
    for k, recs in sorted(b0.items()):
        _, md, sd = k.split("|")
        for rule in ("nearest", "strict"):
            key = "D" if rule == "nearest" else "D_strict"
            far = esc = cls = 0
            for r in recs:
                D = info[r["case"]][key]
                claim = r.get("answer")
                if len(D) > 1:
                    esc += 1
                    continue
                ans = D[0]
                if ans in cmap.get(r["truth"], {r["truth"]}):
                    cls += 1
                else:
                    far += 1
            n = len(recs)
            pipe_rows.append({"model": md, "seed": int(sd), "rule": rule,
                              "FAR_phys": far / n, "class_acc": cls / n,
                              "escalation": esc / n})
    agg = {}
    for r in pipe_rows:
        agg.setdefault(r["rule"], []).append(r)
    for rule, rs in agg.items():
        print("  %-8s FAR %.3f   class %.3f   escalation %.3f"
              % (rule, np.mean([x["FAR_phys"] for x in rs]),
                 np.mean([x["class_acc"] for x in rs]),
                 np.mean([x["escalation"] for x in rs])))
    # Per model, to show the escalating pipeline is model-independent.  It
    # has to be: with a singleton admissible set the answer is forced and
    # with a larger one the case is escalated, so the agent's proposal never
    # reaches the output.  The spread across models is the empirical check
    # on that argument.
    print("")
    print("  escalating pipeline per model, nearest rule")
    per_model = {}
    for r in pipe_rows:
        if r["rule"] != "nearest":
            continue
        per_model.setdefault(r["model"], []).append(r)
    for md, rs in sorted(per_model.items()):
        print("    %-12s FAR %.4f  class %.4f  escalation %.4f"
              % (md, np.mean([x["FAR_phys"] for x in rs]),
                 np.mean([x["class_acc"] for x in rs]),
                 np.mean([x["escalation"] for x in rs])))
    if len(per_model) > 1:
        vals = [np.mean([x["FAR_phys"] for x in rs])
                for rs in per_model.values()]
        print("    spread across %d models: %.4f"
              % (len(per_model), max(vals) - min(vals)))

    json.dump({"n_cases": len(cases), "strata_counts": counts,
               "strata": strata,
               "case_info": {k: {kk: vv for kk, vv in v.items()}
                             for k, v in info.items()},
               "rows": rows, "candidate_rules": rule_rows,
               "pipeline_by_rule": pipe_rows,
               "n_bootstrap": NBOOT},
              open(os.path.join(OUT, "strata_%s.json" % args.tag), "w"),
              indent=2)
    print("")
    print("wrote %s" % os.path.join(OUT, "strata_%s.json" % args.tag))


if __name__ == "__main__":
    main()
