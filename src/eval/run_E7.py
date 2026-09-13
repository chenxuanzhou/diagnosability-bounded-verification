"""
E7: invariance to model capability.

R1 is a statement about the plant's instrumentation, not about the agent.
If it is right, then swapping the language model should move the unverified
numbers a lot and leave the verified ambiguity untouched.  The strong form
of that is not a small variance but an identity: the admissible set D(y) is
computed from the residual pattern alone, so it cannot depend on which model
proposed the answer.  That is checked rather than assumed -- a non-zero
spread in |D(y)| across models would mean something is leaking from the
agent into the verifier, which is exactly the failure mode the plan warns
about.

What is reported:

  spread across models of the unverified false-acceptance rate
  spread across models of the verified false-acceptance rate
  spread across models of the mean admissible-set size, which must be zero

Read from the saved grid results, so it costs nothing to recompute.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.agents import llm as LLM                           # noqa: E402

OUT = os.path.join("results", "E7")
os.makedirs(OUT, exist_ok=True)


def load_rows(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in d.get("rows", []):
            r["_source"] = os.path.basename(p)
            rows.append(r)
    return rows


def agg(rows, baseline, metric):
    """metric per model, averaged over seeds, with the seed spread."""
    by = defaultdict(list)
    for r in rows:
        if r["baseline"] == baseline and r["model"] != "-":
            by[r["model"]].append(r[metric])
    return {m: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "n_seeds": len(v)} for m, v in by.items()}


def main(prefix=None):
    # Only files from one case set may be mixed: a spread across models is
    # meaningless if the models saw different cases.  The default is the
    # corrected 17-location set; pass a prefix to use another.
    prefix = prefix or (sys.argv[1] if len(sys.argv) > 1 else "E4_final")
    paths = [os.path.join("results", "E4", f)
             for f in sorted(os.listdir(os.path.join("results", "E4")))
             if f.startswith(prefix) and f.endswith(".json")
             and "backup" not in f and "ckpt" not in f]
    rows = load_rows(paths)
    if not rows:
        print("no grid results yet")
        return
    models = sorted({r["model"] for r in rows if r["model"] != "-"})
    print("case set: %s ; models present: %s" % (prefix, models))

    out = {"case_set": prefix, "models": models, "families": {}}
    for m in models:
        if m in LLM.MODELS:
            _, mid, fam, tier, _ = LLM.MODELS[m]
            out["families"][m] = {"model_id": mid, "family": fam,
                                  "tier": tier}

    print("\n%-14s %-10s %10s %10s %10s %10s"
          % ("baseline", "metric", "min", "max", "spread", "mean"))
    summary = {}
    for baseline in ("B0", "B1", "B2", "B3", "B4", "B5", "B5L"):
        for metric in ("FAR_phys", "class_acc_all", "mean_D",
                       "escalation_rate"):
            a = agg(rows, baseline, metric)
            vals = [v["mean"] for v in a.values()
                    if v["mean"] == v["mean"]]
            if len(vals) < 2:
                continue
            summary["%s|%s" % (baseline, metric)] = {
                "per_model": a, "min": float(min(vals)),
                "max": float(max(vals)),
                "spread": float(max(vals) - min(vals)),
                "mean": float(np.mean(vals))}
            print("%-14s %-10s %10.3f %10.3f %10.3f %10.3f"
                  % (baseline, metric, min(vals), max(vals),
                     max(vals) - min(vals), np.mean(vals)))

    key = []
    for metric in ("FAR_phys", "class_acc_all", "mean_D"):
        u = summary.get("B0|%s" % metric)
        for vb in ("B5", "B5L"):
            v = summary.get("%s|%s" % (vb, metric))
            if u and v:
                key.append({"metric": metric, "verified_baseline": vb,
                            "unverified_spread": u["spread"],
                            "verified_spread": v["spread"]})
    if key:
        print("\nspread across models, unverified against verified")
        for k in key:
            print("  %-16s %-4s unverified %.3f   verified %.3f"
                  % (k["metric"], k["verified_baseline"],
                     k["unverified_spread"], k["verified_spread"]))
    out["summary"] = summary
    out["key"] = key
    json.dump(out, open(os.path.join(OUT, "E7_invariance.json"), "w"),
              indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E7_invariance.json"))


if __name__ == "__main__":
    main()
