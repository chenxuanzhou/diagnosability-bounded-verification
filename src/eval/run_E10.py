"""
E10: latency and real-time feasibility.

The verification layer sits inside the control loop, so what matters is not
whether it is fast in the abstract but whether it fits inside the interval
between two samples of the plant it guards.  Three stages are timed
separately, because they have very different budgets and only the first two
are on the critical path:

  residual evaluation   the analytical redundancy relations over one window
  verification          standardise, threshold, and look up the admissible
                        set in the fault signature matrix
  agent call            the language model, which is off the critical path:
                        the verifier's decision does not wait for it

The Tennessee Eastman plant samples its composition analysers every three
minutes and its continuous measurements every 1.8 s in the reference
implementation; the robot logs at 100 Hz, a 10 ms budget.  Both are
reported.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval import cases as CS                            # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import fsm as FSM                         # noqa: E402
from src.verifier import residuals_tep as R                 # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

OUT = os.path.join("results", "E10")
os.makedirs(OUT, exist_ok=True)
ALPHA = E2.ALPHA


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ts),
            "p95_ms": float(np.percentile(ts, 95)),
            "n": n}


def llm_latency():
    """Per-call latency, read back from the raw call log."""
    path = os.path.join("logs", "llm")
    by_model = {}
    if not os.path.isdir(path):
        return by_model
    for fn in os.listdir(path):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(path, fn), encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:                           # noqa: BLE001
                    continue
                m = rec.get("meta", {})
                a, lat = m.get("alias"), m.get("latency_s")
                if a and lat:
                    by_model.setdefault(a, []).append(float(lat))
    return {k: {"n": len(v), "median_s": statistics.median(v),
                "p95_s": float(np.percentile(v, 95))}
            for k, v in by_model.items()}


def main():
    Xfit, mfit = E2.load("nominal_fit")
    Xcal, mcal = E2.load("nominal_cal")
    Ffit, _, _ = E2.features(Xfit, mfit, faulty=False)
    Fcal, _, _ = E2.features(Xcal, mcal, faulty=False)
    bank = CF.ConformalBank().fit_scale(Ffit).calibrate(Fcal)
    rnames, fnames, Mstruct, _ = FSM.build_fsm()
    Mloc = P.fsm_over_locations(Mstruct, fnames)
    V = CF.Verifier(Mloc, rnames, P.LOCATION_IDS)
    order = [R.RESIDUAL_NAMES.index(n) for n in rnames]

    cases, _ = CS.build_cases(2, 4, 0)
    W = cases[0]["window"]

    print("note: other jobs may be competing for the CPU; the figures are "
          "an upper bound on a quiet machine")
    print("hardware: %s, %s" % (platform.processor() or platform.machine(),
                                platform.platform()))
    res_t = timeit(lambda: R.residuals(W, with_scale=True), 50)
    win_t = timeit(lambda: CF.window_matrix(W, [0], E2.WIDTH), 50)
    F = CF.window_matrix(W, [0], E2.WIDTH)
    trig = bank.residual_triggers(F, ALPHA)[:, order]
    ver_t = timeit(lambda: V.candidates(trig[0]), 2000)
    thr_t = timeit(lambda: bank.residual_triggers(F, ALPHA), 500)

    print("\nstage                              median    p95")
    rows = [("residual evaluation, 60-sample window", res_t),
            ("window feature extraction", win_t),
            ("threshold and trigger", thr_t),
            ("admissible set lookup", ver_t)]
    for name, t in rows:
        print("  %-36s %7.3f ms %7.3f ms"
              % (name, t["median_ms"], t["p95_ms"]))
    online = win_t["median_ms"] + thr_t["median_ms"] + ver_t["median_ms"]
    print("  %-36s %7.3f ms" % ("verification path, total", online))

    # Per-tick cost.  The batch figure above recomputes the whole window on
    # every call, which is how the offline evaluation works but not how an
    # online implementation would: a streaming verifier evaluates the
    # relations once per new sample and updates the window statistics
    # incrementally.  The per-sample cost is what has to fit the log rate.
    W1 = W[:1]
    per_tick = timeit(lambda: R.residuals(W1, with_scale=True), 200)
    tick = per_tick["median_ms"] + thr_t["median_ms"] + ver_t["median_ms"]
    print("  %-36s %7.3f ms %7.3f ms"
          % ("streaming, per new sample", per_tick["median_ms"],
             per_tick["p95_ms"]))
    print("  %-36s %7.3f ms" % ("streaming path, total", tick))

    budgets = {"TEP composition analyser, 3 min": 180000.0,
               "TEP continuous measurement, 1.8 s": 1800.0,
               "robot log rate, 100 Hz": 10.0}
    print("")
    print("fraction of the control interval consumed, batch and streaming")
    for k, v in budgets.items():
        print("  %-36s %9.4f %%  %9.4f %%"
              % (k, 100.0 * online / v, 100.0 * tick / v))

    lat = llm_latency()
    print("\nagent call latency, from the raw call log (off the critical "
          "path)")
    for k, v in sorted(lat.items()):
        print("  %-20s n=%5d  median %6.2f s  p95 %6.2f s"
              % (k, v["n"], v["median_s"], v["p95_s"]))

    json.dump({"hardware": platform.platform(),
               "processor": platform.processor(),
               "stages": {n: t for n, t in rows},
               "verification_path_ms": online,
               "streaming_path_ms": tick,
               "per_tick_residual_ms": per_tick,
               "budgets_ms": budgets,
               "fraction_percent_batch": {k: 100.0 * online / v
                                          for k, v in budgets.items()},
               "fraction_percent_streaming": {k: 100.0 * tick / v
                                              for k, v in budgets.items()},
               "llm_latency": lat},
              open(os.path.join(OUT, "E10_latency.json"), "w"), indent=2)
    print("\nwrote %s" % os.path.join(OUT, "E10_latency.json"))


if __name__ == "__main__":
    main()
