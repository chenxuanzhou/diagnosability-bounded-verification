"""
Evaluation cases, and how a case is shown to an agent.

A case is one three-hour window of plant data with a known ground-truth
fault location.  The same window is shown to every baseline, so the only
thing that differs across the ladder is what the agent is allowed to do
with it.

The textual view gives, for each of the 53 logged channels, the window
mean, its deviation from nominal in units of the nominal standard
deviation, and the ratio of the window's spread to the nominal spread.
Both statistics are included because the benchmark contains step
deviations, which move a mean, and random-variation deviations, which
inflate a spread.  The verifier is given exactly the same two statistics of
its residuals, so neither side is handed information the other lacks.

Channels are listed in fixed instrument order, not sorted by deviation, so
the presentation itself carries no ranking hint.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.agents import catalogue as CAT                     # noqa: E402
from src.eval import propositions as P                      # noqa: E402
from src.verifier import conformal as CF                    # noqa: E402
from src.verifier import run_E2 as E2                       # noqa: E402

DATA = os.path.join("data", "tep")
WIDTH = E2.WIDTH
IDV_ON = E2.IDV_ON
SETTLE = E2.SETTLE


def _load(tag):
    X = np.load(os.path.join(DATA, "%s_X.npy" % tag))
    meta = json.load(open(os.path.join(DATA, "%s_meta.json" % tag)))
    return X, meta


class Presenter:
    """Turns a raw window into the numbers an agent is shown."""

    def __init__(self, X_nom):
        flat = X_nom[:, 60:, :].reshape(-1, X_nom.shape[-1])
        self.mu = flat.mean(0)
        self.sd = flat.std(0)
        med = np.median(flat, axis=0)
        self.disp = np.median(np.abs(flat - med), axis=0)
        self.ok = self.sd > 1e-9

    def stats(self, W):
        mean = W.mean(0)
        med = np.median(W, axis=0)
        disp = np.median(np.abs(W - med), axis=0)
        z = np.zeros_like(mean)
        z[self.ok] = (mean[self.ok] - self.mu[self.ok]) / self.sd[self.ok]
        ratio = np.ones_like(mean)
        good = self.disp > 1e-9
        ratio[good] = disp[good] / self.disp[good]
        return mean, z, ratio

    def text(self, W):
        mean, z, ratio = self.stats(W)
        lines = ["channel   description                                 "
                 "window mean   deviation   spread vs normal"]
        for i, ch in enumerate(CAT.CHANNELS):
            lines.append("%-9s %-42s %12.4g %+10.2f sd %11.2fx"
                         % (ch, CAT.CHANNEL_DESC.get(ch, ""), mean[i],
                            z[i], ratio[i]))
        return "\n".join(lines)


def build_cases(n_per_location=4, n_nf=8, seed=0):
    """One balanced set of cases, drawn deterministically from the seed."""
    rng = np.random.default_rng(seed)
    Xf, mf = _load("fault")
    Xn, mn = _load("nominal_test")
    Xfit, _ = _load("nominal_fit")
    pres = Presenter(Xfit)

    cases = []
    T = Xf.shape[1]
    by_loc = {}
    for i, m in enumerate(mf):
        idv = m["fault"]
        if idv not in P.IDV_TO_LOCATION:
            continue
        end = E2._usable_end(m, T)
        starts = [s for s in range(IDV_ON + SETTLE, end - WIDTH + 1, 60)]
        if not starts:
            continue
        by_loc.setdefault(P.location_of(idv), []).append((i, starts))

    for loc in P.LOCATION_IDS:
        pool = by_loc.get(loc, [])
        if not pool:
            continue
        picks = []
        # spread the draws over distinct runs before reusing a run
        order = rng.permutation(len(pool))
        k = 0
        while len(picks) < n_per_location and k < 4 * n_per_location:
            i, starts = pool[order[k % len(order)]]
            s = starts[int(rng.integers(len(starts)))]
            if (i, s) not in picks:
                picks.append((i, s))
            k += 1
        for k, (i, s) in enumerate(picks):
            W = Xf[i, s:s + WIDTH]
            # the index keeps ids unique even if the same (run, window) is
            # drawn twice; duplicate ids would silently collapse cases in
            # any per-case aggregation
            cases.append({"id": "%s_r%d_s%d_%d" % (loc, i, s, k),
                          "location": loc,
                          "idv": int(mf[i]["fault"]), "split": "fault",
                          "run": int(i), "start": int(s), "window": W,
                          "seed_run": int(mf[i]["seed"])})

    Tn = Xn.shape[1]
    nstarts = list(range(60, Tn - WIDTH + 1, 60))
    for j in range(n_nf):
        i = int(rng.integers(len(Xn)))
        s = int(nstarts[int(rng.integers(len(nstarts)))])
        cases.append({"id": "NF_r%d_s%d_%d" % (i, s, j),
                      "location": P.NO_FAULT,
                      "idv": 0, "split": "nominal", "run": int(i),
                      "start": s, "window": Xn[i, s:s + WIDTH],
                      "seed_run": int(mn[i]["seed"])})
    return cases, pres


def case_features(cases):
    """Residual window features for every case, in case order."""
    F = []
    for c in cases:
        F.append(CF.window_matrix(c["window"], [0], WIDTH)[0])
    return np.asarray(F)
