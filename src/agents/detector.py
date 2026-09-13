"""
A learned anomaly detector, for the B4 baseline.

Principal component analysis on normal operation only, with Hotelling's
T-squared inside the retained subspace and the squared prediction error
outside it, plus the per-variable contributions.  This is the standard
data-driven detector for this plant and is what "bolt a detector onto the
agent" means in practice.  Its control limits are set on the same nominal
calibration block the conformal layer uses, at the same level, so B4 and B5
are compared at equal false-alarm budget.
"""
from __future__ import annotations

import numpy as np


class PCADetector:
    def __init__(self, n_comp=None, var_target=0.90):
        self.n_comp = n_comp
        self.var_target = var_target

    def fit(self, X_nom, channels):
        """X_nom: (N, T, C) nominal runs."""
        self.channels = list(channels)
        flat = X_nom[:, 60:, :].reshape(-1, X_nom.shape[-1])
        self.mu = flat.mean(0)
        sd = flat.std(0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        self.keep = sd > 1e-9
        Z = (flat - self.mu) / self.sd
        Z = Z[:, self.keep]
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        ev = S ** 2 / (len(Z) - 1)
        frac = np.cumsum(ev) / ev.sum()
        a = self.n_comp or int(np.searchsorted(frac, self.var_target) + 1)
        self.a = a
        self.P = Vt[:a].T
        self.lam = np.maximum(ev[:a], 1e-12)
        return self

    def calibrate(self, X_cal, width, starts, alpha):
        """Control limits at the same level the conformal layer uses."""
        t2, spe = [], []
        for run in X_cal:
            for s in starts:
                a, b = self.scores(run[s:s + width])
                t2.append(a)
                spe.append(b)
        q = 100.0 * (1.0 - alpha)
        self.t2_lim = float(np.percentile(t2, q))
        self.spe_lim = float(np.percentile(spe, q))
        return self

    def scores(self, W):
        z = ((W.mean(0) - self.mu) / self.sd)[self.keep]
        t = z @ self.P
        t2 = float((t ** 2 / self.lam).sum())
        rec = self.P @ t
        spe = float(((z - rec) ** 2).sum())
        return t2, spe

    def contributions(self, W, top=6):
        z = ((W.mean(0) - self.mu) / self.sd)[self.keep]
        t = z @ self.P
        rec = self.P @ t
        res = (z - rec) ** 2
        names = [c for c, k in zip(self.channels, self.keep) if k]
        order = np.argsort(-res)[:top]
        return [(names[i], float(res[i])) for i in order]

    def report(self, W):
        t2, spe = self.scores(W)
        flag = (t2 > self.t2_lim) or (spe > self.spe_lim)
        contrib = self.contributions(W)
        lines = ["anomaly flag: %s" % ("YES" if flag else "no"),
                 "T2 = %.1f (limit %.1f), SPE = %.2f (limit %.2f)"
                 % (t2, self.t2_lim, spe, self.spe_lim),
                 "largest residual contributions: "
                 + ", ".join("%s %.2f" % c for c in contrib)]
        return "\n".join(lines)
