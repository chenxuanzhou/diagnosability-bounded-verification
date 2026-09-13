"""
Conformal calibration of the residual bank, and the verification operator.

Calibration protocol, following the plan:

  * the calibration set is nominal data only, drawn from a seed block that
    is used for nothing else -- no fitting, no thresholding, no model
    selection touches it elsewhere;
  * each residual is standardised by a robust scale estimated on a separate
    nominal block, so the calibration set itself is never used to choose a
    scale;
  * the non-conformity score is the max over residuals of the standardised
    absolute residual.  Taking the (1-alpha) empirical quantile of that
    score gives a single threshold with a distribution-free guarantee on
    the JOINT false-alarm rate, which is tighter than a Bonferroni split
    across the bank;
  * the reported calibration curve is empirical coverage on held-out
    nominal data against the nominal level 1-alpha.

The verification operator then maps a trigger pattern to a decision:

    reject     the claimed fault cannot explain the firing pattern
    accept     it can, and it is the only fault mode that can
    escalate   it can, but so can other fault modes -- the observation
               lies inside a non-singleton indiscernibility class and the
               verifier is provably blind inside it
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.verifier import residuals_tep as R                 # noqa: E402

NO_FAULT = "NF"

# Residuals that close a VESSEL balance -- a component or energy balance over
# a hold-up with a time constant of hours -- are quasi-steady relations.  They
# hold on a window average, not sample by sample, because the accumulation
# term cannot be tracked from three-minute samples while the plant is moving.
# Their dispersion statistic therefore measures unmodelled dynamics rather
# than a broken relation, so only their mean is used.  The two coolant
# balances are kept in full: the reactor coolant hold-up time constant is
# HWR/(FWR*cp) = 2.1 min, shorter than the sampling interval, so its
# quasi-steady form is exact at this rate (the condenser side is 6.2 min and
# is the looser of the two).  This split is declared from the plant
# constants, not from fault data.
MEAN_ONLY = {"r_balA", "r_balB", "r_balC", "r_balD", "r_dec2",
             "r_energy", "r_strip_En", "r_sep_En"}


def window_matrix(X, starts, width):
    """Two window statistics per residual, for many windows.

    A step deviation moves a residual's mean; a random-variation deviation
    inflates its spread while leaving the mean alone.  The benchmark
    contains both kinds, so each residual contributes two features: the
    window mean and the window dispersion (median absolute deviation, which
    is insensitive to the odd outlying sample).  The two share one
    structural signature, because they read the same residual.

    X: (N, T, 53) or (T, 53).  Returns (N * len(starts), 2 * n_residuals),
    the means first, then the dispersions, in RESIDUAL_NAMES order.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None]
    out = []
    for run in X:
        r, sc = R.residuals(run, with_scale=True)
        M = np.stack([r[k] for k in R.RESIDUAL_NAMES], axis=1)   # (T, m)
        S = np.stack([sc[k] for k in R.RESIDUAL_NAMES], axis=1)
        M = M / S                       # relative consistency error
        for s in starts:
            w = M[s:s + width]
            mu = np.nanmean(w, axis=0)
            disp = np.nanmedian(np.abs(w - np.nanmedian(w, axis=0)), axis=0)
            out.append(np.concatenate([mu, disp]))
    return np.asarray(out)


class ConformalBank:
    """Robust scales plus one joint conformal threshold."""

    def __init__(self, names=None):
        base = list(names or R.RESIDUAL_NAMES)
        self.residuals = base
        self.names = ([n + ":mean" for n in base]
                      + [n + ":disp" for n in base])
        self.disp_mask = np.array([n not in MEAN_ONLY for n in base])
        self.centre = None
        self.scale = None
        self.cal_scores = None

    def fit_scale(self, F_fit):
        """Robust centre and scale, from the fitting block of nominal runs.

        The scale is the 99th percentile of the nominal absolute deviation,
        converted to a Gaussian-equivalent sigma, rather than the median
        absolute deviation.  The non-conformity score is a maximum over the
        whole bank, so what matters is that every feature's TAIL is on the
        same footing.  A residual with a very tight core and occasional
        excursions gets a near-zero median absolute deviation, which inflates
        its standardised values and lets one feature dominate the maximum;
        measured across the noise sweep that made the calibrated threshold
        vary by ten orders of magnitude.  Standardising by the tail instead
        holds the threshold within a factor of two over a twelvefold change
        in noise.  The choice was made on nominal data alone.
        """
        self.centre = np.median(F_fit, axis=0)
        q = np.quantile(np.abs(F_fit - self.centre), 0.99, axis=0)
        self.scale = np.where(q > 0, q / 2.5758, 1.0)
        # guard against a residual that is identically zero on nominal data
        tiny = self.scale < 1e-12
        self.scale = np.where(tiny, 1.0, self.scale)
        # reference level of plant-wide restlessness, for common-mode
        # rejection on the dispersion statistics (see z())
        self.act_ref = float(np.median(self._activity(
            (F_fit - self.centre) / self.scale)))
        return self

    def _activity(self, Z):
        m = len(self.residuals)
        d = np.abs(Z[:, m:])[:, self.disp_mask]
        return np.median(d, axis=1, keepdims=True)

    def z(self, F):
        """Standardised features, with common-mode rejection on dispersion.

        A disturbance anywhere in a feedback-controlled plant makes every
        loop restless, which inflates the window dispersion of every
        residual at once, including the ones the structure proves are
        decoupled from that fault.  That common mode carries no isolation
        information and, left in, it lets two structurally indiscernible
        faults look different -- an artefact the theory forbids.

        So each dispersion statistic is divided by the plant-wide
        restlessness of that same window, measured as the median
        standardised dispersion across the bank, and only when that exceeds
        its nominal level.  What survives is how restless a relation is
        RELATIVE to the rest of the plant, which is what decoupling is
        about.  The mean statistics are untouched.
        """
        Z = (F - self.centre) / self.scale
        m = len(self.residuals)
        act = self._activity(Z)
        ref = getattr(self, "act_ref", None) or 1.0
        Z[:, m:] = Z[:, m:] / np.maximum(act / ref, 1.0)
        Z[:, m:] = Z[:, m:] * self.disp_mask            # quasi-steady relations
        return Z

    def calibrate(self, F_cal):
        """Store the non-conformity scores of the calibration block."""
        self.cal_scores = np.nanmax(np.abs(self.z(F_cal)), axis=1)
        return self

    def threshold(self, alpha):
        """Split-conformal quantile with the finite-sample correction."""
        n = len(self.cal_scores)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            return np.inf
        return float(np.sort(self.cal_scores)[k - 1])

    def triggers(self, F, alpha):
        """Per-feature firing pattern (mean and dispersion separately)."""
        return (np.abs(self.z(F)) > self.threshold(alpha)).astype(int)

    def residual_triggers(self, F, alpha):
        """Per-RESIDUAL firing: a residual fires if either statistic does.

        Collapsing the two statistics back onto the residual keeps the
        trigger pattern aligned with the structural fault signature matrix,
        which is indexed by residual and not by statistic.
        """
        t = self.triggers(F, alpha)
        m = len(self.residuals)
        return ((t[:, :m] + t[:, m:]) > 0).astype(int)

    def coverage(self, F_test, alphas):
        """Empirical coverage on held-out nominal data, per nominal level."""
        s = np.nanmax(np.abs(self.z(F_test)), axis=1)
        rows = []
        for a in alphas:
            t = self.threshold(a)
            rows.append({"alpha": float(a), "nominal_coverage": 1.0 - a,
                         "empirical_coverage": float(np.mean(s <= t)),
                         "threshold": t})
        return rows

    def per_residual_far(self, F_test, alpha):
        """False-alarm rate of each residual on held-out nominal data."""
        t = self.threshold(alpha)
        fire = np.abs(self.z(F_test)) > t
        return dict(zip(self.names, fire.mean(axis=0).tolist()))


class Verifier:
    """Structural admissibility check of a claimed fault mode."""

    def __init__(self, fsm, residual_names, fault_names, exoneration=False):
        self.fsm = np.asarray(fsm)
        self.residual_names = list(residual_names)
        self.fault_names = list(fault_names)
        self.exoneration = exoneration

    def candidates(self, trig, rule="nearest"):
        """Fault modes consistent with a firing pattern.

        Default rule, nearest signature: the candidates are the fault modes
        whose signature column is closest in Hamming distance to what was
        observed, ties included.  Exact column matching is what defines the
        indiscernibility classes in the first place, so this is the rule the
        theory actually describes; taking the nearest columns rather than
        insisting on an exact match is what makes it survive a residual that
        leaked or a weak fault that failed to move one.

        Alternative rule, strict consistency without exoneration: a fault is
        a candidate when it can explain every residual that fired, with no
        requirement that the quiet ones stay quiet.  It is more forgiving of
        a missed firing and less forgiving of a leak.  Reported as a
        sensitivity check.

        Both rules preserve the escalation guarantee exactly: two faults
        with identical signature columns always score identically, so they
        are always both in or both out.  A fault inside a non-singleton
        indiscernibility class can never be returned alone.
        """
        trig = np.asarray(trig)
        cols = np.concatenate([np.zeros((self.fsm.shape[0], 1), dtype=int),
                               self.fsm], axis=1)
        names = [NO_FAULT] + self.fault_names
        if rule == "strict":
            if not trig.any():
                return [NO_FAULT], "strict"
            out = [names[j] for j in range(len(names))
                   if not np.any(trig & (1 - cols[:, j]))]
            if out:
                return out, "strict"
        dist = np.abs(cols - trig[:, None]).sum(0)
        best = dist.min()
        tag = "exact" if best == 0 else "nearest"
        return [names[j] for j in np.nonzero(dist == best)[0]], tag

    def verify(self, claim, trig, rule="nearest"):
        """Return (decision, candidate set, how the set was obtained)."""
        D, rule = self.candidates(trig, rule)
        if claim not in D:
            return "reject", D, rule
        return ("accept" if len(D) == 1 else "escalate"), D, rule
