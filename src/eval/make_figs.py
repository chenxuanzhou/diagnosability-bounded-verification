"""Figures for the paper, from the saved result files."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
# Figures are reduced to one or two columns of a two-column journal page, so
# the axis furniture is set larger than the default: at print size a 9 pt
# label in a 3.2 in panel scaled to a 3.4 in column is legible, but tick
# labels at the default 0.8 scale are not.
plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9,
                     "grid.alpha": 0.3, "figure.dpi": 200,
                     # TrueType rather than Type 3 fonts in PDF output:
                     # IEEE PDF eXpress rejects Type 3 fonts
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight"})


def fig_ablation():
    """Three lines, because the middle one is the achievability result.

    Plotting only the bound against the empirical partition reads as R2
    failing.  It is not: at the full sensor set the implemented bank's own
    structural partition sits exactly on the bound, and the empirical
    partition differs from it only by detection power.  Below full
    instrumentation the bank diverges from the bound, and the divergence
    tracks how many residuals survive, which is a statement about residual
    availability rather than about the plant.
    """
    import numpy as np
    d = json.load(open("results/E6/E6_ablation.json"))
    rows = [r for r in d["rows"] if r.get("empirical")]
    n = [r["n_sensors"] for r in rows]
    removed = [41 - x for x in n]
    theo = [r["theoretical"]["indiscernible_mass"] for r in rows]
    bank = [r["realised_structural"]["indiscernible_mass"] for r in rows]
    emp = [r["empirical"]["empirical_mass"] for r in rows]
    cons = [r["empirical"].get("empirical_mass_consistent", np.nan)
            for r in rows]
    det = [r["empirical"]["mean_detection"] for r in rows]
    nres = [r["empirical"]["n_residuals"] for r in rows]
    dsz = [r["empirical"]["mean_D"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.6))

    a0 = ax[0]
    a0.plot(removed, theo, "o-", color="#1b4965", lw=2,
            label="structural bound")
    a0.plot(removed, bank, "s--", color="#7a9e7e",
            label="implemented bank, structural")
    a0.plot(removed, cons, "^:", color="#c1666b",
            label="empirical, decoupling-consistent")
    a0.plot(removed, emp, "x", color="#c1666b", ms=6, mew=1.2, ls="none",
            label="empirical, as measured (leak included)")
    # the one rung where the as-measured partition is finer than the bank's
    # own structural partition: the leak, see report 1.8
    for xr, e, b in zip(removed, emp, bank):
        if e < b - 1e-9:
            a0.annotate("leak", xy=(xr, e), xytext=(xr + 2.5, e - 0.13),
                        fontsize=7, color="#c1666b",
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color="#c1666b"))
    a0.annotate("bank = bound", xy=(removed[0], bank[0]),
                xytext=(removed[0] + 3, bank[0] - 0.17), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    a0.set_xlabel("instruments removed")
    a0.set_ylabel("indiscernible mass")
    a0.set_title("(a) indiscernible mass")
    # legend above the axes so it can never sit on the annotation
    a0.legend(frameon=False, fontsize=6.6, loc="lower center",
              bbox_to_anchor=(0.5, 1.08), ncol=2, handlelength=2.2)
    a0.set_ylim(0, 1)
    a1 = a0.twinx()
    a1.plot(removed, nres, "v-", color="#9aa5ab", lw=0.9, ms=4)
    a1.set_ylabel("residuals available", color="#6d7a82", fontsize=8)
    a1.tick_params(axis="y", labelcolor="#6d7a82", labelsize=8)
    a1.grid(False)

    ax[1].plot(removed, det, "o-", color="#1b4965")
    ax[1].set_xlabel("instruments removed")
    ax[1].set_ylabel("mean detection rate")
    ax[1].set_title("(b) detection")
    ax[1].set_ylim(0, 1)

    ax[2].plot(removed, dsz, "o-", color="#1b4965")
    ax[2].set_xlabel("instruments removed")
    ax[2].set_ylabel("mean admissible set size")
    ax[2].set_title("(c) admissible-set size")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.png"))
    fig.savefig(os.path.join(OUT, "fig_ablation.pdf"))
    print("wrote figs/fig_ablation.png")


def fig_tightness():
    """Two effects that pull in opposite directions as deviations grow.

    The curve is computed over one fixed population of fault locations, the
    ones with usable windows at every level, so any two points are
    comparable.  The weak-deviation-only extension is drawn as its own
    series and never against this curve.
    """
    import numpy as np
    d = json.load(open("results/E5/E5_bound_tightness.json"))
    mag = d["sweeps"]["magnitude"]
    rows = mag["levels"]
    lv = [r["level"] for r in rows]
    gap = [r["merge_gap"] for r in rows]
    viol = [r["decoupling_violation"] for r in rows]
    det = [r["mean_detection"] for r in rows]
    npop = mag["levels"][0]["n_locations_common"]

    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    a0 = ax[0]
    a0.semilogx(lv, gap, "o-", color="#1b4965", lw=2,
                label="merge gap above the bound")
    a0.semilogx(lv, viol, "s--", color="#c1666b",
                label="decoupling violation")
    a0.axhline(0.0, color="k", lw=0.8, ls=":")
    a0.set_xlabel("deviation magnitude, relative to benchmark")
    a0.set_ylabel("fraction of fault locations")
    a0.set_title("(a) merge gap and decoupling violation, %d locations" % npop)
    a0.legend(frameon=False, fontsize=7.5, loc="center left")
    a0.set_ylim(-0.03, 0.65)
    # one label per swept level and no minor-tick labels; the default
    # log formatter stacked 2x10^-1, 3x10^-1, ... into an unreadable run
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    a0.xaxis.set_major_locator(FixedLocator(lv))
    a0.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "x"))
    a0.xaxis.set_minor_formatter(NullFormatter())
    a0.tick_params(axis="x", labelsize=7.5)
    # mark the crossing, which is the finding
    cross = [l for l, g, v in zip(lv, gap, viol) if abs(g - v) < 1e-9]
    if cross:
        a0.axvline(cross[-1], color="#9aa5ab", lw=0.8, ls="--")
        a0.annotate("curves cross", xy=(cross[-1], 0.273),
                    xytext=(cross[-1] * 1.35, 0.50), fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.6))

    curves = d["weak_detection_curves"]
    styles = {"L_Dfeed_temp": ("o-", "#1b4965", "D feed temperature"),
              "L_Cfeed_temp": ("s--", "#c1666b", "A and C feed temperature"),
              "L_cond_cw_valve": ("^:", "#7a9e7e",
                                  "condenser coolant valve (deadband)")}
    for key, (st, col, lab) in styles.items():
        pts = []
        for k, v in curves.get(key, {}).items():
            fam, level = k.split("@")
            if key == "L_cond_cw_valve" and fam != "stiction":
                continue
            if key != "L_cond_cw_valve" and fam == "stiction":
                continue
            pts.append((float(level), v))
        pts = sorted(set(pts))
        if pts:
            ax[1].semilogx([p[0] for p in pts], [p[1] for p in pts],
                           st, color=col, label=lab, ms=4)
    ax[1].set_xlabel("deviation magnitude, relative to benchmark")
    ax[1].set_ylabel("detection rate")
    ax[1].set_title("(b) detection of weak deviations")
    ax[1].set_ylim(-0.03, 1.05)
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tightness.png"))
    fig.savefig(os.path.join(OUT, "fig_tightness.pdf"))
    print("wrote figs/fig_tightness.png")


def fig_calibration():
    d = json.load(open("results/E2/E2_calibration.json"))
    cov = d["coverage"]
    nom = [r["nominal_coverage"] for r in cov]
    emp = [r["empirical_coverage"] for r in cov]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.plot([0.75, 1.0], [0.75, 1.0], "k:", lw=0.8)
    ax.plot(nom, emp, "o-", color="#1b4965")
    ax.set_xlabel("nominal coverage, 1 - alpha")
    ax.set_ylabel("empirical coverage")
    ax.set_title("conformal calibration")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_calibration.png"))
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf"))
    print("wrote figs/fig_calibration.png")


def fig_complementarity():
    """Within-subject: every case is run under every condition.

    Condition B never sees a log, so it has no level of its own and is drawn
    as a horizontal reference.  That is the point of the design: any
    gradient in condition C is attributable to the text, because the cases,
    the window, the admissible set, the model and the seed are all held
    fixed across the three levels.
    """
    import numpy as np
    path = "results/E8/E8_within.json"
    if not os.path.exists(path):
        print("E8 within-subject not run yet")
        return
    d = json.load(open(path))
    rows = d["rows"]
    chance = d["chance"]
    levels = ["irrelevant", "weak", "strong"]

    def get(cond, lv=None):
        v = [r["accuracy"] for r in rows if r["condition"] == cond
             and r["level"] == lv]
        return (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if v else 0.0)

    b_mu, b_sd = get("B")
    c = {lv: get("C", lv) for lv in levels}

    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2))
    a0 = ax[0]
    vals = [chance, b_mu] + [c[lv][0] for lv in levels]
    errs = [0.0, b_sd] + [c[lv][1] for lv in levels]
    cols = ["#9aa5ab", "#c1666b", "#b8b8b8", "#5b8ba0", "#1b4965"]
    a0.bar(range(5), vals, yerr=errs, capsize=3, color=cols)
    nl = chr(10)
    a0.set_xticks(range(5))
    a0.set_xticklabels(["verifier" + nl + "alone",
                        "+ agent," + nl + "no log",
                        "+ log," + nl + "irrelevant",
                        "+ log," + nl + "weak",
                        "+ log," + nl + "strong"], fontsize=7.5)
    a0.axhline(chance, color="k", ls=":", lw=0.8)
    a0.set_ylabel("accuracy inside the class")
    a0.set_title("(a) accuracy by condition")
    a0.set_ylim(0, 1.05)

    x = np.arange(3)
    ax[1].axhline(b_mu, color="#c1666b", ls="--", lw=1.4,
                  label="no log")
    ax[1].fill_between([-0.5, 2.5], b_mu - b_sd, b_mu + b_sd,
                       color="#c1666b", alpha=0.15)
    ax[1].errorbar(x, [c[lv][0] for lv in levels],
                   yerr=[c[lv][1] for lv in levels], fmt="o-",
                   color="#1b4965", capsize=3, label="with the log")
    ax[1].axhline(chance, color="k", ls=":", lw=0.8, label="chance")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(levels)
    ax[1].set_xlim(-0.5, 2.5)
    ax[1].set_xlabel("information content of the log")
    ax[1].set_ylabel("accuracy inside the class")
    ax[1].set_title("(b) accuracy by log information level")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_complementarity.png"))
    fig.savefig(os.path.join(OUT, "fig_complementarity.pdf"))
    print("wrote figs/fig_complementarity.png")


def fig_invariance():
    """Grouped bars on a categorical axis, three series.

    An earlier version ordered the models along the x axis by the very
    metric being plotted, which makes the unverified line monotone by
    construction rather than by observation.  Models are a categorical axis
    in the order the report's table uses.

    Two defects the second review found are fixed here.  The bar offset was
    indexed by position in the declared series list rather than in the list
    of series actually present, so when one series was missing the others
    shifted a slot to the right and the per-model pairing was wrong.  And the
    escalating pipeline B5, whose cross-model spread is exactly zero, was not
    drawn at all; it comes from the stratified table, where it is
    reconstructed from the admissible sets, because the hosted grid never ran
    it as a separate configuration.
    """
    p = "results/E7/E7_invariance.json"
    if not os.path.exists(p):
        print("E7 not run yet")
        return
    import numpy as np
    d = json.load(open(p))
    s = d["summary"]
    order = ["kimi-k3", "kimi-k2.6", "qwen3.5-9b", "llama3.1-8b",
             "qwen2.5-7b"]

    # B5 per model, nearest rule, from the reconstructed pipeline
    b5 = {}
    q = "results/strata/strata_final.json"
    if os.path.exists(q):
        rows = [r for r in json.load(open(q))["pipeline_by_rule"]
                if r["rule"] == "nearest"]
        by = {}
        for r in rows:
            by.setdefault(r["model"], []).append(r["FAR_phys"])
        b5 = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
              for m, v in by.items()}

    series = [("unverified (B0)", "#c1666b", s["B0|FAR_phys"]["per_model"]),
              ("verified, escalating (B5)", "#1b4965", b5),
              ("verified + ops context (B5L)", "#7a9e7e",
               s["B5L|FAR_phys"]["per_model"])]
    series = [x for x in series if x[2]]
    models = [m for m in order if any(m in per for _, _, per in series)]
    x = np.arange(len(models))
    w = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (lab, col, per) in enumerate(series):
        vals = [per[m]["mean"] if m in per else np.nan for m in models]
        errs = [per[m]["sd"] if m in per else 0.0 for m in models]
        off = (i - (len(series) - 1) / 2.0) * w
        ax.bar(x + off, vals, w * 0.92, yerr=errs, capsize=2.5,
               label=lab, color=col)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        print("  %-30s spread across models %.4f" % (lab, spread))
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("FAR_phys")
    ax.set_ylim(0, 1.18)
    # the spreads, written on the figure so the zero is visible
    sp = {lab: float(np.nanmax([per[m]["mean"] for m in models if m in per])
                     - np.nanmin([per[m]["mean"] for m in models if m in per]))
          for lab, _, per in series}
    txt = "\n".join("%s: spread %.4f" % (lab.split(" (")[1].rstrip(")"), v)
                    for lab, v in sp.items())
    ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, family="monospace")
    ax.set_title("physical false acceptance by model")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_invariance.png"))
    fig.savefig(os.path.join(OUT, "fig_invariance.pdf"))
    print("wrote figs/fig_invariance.png")


if __name__ == "__main__":
    fig_ablation()
    fig_tightness()
    fig_calibration()
    fig_complementarity()
    fig_invariance()
