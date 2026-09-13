"""
Analytical redundancy relations for the Tennessee Eastman plant.

Every relation below is an exact rearrangement of an equation in teprob.f;
no coefficient is fitted.  Each residual is annotated with the structural
equations it consumes, which is what determines its fault signature -- the
signature is read off the structural model rather than asserted by hand, so
the implemented verifier and the theory cannot drift apart silently.

Two measurement inversions make the plant far more observable than it first
appears and are what let the energy balances close analytically:

  separator VLE inversion   the separator is at equilibrium, so the liquid
      composition follows from the measured vapour composition, pressure
      and temperature.  That yields the liquid density, hence the molar
      underflow from the volumetric measurement, hence the separator's
      total and component balances.
  separator balance         those give the reactor effluent flow and
      composition, which are not measured anywhere, and with the measured
      reactor feed composition they give every reaction rate.

Residuals are evaluated on a window of samples and reduced to a scalar per
window; the quasi-steady relations use the window mean, which is why the
window has to be long compared with the loop it closes over.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.tep import tep_constants as C                      # noqa: E402

DT_S = 180.0                     # sampling interval, seconds

# When true, the vessel balances revert to steady-state relations: their
# inventory and stored-energy accumulation terms are dropped.  Used by the
# model-mismatch experiment to measure what unmodelled dynamics cost.
NO_ACCUMULATION = False

# residual name -> structural equations consumed.  Fault signatures are
# derived from these by src/verifier/fsm.py using the structural model.
RESIDUAL_EQUATIONS = {
    "r_v1_Dfeed":   ["e_FTM1", "e_vpos1"],
    "r_v2_Efeed":   ["e_FTM2", "e_vpos2"],
    "r_v3_Afeed":   ["e_FTM3", "e_vpos3"],
    "r_v4_ACfeed":  ["e_FTM4", "e_vpos4"],
    "r_v5_recycle": ["e_FTM9", "e_vpos5", "e_XMWS9"],
    "r_v6_purge":   ["e_FTM10", "e_vpos6", "e_XMWS9"],
    "r_v7_sepflow": ["e_FTM11", "e_vpos7"],
    "r_v8_product": ["e_FTM13", "e_vpos8"],
    "r_steam":      ["e_UAC", "e_vpos9", "e_QUC"],
    "r_cpdh":       ["e_CPDH"],
    "r_f6_reacfeed": ["e_FTM6", "e_XMWS6"],
    "r_f8_reacout": ["e_FTM8", "e_XMWS8"],
    "r_cwR":        ["e_FWR", "e_vpos10", "e_UAR", "e_QUR", "e_TWR",
                     "e_TCWR", "e_AGSP"],
    "r_cwS":        ["e_FWS", "e_vpos11", "e_UAS", "e_QUS", "e_TWS",
                     "e_TCWS"],
    "r_balA":       ["e_x4A", "e_d_ac", "e_d_b"],
    "r_balB":       ["e_x4B", "e_d_b"],
    "r_balC":       ["e_x4C", "e_x4A", "e_x4B", "e_d_ac", "e_d_b"],
    "r_balD":       ["e_balR_D"],
    "r_dec2":       ["e_x4A", "e_x4B", "e_d_ac", "e_d_b"],
    "r_energy":     ["e_TST1", "e_HST1", "e_balV_E", "e_balR_En", "e_QUR"],
    "r_strip_En":   ["e_TST4", "e_HST4", "e_balC_E", "e_QUC"],
    "r_sep_En":     ["e_balS_E", "e_QUS"],
    "r_kin1":       ["e_rate1", "e_kdrift"],
    "r_kin2":       ["e_rate2", "e_kdrift"],
    "r_kin3":       ["e_rate3"],
    "r_kin4":       ["e_rate4"],
}
RESIDUAL_NAMES = list(RESIDUAL_EQUATIONS.keys())


def _pos(a):
    return np.maximum(a, 0.0)


def _slope(Y, dt, half=10):
    """Local least-squares slope of each column of Y, per unit time.

    A centred window of 2*half+1 samples is used, so the estimate is the
    accumulation rate at that sample rather than a difference across the
    whole record.
    """
    n = len(Y)
    t = (np.arange(-half, half + 1) * dt)
    denom = (t ** 2).sum()
    out = np.zeros_like(Y)
    pad = np.vstack([np.repeat(Y[:1], half, axis=0), Y,
                     np.repeat(Y[-1:], half, axis=0)])
    for i in range(n):
        w = pad[i:i + 2 * half + 1]
        out[i] = (t[:, None] * (w - w.mean(0))).sum(0) / denom
    return out


def derived(X):
    """Everything the residuals are built from, for one run of shape (T,53)."""
    d = {}
    x = np.asarray(X, dtype=float)
    d["FTM1"] = C.ftm1(x)
    d["FTM2"] = C.ftm2(x)
    d["FTM3"] = C.ftm3(x)
    d["FTM4"] = C.ftm4(x)
    d["FTM6"] = C.ftm6(x)
    d["FTM9"] = C.ftm9(x)
    d["FTM10"] = C.ftm10(x)
    d["PTR"], d["PTS"], d["PTV"] = C.ptr(x), C.pts(x), C.ptv(x)
    d["TCR"], d["TCS"], d["TCC"] = C.tcr(x), C.tcs(x), C.tcc(x)
    d["TWR"], d["TWS"] = C.twr(x), C.tws(x)
    d["VLR"] = C.vlr(x)
    d["QUC"], d["CPDH"] = C.quc(x), C.cpdh(x)

    d["XVV"] = C.xvv(x)
    d["XVS"] = C.xvs(x)
    d["XLC"] = C.xlc(x)
    d["XLS"] = C.xls_from_vle(x)
    d["DLS"] = C.liquid_density(d["XLS"], d["TCS"])
    d["DLC"] = C.liquid_density(d["XLC"], d["TCC"])
    d["FTM11"] = x[..., 13] * d["DLS"] * 35.3145
    d["FTM13"] = x[..., 16] * d["DLC"] * 35.3145

    # separator total and component balances give the reactor effluent
    d["FTM8"] = d["FTM9"] + d["FTM10"] + d["FTM11"]
    fcm8 = (d["XVS"] * (d["FTM9"] + d["FTM10"])[:, None]
            + d["XLS"] * d["FTM11"][:, None])
    d["FCM8"] = fcm8
    d["XST8"] = fcm8 / np.maximum(d["FTM8"], 1e-9)[:, None]
    d["XMWS8"] = d["XST8"] @ C.XMW
    d["XMWS9"] = d["XVS"] @ C.XMW
    d["XMWS6"] = d["XVV"] @ C.XMW

    # The reactor feed analyser covers A..F only, but the recycle and the
    # stripper overhead both carry G and H into the mixing zone.  Forcing
    # those to zero biases the mean molecular weight of the reactor feed,
    # which leaks into every relation that uses it.  The mixing-zone
    # component balance supplies them instead: what enters as G or H must
    # leave in the reactor feed.
    # (computed below, once the stripper overhead is known)

    # stripper balances give the overhead vapour that returns to the mixer
    x4 = np.zeros(8)
    x4[0], x4[1], x4[2] = C.X4A_NOM, C.X4B_NOM, C.X4C_NOM
    fcm4 = x4[None, :] * d["FTM4"][:, None]
    fcm11 = d["XLS"] * d["FTM11"][:, None]
    fcm13 = d["XLC"] * d["FTM13"][:, None]
    fcm5 = fcm4 + fcm11 - fcm13
    d["FTM5"] = fcm5.sum(-1)
    d["XST5"] = fcm5 / np.maximum(d["FTM5"], 1e-9)[:, None]

    gh = (d["XST5"][:, 6:] * d["FTM5"][:, None]
          + d["XVS"][:, 6:] * d["FTM9"][:, None]) / np.maximum(d["FTM6"], 1e-9)[:, None]
    xvv = d["XVV"].copy()
    xvv[:, 6:] = np.clip(gh, 0.0, 1.0)
    xvv[:, :6] *= np.clip(1.0 - xvv[:, 6:].sum(-1, keepdims=True), 0.0, 1.0)
    ssum2 = xvv.sum(-1, keepdims=True)
    d["XVV"] = xvv / np.where(ssum2 > 0, ssum2, 1.0)
    d["XMWS6"] = d["XVV"] @ C.XMW

    # measured reaction rates, from the reactor component balance
    fcm6 = d["XVV"] * d["FTM6"][:, None]
    crxr = fcm8 - fcm6
    d["CRXR"] = crxr
    d["RR1"] = crxr[:, 6]                       # G production
    d["RR2"] = crxr[:, 7]                       # H production
    d["RR3"] = -crxr[:, 4] - d["RR2"]           # from the E balance
    d["RR4"] = crxr[:, 5] - d["RR3"]            # from the F balance
    d["RH"] = d["RR1"] * C.HTR1 + d["RR2"] * C.HTR2

    # valve positions: the plant's own first-order actuator lag
    for j in range(1, 13):
        d["VPOS%d" % j] = C.valve_lag(x[..., 40 + j], C.VTAU_S[j], DT_S)
    d["AGSP"] = (d["VPOS12"] + 150.0) / 100.0
    d["FWR"] = d["VPOS10"] * C.VRNG[10] / 100.0
    d["FWS"] = d["VPOS11"] * C.VRNG[11] / 100.0

    # heat transfer
    lev = d["VLR"] / 7.8
    uarlev = np.clip(0.025 * lev - 0.25, 0.0, 1.0)
    uarlev = np.where(lev > 50.0, 1.0, np.where(lev < 10.0, 0.0, uarlev))
    d["UAR"] = uarlev * (-0.5 * d["AGSP"] ** 2 + 2.75 * d["AGSP"] - 2.5) * 855490e-6
    d["QUR"] = d["UAR"] * (d["TWR"] - d["TCR"])
    d["UAS"] = 0.404655 * (1.0 - 1.0 / (1.0 + (d["FTM8"] / 3528.73) ** 4))
    d["QUS"] = d["UAS"] * (d["TWS"] - d["TCR"])

    # ---- component inventories, so the balances can carry their
    # accumulation terms instead of assuming steady state.  Every holdup
    # below is computable from measurements: the reactor and separator
    # liquid compositions come from the VLE inversion, the vapour
    # inventories from the ideal-gas relation the source itself uses, and
    # the mixing-zone vapour inventory from its pressure at a fixed nominal
    # temperature (its variation is second order).
    xlr = np.zeros_like(d["XST8"])
    psr = C.psat(d["TCR"])
    ppr = d["XST8"] * d["PTR"][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xlr[:, 3:] = np.where(psr[:, 3:] > 1e-9, ppr[:, 3:] / psr[:, 3:], 0.0)
    ssum = xlr.sum(-1, keepdims=True)
    d["XLR"] = xlr / np.where(ssum > 0, ssum, 1.0)
    d["DLR"] = C.liquid_density(d["XLR"], d["TCR"])
    d["UTLR"] = d["VLR"] * d["DLR"]
    d["VVR"] = C.VTR - d["VLR"]
    d["UTVR"] = d["PTR"] * d["VVR"] / (C.RG * (d["TCR"] + 273.15))
    vls = x[..., 11] * 290.0 / 100.0 + 27.5
    d["UTLS"] = vls * d["DLS"]
    d["UTVS"] = d["PTS"] * (C.VTS - vls) / (C.RG * (d["TCS"] + 273.15))
    vlc = x[..., 14] * C.VTC / 100.0 + 78.25
    d["UTLC"] = vlc * d["DLC"]
    d["UTVV"] = d["PTV"] * C.VTV / (C.RG * (C.TCV_NOM + 273.15))
    d["N"] = (d["UTVR"][:, None] * d["XST8"] + d["UTLR"][:, None] * d["XLR"]
              + d["UTVS"][:, None] * d["XVS"] + d["UTLS"][:, None] * d["XLS"]
              + d["UTLC"][:, None] * d["XLC"] + d["UTVV"][:, None] * d["XVV"])

    # enthalpies
    d["HST1"] = C.enthalpy(C.XST1, C.TST1_NOM, 1) * np.ones_like(d["FTM1"])
    d["HST2"] = C.enthalpy(C.XST2, C.TST2_NOM, 1) * np.ones_like(d["FTM1"])
    d["HST3"] = C.enthalpy(C.XST3, C.TST3_NOM, 1) * np.ones_like(d["FTM1"])
    d["HST4"] = C.enthalpy(x4, C.TST4_NOM, 1) * np.ones_like(d["FTM1"])
    d["HST5"] = C.enthalpy(d["XST5"], d["TCC"], 1)
    d["HST8"] = C.enthalpy(d["XST8"], d["TCR"], 1)
    d["HST9b"] = C.enthalpy(d["XVS"], d["TCS"], 1)
    d["HST9"] = d["HST9b"] + d["CPDH"] / np.maximum(d["FTM9"], 1e-9)
    d["HST11"] = C.enthalpy(d["XLS"], d["TCS"], 0)
    d["HST13"] = C.enthalpy(d["XLC"], d["TCC"], 0)

    # stored energy of each vessel, so the energy balances can carry their
    # accumulation terms too
    # teprob.f stores only the liquid internal energy in ETR: ESR = ETR/UTLR
    # is inverted against the liquid enthalpy correlation (l.456, l.460).
    d["ETR"] = d["UTLR"] * C.enthalpy(d["XLR"], d["TCR"], 0)
    d["ETV"] = d["UTVV"] * C.enthalpy(d["XVV"], C.TCV_NOM, 2)
    d["ETC"] = d["UTLC"] * C.enthalpy(d["XLC"], d["TCC"], 0)
    d["ETS"] = (d["UTLS"] * C.enthalpy(d["XLS"], d["TCS"], 0)
                + d["UTVS"] * C.enthalpy(d["XVS"], d["TCS"], 2))
    return d


def residuals(X, with_scale=False):
    """Per-sample residuals for one run.

    Returns dict name -> (T,) array.  With with_scale=True also returns a
    dict of per-sample term magnitudes, one per residual: the size of the
    quantities the relation balances.  Dividing a residual by that turns it
    into a RELATIVE consistency error, which is what makes it comparable
    across operating points.

    This matters because the plant is under feedback control.  A disturbance
    anywhere makes every flow and duty in the plant larger and more
    restless, so an absolute residual grows even where the relation it
    encodes still holds exactly.  Left absolute, that lets a fault appear in
    relations the structure proves are decoupled from it -- including making
    two structurally indiscernible faults look different, which the theory
    forbids.  Relative errors do not have that failure mode.
    """
    d = derived(X)
    r = {}
    sc = {}
    r["r_v1_Dfeed"] = d["FTM1"] - d["VPOS1"] * C.VRNG[1] / 100.0
    r["r_v2_Efeed"] = d["FTM2"] - d["VPOS2"] * C.VRNG[2] / 100.0
    r["r_v3_Afeed"] = d["FTM3"] - d["VPOS3"] * C.VRNG[3] / 100.0
    r["r_v4_ACfeed"] = d["FTM4"] - d["VPOS4"] * C.VRNG[4] / 100.0
    r["r_v7_sepflow"] = d["FTM11"] - d["VPOS7"] * C.VRNG[7] / 100.0
    r["r_v8_product"] = d["FTM13"] - d["VPOS8"] * C.VRNG[8] / 100.0
    uac = d["VPOS9"] * C.VRNG[9] / 100.0
    r["r_steam"] = d["QUC"] - uac * (100.0 - d["TCC"])

    r["r_v6_purge"] = (d["FTM10"] - d["VPOS6"] * 0.151169
                       * np.sqrt(_pos(d["PTS"] - 760.0)) / d["XMWS9"])

    pr = np.clip(d["PTV"] / d["PTS"], 1.0, C.CPPRMX)
    flms = C.CPFLMX + C.FLCOEF * (1.0 - pr ** 3)
    r["r_v5_recycle"] = (d["FTM9"]
                         - (flms - d["VPOS5"] * 53.349
                            * np.sqrt(_pos(d["PTV"] - d["PTS"]))) / d["XMWS9"])
    r["r_cpdh"] = (d["CPDH"] - flms * (d["TCS"] + 273.15) * 1.8e-6 * 1.9872
                   * (d["PTV"] - d["PTS"]) / (d["XMWS9"] * d["PTS"]))
    r["r_f6_reacfeed"] = (d["FTM6"] - 1937.6
                          * np.sqrt(_pos(d["PTV"] - d["PTR"])) / d["XMWS6"])
    r["r_f8_reacout"] = (d["FTM8"] - 4574.21
                         * np.sqrt(_pos(d["PTR"] - d["PTS"])) / d["XMWS8"])

    # coolant loops, quasi-steady: the coolant hold-up time constant is about
    # two minutes, shorter than the three-minute sampling interval, so the
    # stored-energy term is negligible on a windowed average.
    r["r_cwR"] = (d["FWR"] * 500.53 * (C.TCWR_NOM - d["TWR"])
                  - d["QUR"] * 1e6 / 1.8)
    r["r_cwS"] = (d["FWS"] * 500.53 * (C.TCWS_NOM - d["TWS"])
                  - d["QUS"] * 1e6 / 1.8)

    # element balances.  A, B and C leave only in the purge; the stripper
    # sends all three overhead, so the product carries none of them.
    # accumulation rate of each component inventory, kmol per hour, from a
    # least-squares slope over the window; DT_S/3600 is the sample interval
    # in hours.
    dN = (np.zeros_like(d["N"]) if NO_ACCUMULATION
          else _slope(d["N"], DT_S / 3600.0))
    purge, prod = d["FTM10"], d["FTM13"]
    yv, xl = d["XVS"], d["XLC"]
    out = lambda i: prod * xl[:, i] + purge * yv[:, i]      # noqa: E731
    Gout, Hout, Fout, Eout, Dout = out(6), out(7), out(5), out(4), out(3)
    Ein = d["FTM2"] * 0.9999
    RR3 = Ein - Eout - Hout
    r["r_balB"] = ((d["FTM1"] + d["FTM3"]) * 1e-4 + d["FTM4"] * C.X4B_NOM
                   - purge * yv[:, 1] - dN[:, 1])
    r["r_balC"] = (d["FTM4"] * C.X4C_NOM - purge * yv[:, 2]
                   - (Gout + Hout) - dN[:, 2])
    r["r_balA"] = (d["FTM3"] * 0.9999 + d["FTM4"] * C.X4A_NOM
                   - purge * yv[:, 0] - (Gout + Ein - Eout) - dN[:, 0])
    r["r_balD"] = (d["FTM1"] * 0.9999 - Dout
                   - (Gout + 1.5 * (Fout - RR3)) - dN[:, 3])
    # Decoupling combination: annihilates the direction in which IDV(2)
    # moves the stream-4 composition (dxA = -2.43719e-3, dxB = +5.0e-3 per
    # unit fault, teprob.f l.407-409).  r_balA responds to dxA and r_balB to
    # dxB, both scaled by FTM4, so this combination is identically zero
    # under IDV(2) and non-zero under IDV(1) and IDV(8).
    r["r_dec2"] = 5.0e-3 * r["r_balA"] + 2.43719e-3 * r["r_balB"]

    # energy balances
    if NO_ACCUMULATION:
        dE_reac = np.zeros_like(d["ETR"])
        dE_strip = np.zeros_like(d["ETC"])
    else:
        dE_reac = _slope((d["ETR"] + d["ETV"])[:, None], DT_S / 3600.0)[:, 0]
        dE_strip = _slope(d["ETC"][:, None], DT_S / 3600.0)[:, 0]
    r["r_energy"] = (d["HST1"] * d["FTM1"] + d["HST2"] * d["FTM2"]
                     + d["HST3"] * d["FTM3"] + d["HST5"] * d["FTM5"]
                     + d["HST9"] * d["FTM9"] - d["HST8"] * d["FTM8"]
                     + d["RH"] + d["QUR"] - dE_reac)
    # Separator energy balance.  It contains the condenser duty QUS but not
    # the coolant flow or the coolant inlet temperature, which appear only in
    # the coolant balance.  That is what separates a drift in the condenser
    # heat transfer coefficient from a deviation on the coolant side, and it
    # is the relation the bank was missing: without it those three locations
    # share a signature and the bank falls one location short of the
    # structural bound.
    dE_sep = _slope(d["ETS"][:, None], DT_S / 3600.0)[:, 0]
    r["r_sep_En"] = (d["HST8"] * d["FTM8"] - d["HST9"] * d["FTM9"]
                     - d["HST9b"] * d["FTM10"] - d["HST11"] * d["FTM11"]
                     + d["QUS"] - dE_sep)

    r["r_strip_En"] = (d["HST4"] * d["FTM4"] + d["HST11"] * d["FTM11"]
                       - d["HST5"] * d["FTM5"] - d["HST13"] * d["FTM13"]
                       + d["QUC"] - dE_strip)

    # kinetics: rates measured from the component balances, against the
    # Arrhenius law with the source's own constants
    tkr = d["TCR"] + 273.15
    vvr = C.VTR - d["VLR"]
    ppr = d["XST8"] * d["PTR"][:, None]
    p1, p3, p4, p5 = (np.maximum(ppr[:, 0], 1e-9), np.maximum(ppr[:, 2], 1e-9),
                      ppr[:, 3], ppr[:, 4])
    k1 = np.exp(31.5859536 - 40000.0 / 1.987 / tkr)
    k2 = np.exp(3.00094014 - 20000.0 / 1.987 / tkr)
    k3 = np.exp(53.4060443 - 60000.0 / 1.987 / tkr)
    f13 = p1 ** 1.1544 * p3 ** 0.3735
    r["r_kin1"] = d["RR1"] - k1 * f13 * p4 * vvr
    r["r_kin2"] = d["RR2"] - k2 * f13 * p5 * vvr
    r["r_kin3"] = RR3 - k3 * p1 * p5 * vvr
    r["r_kin4"] = d["RR4"] - k3 * 0.767488334 * p1 * p4 * vvr

    # term magnitudes: the size of what each relation balances
    one = np.ones_like(d["FTM1"])
    sc["r_v1_Dfeed"] = np.abs(d["FTM1"])
    sc["r_v2_Efeed"] = np.abs(d["FTM2"])
    sc["r_v3_Afeed"] = np.abs(d["FTM3"])
    sc["r_v4_ACfeed"] = np.abs(d["FTM4"])
    sc["r_v7_sepflow"] = np.abs(d["FTM11"])
    sc["r_v8_product"] = np.abs(d["FTM13"])
    sc["r_steam"] = np.abs(d["QUC"])
    sc["r_v6_purge"] = np.abs(d["FTM10"])
    sc["r_v5_recycle"] = np.abs(d["FTM9"])
    sc["r_cpdh"] = np.abs(d["CPDH"])
    sc["r_f6_reacfeed"] = np.abs(d["FTM6"])
    sc["r_f8_reacout"] = np.abs(d["FTM8"])
    sc["r_cwR"] = np.abs(d["QUR"]) * 1e6 / 1.8
    sc["r_cwS"] = np.abs(d["QUS"]) * 1e6 / 1.8
    feed4 = np.abs(d["FTM4"])
    sc["r_balA"] = feed4 + np.abs(d["FTM3"]) + np.abs(Gout)
    sc["r_balB"] = feed4 * C.X4B_NOM + np.abs(purge * yv[:, 1])
    sc["r_balC"] = feed4 * C.X4C_NOM + np.abs(Gout + Hout)
    sc["r_balD"] = np.abs(d["FTM1"]) + np.abs(Dout) + np.abs(Gout)
    sc["r_dec2"] = 5.0e-3 * sc["r_balA"] + 2.43719e-3 * sc["r_balB"]
    sc["r_energy"] = (np.abs(d["HST8"] * d["FTM8"]) + np.abs(d["RH"])
                      + np.abs(d["QUR"]))
    sc["r_sep_En"] = (np.abs(d["HST8"] * d["FTM8"])
                      + np.abs(d["HST9"] * d["FTM9"]) + np.abs(d["QUS"]))
    sc["r_strip_En"] = (np.abs(d["HST13"] * d["FTM13"]) + np.abs(d["QUC"])
                        + np.abs(d["HST11"] * d["FTM11"]))
    sc["r_kin1"] = np.abs(d["RR1"]) + np.abs(k1 * f13 * p4 * vvr)
    sc["r_kin2"] = np.abs(d["RR2"]) + np.abs(k2 * f13 * p5 * vvr)
    sc["r_kin3"] = np.abs(RR3) + np.abs(k3 * p1 * p5 * vvr)
    sc["r_kin4"] = np.abs(d["RR4"]) + np.abs(k3 * 0.767488334 * p1 * p4 * vvr)
    for k in RESIDUAL_NAMES:
        sc[k] = np.maximum(np.abs(sc.get(k, one)), 1e-9)
    if with_scale:
        return r, sc
    return r


def window_features(X, lo, hi):
    """Reduce each residual over samples [lo, hi) to one scalar per residual.

    The reduction is the window mean, which is the right statistic for the
    quasi-steady relations and is what the conformal calibration is applied
    to.
    """
    r = residuals(X)
    return np.array([np.nanmean(r[k][lo:hi]) for k in RESIDUAL_NAMES])
