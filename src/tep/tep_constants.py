"""
Plant constants and measurement conversions, transcribed from teprob.f.

These make the analytical redundancy relations exact rather than fitted:
every coefficient below appears verbatim in the reference source, with the
line number given in the comment.
"""
from __future__ import annotations

import numpy as np

# component order A B C D E F G H  (teprob.f indices 1..8)
XMW = np.array([2.0, 25.4, 28.0, 32.0, 46.0, 48.0, 62.0, 76.0])      # l.941
AVP = np.array([0.0, 0.0, 0.0, 15.92, 16.35, 16.35, 16.43, 17.21])   # l.949
BVP = np.array([0.0, 0.0, 0.0, -1444.0, -2114.0, -2114.0, -2748.0, -3318.0])
CVP = np.array([0.0, 0.0, 0.0, 259.0, 265.5, 265.5, 232.9, 249.6])
AD = np.array([1.0, 1.0, 1.0, 23.3, 33.9, 32.8, 49.9, 50.5])         # l.973
BD = np.array([0.0, 0.0, 0.0, -0.0700, -0.0957, -0.0995, -0.0191, -0.0541])
CD = np.array([0.0, 0.0, 0.0, -0.0002, -0.000152, -0.000233,
               -0.000425, -0.000150])

VRNG = {1: 400.0, 2: 400.0, 3: 100.0, 4: 1500.0, 7: 1500.0, 8: 1000.0,
        9: 0.03, 10: 1000.0, 11: 1200.0}                             # l.1109
VTAU_S = {1: 8.0, 2: 8.0, 3: 6.0, 4: 9.0, 5: 7.0, 6: 5.0, 7: 5.0,
          8: 5.0, 9: 120.0, 10: 5.0, 11: 5.0, 12: 5.0}               # l.1172
HWR, HWS = 7060.0, 11138.0                                           # l.1124
VTR, VTS, VTC, VTV = 1300.0, 3500.0, 156.5, 5000.0                   # l.1118
CPFLMX, CPPRMX = 280275.0, 1.3                                       # l.1170
FLCOEF = CPFLMX / 1.197
RG = 998.9

# nominal levels of the disturbance generator (SZERO, l.1300+)
X4A_NOM = 0.485        # stream 4 mole fraction of A
X4B_NOM = 0.005        # stream 4 mole fraction of B
X4C_NOM = 1.0 - X4A_NOM - X4B_NOM
TST1_NOM = 45.0        # D feed temperature, degC
TST4_NOM = 45.0        # A/C feed temperature, degC
TCWR_NOM = 35.0        # reactor coolant inlet temperature, degC
TCWS_NOM = 40.0        # condenser coolant inlet temperature, degC
TCV_NOM = 86.0         # mixing-zone vapour temperature at the nominal
                       # operating point; it is not measured and its
                       # variation only enters the inventory correction

# feed compositions of the pure streams (l.1134-1159)
XST1 = np.array([0.0, 1e-4, 0.0, 0.9999, 0.0, 0.0, 0.0, 0.0])    # D feed
XST2 = np.array([0.0, 0.0, 0.0, 0.0, 0.9999, 1e-4, 0.0, 0.0])    # E feed
XST3 = np.array([0.9999, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # A feed
XMWS1 = float(XST1 @ XMW)
XMWS2 = float(XST2 @ XMW)
XMWS3 = float(XST3 @ XMW)

# Random-walk amplitudes (SSPAN).  A trap worth documenting, because the
# table below looks as if it disables three disturbances and does not.
#
# SSPAN(10) = SSPAN(11) = SSPAN(12) = 0, which would zero walks 10 to 12 IF
# they were driven the way walks 1 to 9 are.  They are not.  Walks 1 to 9 get
# their amplitude from TESUB5 as SSPAN * U(-1,1) * IDVFLAG + SZERO.  Walks 10
# to 12 never call TESUB5: they are updated in the DO 910 block of TEFUNC
# (teprob.f l.372-396), where the amplitude comes straight from the fault
# flag as CDIST(I) = IDVWLK(I) / HWLK**2, ramping quadratically to about 1
# over HWLK = HSPAN(I)*U(-1,1) + HZERO(I) hours and then decaying.  SSPAN is
# simply unused for those three.
#
# So IDV(17), (18) and (20) are active and strong.  Measured against nominal
# on the generated corpus: IDV(17) shifts the reactor coolant outlet
# temperature by 29 standard deviations, IDV(18) shifts the E feed valve by
# 59 and trips the plant in all eight runs, IDV(20) shifts the compressor
# recycle valve by 4.4.
#
# An earlier version of this file asserted the opposite, reading SSPAN = 0 as
# disabling them.  The error surfaced when the shutdown accounting showed
# eight of eight IDV(18) runs tripping the plant, which a zero-amplitude
# disturbance cannot do.
SSPAN = {1: 0.03, 2: 0.003, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0,
         7: 0.25, 8: 0.25, 9: 0.25, 10: 0.0, 11: 0.0, 12: 0.0}
RAMP_WALKS = {10: 17, 11: 18, 12: 20}    # walk index -> IDV it drives
INACTIVE_IDV = []
ACTIVE_IDV = [k for k in range(1, 21) if k not in INACTIVE_IDV]


# enthalpy correlations (TESUB1, l.1375)
AH = np.array([1.0e-6, 1.0e-6, 1.0e-6, 0.960e-6, 0.573e-6, 0.652e-6,
               0.515e-6, 0.471e-6])
BH = np.array([0.0, 0.0, 0.0, 8.70e-9, 2.41e-9, 2.18e-9, 5.65e-10, 8.70e-10])
CH = np.array([0.0, 0.0, 0.0, 4.81e-11, 1.82e-11, 1.94e-11, 3.82e-12,
               2.62e-12])
AG = np.array([3.411e-6, 0.3799e-6, 0.2491e-6, 0.3567e-6, 0.3463e-6,
               0.3930e-6, 0.170e-6, 0.150e-6])
BG = np.array([7.18e-10, 1.08e-9, 1.36e-11, 8.51e-10, 8.96e-10, 1.02e-9,
               0.0, 0.0])
CG = np.array([6.0e-13, -3.98e-13, -3.93e-14, -3.12e-13, -3.27e-13,
               -3.12e-13, 0.0, 0.0])
AV = np.array([1.0e-6, 1.0e-6, 1.0e-6, 86.7e-6, 160.0e-6, 160.0e-6,
               225.0e-6, 209.0e-6])
HTR1, HTR2 = 0.06899381054, 0.05        # heats of reaction, l.1122
TST2_NOM = 45.0                          # E feed temperature, l.1151
TST3_NOM = 45.0                          # A feed temperature, l.1160


def enthalpy(Z, T, phase):
    """TESUB1.  phase 0 = liquid, 1 = vapour, 2 = vapour minus RT."""
    T = np.asarray(T, dtype=float)[..., None]
    if phase == 0:
        hi = 1.8 * T * (AH + BH * T / 2.0 + CH * T ** 2 / 3.0)
    else:
        hi = 1.8 * T * (AG + BG * T / 2.0 + CG * T ** 2 / 3.0) + AV
    h = (Z * XMW * hi).sum(-1)
    if phase == 2:
        h = h - (3.57696e-6) * (T[..., 0] + 273.15)
    return h


# ------------------------------------------------ measurement conversions
def ftm3(x):    return x[..., 0] * 35.3145 / 0.359          # A feed, stream 3
def ftm1(x):    return x[..., 1] / (XMWS1 * 0.454)          # D feed, stream 1
def ftm2(x):    return x[..., 2] / (XMWS2 * 0.454)          # E feed, stream 2
def ftm4(x):    return x[..., 3] * 35.3145 / 0.359          # A and C feed
def ftm9(x):    return x[..., 4] * 35.3145 / 0.359          # recycle
def ftm6(x):    return x[..., 5] * 35.3145 / 0.359          # reactor feed
def ptr(x):     return x[..., 6] * 760.0 / 101.325 + 760.0  # reactor pressure
def vlr(x):     return x[..., 7] * 666.7 / 100.0 + 84.6     # reactor volume
def tcr(x):     return x[..., 8]
def ftm10(x):   return x[..., 9] * 35.3145 / 0.359          # purge
def tcs(x):     return x[..., 10]
def vls(x):     return x[..., 11] * 290.0 / 100.0 + 27.5
def pts(x):     return x[..., 12] * 760.0 / 101.325 + 760.0
def vlc(x):     return x[..., 14] * VTC / 100.0 + 78.25
def ptv(x):     return x[..., 15] * 760.0 / 101.325 + 760.0
def tcc(x):     return x[..., 17]
def quc(x):     return x[..., 18] / (1.04e3 * 0.454)
def cpdh(x):    return x[..., 19] / 0.29307e3
def twr(x):     return x[..., 20]
def tws(x):     return x[..., 21]


def xvv(x):
    """Reactor feed composition A..F (XMEAS 23..28), renormalised.

    G and H are not analysed on this stream; they are negligible there, so
    the six measured fractions are renormalised to sum to one.
    """
    v = x[..., 22:28] / 100.0
    s = v.sum(-1, keepdims=True)
    out = np.zeros(x.shape[:-1] + (8,))
    out[..., :6] = v / np.where(s > 0, s, 1.0)
    return out


def xvs(x):
    """Purge / separator overhead composition A..H (XMEAS 29..36)."""
    v = x[..., 28:36] / 100.0
    s = v.sum(-1, keepdims=True)
    return v / np.where(s > 0, s, 1.0)


def xlc(x):
    """Stripper product composition (XMEAS 37..41 are D..H; A..C are zero).

    The stripper sends all of A, B and C overhead (SFR(1..3) = 1 in the
    source), so the product carries only D..H.
    """
    out = np.zeros(x.shape[:-1] + (8,))
    v = x[..., 36:41] / 100.0
    s = v.sum(-1, keepdims=True)
    out[..., 3:] = v / np.where(s > 0, s, 1.0)
    return out


def psat(T):
    """Antoine vapour pressure, mmHg, for the condensable components."""
    T = np.asarray(T)[..., None]
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(AVP + BVP / (T + CVP))
    p[..., :3] = 0.0
    return p


def liquid_density(X, T):
    """TESUB4: liquid density from composition and temperature."""
    T = np.asarray(T)[..., None]
    v = (X * XMW / (AD + (BD + CD * T) * T)).sum(-1)
    return 1.0 / v


def xls_from_vle(x):
    """Separator liquid composition, recovered from measurements.

    The separator holds vapour and liquid in equilibrium, so for the
    condensable components  PPS_i = Psat_i(TCS) * XLS_i  and also
    PPS_i = XVS_i * PTS.  Both sides are measured, so XLS follows.  A, B
    and C are non-condensable and do not appear in the liquid.
    """
    yv = xvs(x)
    P = pts(x)[..., None]
    ps = psat(tcs(x))
    out = np.zeros_like(yv)
    ok = ps > 1e-9
    out[..., 3:] = np.where(ok[..., 3:], yv[..., 3:] * P / np.where(ok, ps, 1.0)[..., 3:], 0.0)
    s = out.sum(-1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def valve_lag(u, tau_s, dt_s):
    """First-order valve position, the plant's VPOS dynamics.

    teprob.f integrates dVPOS/dt = (XMV - VPOS)/VTAU with a 1 s Euler step.
    Only the sampled command is available here, so the same lag is applied
    at the sampling rate.
    """
    a = np.exp(-dt_s / tau_s)
    out = np.empty_like(u)
    acc = u[0]
    for i in range(len(u)):
        acc = a * acc + (1.0 - a) * u[i]
        out[i] = acc
    return out
