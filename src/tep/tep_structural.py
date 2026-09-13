"""
Structural model of the Tennessee Eastman Process, transcribed directly
from the reference implementation `teprob.f` (Downs & Vogel 1993; Fortran
source as distributed by N. L. Ricker / Braatz group).

Every equation below corresponds to a numbered line range in teprob.f, and
every fault symbol is placed on the equation the Fortran actually perturbs.
The file data/raw/tep_src/teprob.f is the authority, and the line-by-line
correspondence is the inline citation on each equation below -- there is no
separate map file.

Model version 2.  Version 1 (src/tep/tep_structural_v1.py) used textbook
heat-exchanger relations with a constant UA and a log-mean coolant
temperature.  That introduced redundancy on the cooling loops that the
benchmark does not have, and it failed Gate B.  The reference source shows
that (i) the coolant outlet temperature is a state driven by the coolant
energy balance, (ii) UAR depends on reactor level and agitator speed only,
and (iii) UAS depends on the *process* flow FTM(8), never on the coolant
flow.  Version 2 follows the source.

Internal stream numbering (teprob.f convention, NOT the paper's):
    1  D feed            2  E feed           3  A feed
    4  A and C feed      5  stripper overhead vapour to mixing zone
    6  mixing zone outlet = reactor feed     7  copy of 6
    8  reactor effluent  9  separator overhead (compressed recycle)
   10  purge            11  separator underflow
   12  stripper internal vapour             13  stripper product

Vessels: R reactor, S separator, C stripper (liquid), V mixing zone /
vapour header.  States: 8 component holdups + 1 energy per vessel, the two
coolant outlet temperatures TWR / TWS, and 12 valve positions -> 50 states,
matching NN = 50 in the source.
"""
from __future__ import annotations

import faultdiagnosistoolbox as fdt

NC = 8                                   # components A..H -> indices 1..8
CIDX = list(range(1, NC + 1))
CONDENSABLE = [4, 5, 6, 7, 8]            # D..H obey Antoine/Raoult
NONCOND = [1, 2, 3]                      # A, B, C are non-condensable

# --------------------------------------------------------------- faults
# IDV -> (description, teprob.f line, difficulty label from the literature)
# Difficulty labels were fixed BEFORE any structural result was computed
# (see results/gates/prereg_labels.json).
IDV_INFO = {
    "IDV1":  ("A/C feed ratio, B composition constant (stream 4), step", "easy"),
    "IDV2":  ("B composition, A/C ratio constant (stream 4), step", "easy"),
    "IDV3":  ("D feed temperature (stream 2), step", "hard"),
    "IDV4":  ("Reactor cooling water inlet temperature, step", "easy"),
    "IDV5":  ("Condenser cooling water inlet temperature, step", "medium"),
    "IDV6":  ("A feed loss (stream 1), step", "easy"),
    "IDV7":  ("C header pressure loss, reduced availability (stream 4)", "easy"),
    "IDV8":  ("A, B, C feed composition (stream 4), random variation", "medium"),
    "IDV9":  ("D feed temperature (stream 2), random variation", "hard"),
    "IDV10": ("C feed temperature (stream 4), random variation", "medium"),
    "IDV11": ("Reactor cooling water inlet temperature, random variation", "medium"),
    "IDV12": ("Condenser cooling water inlet temperature, random variation", "medium"),
    "IDV13": ("Reaction kinetics, slow drift", "medium"),
    "IDV14": ("Reactor cooling water valve, sticking", "easy"),
    "IDV15": ("Condenser cooling water valve, sticking", "hard"),
    "IDV16": ("Stripper steam valve / reboiler UA, random variation", "hard"),
    "IDV17": ("Reactor heat transfer coefficient, random variation", "medium"),
    "IDV18": ("Condenser heat transfer coefficient, random variation", "medium"),
    "IDV19": ("Valve sticking, valves 5/7/8/9", "hard"),
    "IDV20": ("Reactor-to-separator flow coefficient, random variation", "hard"),
}
IDVS = list(IDV_INFO.keys())
FAULTS = ["f" + k for k in IDVS]

# --------------------------------------------------------------- sensors
# XMEAS name -> list of unknowns appearing in its measurement equation.
# Most are a single variable; XMEAS(14) and XMEAS(17) are volumetric flows
# and therefore also carry the corresponding liquid density.
SENSORS = {
    "XMEAS1": ["FTM3"],            # A feed
    "XMEAS2": ["FTM1"],            # D feed
    "XMEAS3": ["FTM2"],            # E feed
    "XMEAS4": ["FTM4"],            # A and C feed
    "XMEAS5": ["FTM9"],            # recycle flow
    "XMEAS6": ["FTM6"],            # reactor feed rate
    "XMEAS7": ["PTR"],             # reactor pressure
    "XMEAS8": ["VLR"],             # reactor level
    "XMEAS9": ["TCR"],             # reactor temperature
    "XMEAS10": ["FTM10"],          # purge rate
    "XMEAS11": ["TCS"],            # separator temperature
    "XMEAS12": ["VLS"],            # separator level
    "XMEAS13": ["PTS"],            # separator pressure
    "XMEAS14": ["FTM11", "DLS"],   # separator underflow (volumetric)
    "XMEAS15": ["VLC"],            # stripper level
    "XMEAS16": ["PTV"],            # stripper / header pressure
    "XMEAS17": ["FTM13", "DLC"],   # stripper underflow (volumetric)
    "XMEAS18": ["TCC"],            # stripper temperature
    "XMEAS19": ["QUC"],            # stripper steam flow (reboiler duty)
    "XMEAS20": ["CPDH"],           # compressor work
    "XMEAS21": ["TWR"],            # reactor cooling water outlet temp
    "XMEAS22": ["TWS"],            # condenser cooling water outlet temp
}
for _k in range(6):                                     # XMEAS 23..28
    SENSORS["XMEAS%d" % (23 + _k)] = ["XVV%d" % (_k + 1)]
for _k in range(8):                                     # XMEAS 29..36
    SENSORS["XMEAS%d" % (29 + _k)] = ["XVS%d" % (_k + 1)]
for _k in range(5):                                     # XMEAS 37..41
    SENSORS["XMEAS%d" % (37 + _k)] = ["XLC%d" % (_k + 4)]

XMV = ["XMV%d" % j for j in range(1, 13)]

# Sensor groups used by the E6 ablation ladder.
SENSOR_GROUPS = {
    "feed_flows": ["XMEAS1", "XMEAS2", "XMEAS3", "XMEAS4"],
    "internal_flows": ["XMEAS5", "XMEAS6", "XMEAS10", "XMEAS14", "XMEAS17"],
    "pressures": ["XMEAS7", "XMEAS13", "XMEAS16"],
    "levels": ["XMEAS8", "XMEAS12", "XMEAS15"],
    "temperatures": ["XMEAS9", "XMEAS11", "XMEAS18"],
    "coolant_temps": ["XMEAS21", "XMEAS22"],
    "duties": ["XMEAS19", "XMEAS20"],
    "analyser_feed": ["XMEAS%d" % k for k in range(23, 29)],
    "analyser_purge": ["XMEAS%d" % k for k in range(29, 37)],
    "analyser_product": ["XMEAS%d" % k for k in range(37, 42)],
}

# ---------------------------------------------------- candidate new sensors
# Instruments the plant does NOT have, used to ask which indiscernibility
# classes an instrumentation upgrade could split and which are irreducible.
OPTIONAL_SENSORS = {
    "NEW_cw_flow_cond": ["FWS"],      # condenser coolant flow meter
    "NEW_cw_flow_reac": ["FWR"],      # reactor coolant flow meter
    "NEW_cw_Tin_cond": ["TCWS"],      # condenser coolant inlet thermometer
    "NEW_cw_Tin_reac": ["TCWR"],      # reactor coolant inlet thermometer
    "NEW_Dfeed_temp": ["TST1"],       # D feed thermometer
    "NEW_Cfeed_temp": ["TST4"],       # A/C feed thermometer
    "NEW_reactor_out_flow": ["FTM8"], # reactor effluent flow meter
    "NEW_feed4_analyser": ["XST1_4", "XST2_4"],  # stream 4 composition
}


def _v(prefix, i):
    return "%s%d" % (prefix, i)


# stream -> components actually present in the pure feed streams
# (teprob.f l.1134-1159: D feed carries D and trace B, E feed carries E and
#  trace F, A feed carries A and trace B)
FEED_COMP = {1: [2, 4], 2: [5, 6], 3: [1, 2]}


def tep_equations():
    """Return (equations, diff_pairs).

    equations: list of (name, [variables]);  diff_pairs: (dvar, var).
    """
    E, diff = [], []

    def eq(name, *groups):
        vs = []
        for g in groups:
            if isinstance(g, str):
                vs.append(g)
            else:
                vs.extend(g)
        E.append((name, vs))

    # =============================================== valve dynamics (l.798)
    # YP(I+38) = (VCV(I) - VPOS(I))/VTAU(I); VCV(I) = XMV(I) unless stuck.
    stick = {5: "dev19", 7: "dev19", 8: "dev19", 9: "dev19",
             10: "fIDV14", 11: "fIDV15"}
    for j in range(1, 13):
        v = [_v("dVPOS", j), _v("VPOS", j), _v("XMV", j)]
        if j in stick:
            v.append(stick[j])
        eq(_v("e_vpos", j), v)
        diff.append((_v("dVPOS", j), _v("VPOS", j)))
    eq("e_dev19", ["dev19", "fIDV19"])

    # ======================================== feed specification (l.407-416)
    eq("e_XST1_4", ["XST1_4", "fIDV1", "dev2", "dev8"])
    eq("e_XST2_4", ["XST2_4", "dev2", "dev8"])
    eq("e_XST3_4", ["XST3_4", "XST1_4", "XST2_4"])
    eq("e_dev2", ["dev2", "fIDV2"])
    eq("e_dev8", ["dev8", "fIDV8"])
    eq("e_TST1", ["TST1", "fIDV3", "fIDV9"])
    eq("e_TST2", ["TST2"])
    eq("e_TST3", ["TST3"])
    eq("e_TST4", ["TST4", "fIDV10"])
    eq("e_TCWR", ["TCWR", "fIDV4", "fIDV11"])
    eq("e_TCWS", ["TCWS", "fIDV5", "fIDV12"])
    eq("e_dev13", ["dev13", "fIDV13"])

    # ================================== valve-driven flows (l.565-577)
    eq("e_FTM1", ["FTM1", "VPOS1"])
    eq("e_FTM2", ["FTM2", "VPOS2"])
    eq("e_FTM3", ["FTM3", "VPOS3", "fIDV6"])
    eq("e_FTM4", ["FTM4", "VPOS4", "fIDV7"])
    eq("e_FTM11", ["FTM11", "VPOS7"])
    eq("e_FTM13", ["FTM13", "VPOS8"])
    eq("e_UAC", ["UAC", "VPOS9", "fIDV16"])
    eq("e_FWR", ["FWR", "VPOS10"])
    eq("e_FWS", ["FWS", "VPOS11"])
    eq("e_AGSP", ["AGSP", "VPOS12"])

    # ================================ pressure-driven flows (l.578-604)
    eq("e_FTM6", ["FTM6", "PTV", "PTR", "XMWS6"])
    eq("e_FTM8", ["FTM8", "PTR", "PTS", "XMWS8", "fIDV20"])
    eq("e_FTM10", ["FTM10", "VPOS6", "PTS", "XMWS9"])
    eq("e_FTM9", ["FTM9", "PTV", "PTS", "VPOS5", "XMWS9"])
    eq("e_CPDH", ["CPDH", "PTV", "PTS", "TCS", "XMWS9"])
    eq("e_XMWS6", ["XMWS6"] + [_v("XVV", i) for i in CIDX])
    eq("e_XMWS8", ["XMWS8"] + [_v("XVR", i) for i in CIDX])
    eq("e_XMWS9", ["XMWS9"] + [_v("XVS", i) for i in CIDX])

    # ============================== reactor thermodynamics (l.417-500)
    # YY(4..8) IS the reactor liquid holdup, so UCLR == NR (folded).
    eq("e_UTLR", ["UTLR"] + [_v("NR", i) for i in CONDENSABLE])
    for i in CONDENSABLE:
        eq(_v("e_XLR", i), [_v("XLR", i), _v("NR", i), "UTLR"])
    eq("e_ESR", ["ESR", "ETR", "UTLR"])
    eq("e_TCR", ["TCR", "ESR"] + [_v("XLR", i) for i in CONDENSABLE])
    eq("e_DLR", ["DLR", "TCR"] + [_v("XLR", i) for i in CONDENSABLE])
    eq("e_VLR", ["VLR", "UTLR", "DLR"])
    eq("e_VVR", ["VVR", "VLR"])
    for i in NONCOND:
        eq(_v("e_PPR", i), [_v("PPR", i), _v("NR", i), "TCR", "VVR"])
    for i in CONDENSABLE:
        eq(_v("e_PPR", i), [_v("PPR", i), "TCR", _v("XLR", i)])
    eq("e_PTR", ["PTR"] + [_v("PPR", i) for i in CIDX])
    for i in CIDX:
        eq(_v("e_XVR", i), [_v("XVR", i), _v("PPR", i), "PTR"])

    # ============================ separator thermodynamics (l.417-501)
    eq("e_UTLS", ["UTLS"] + [_v("NS", i) for i in CONDENSABLE])
    for i in CONDENSABLE:
        eq(_v("e_XLS", i), [_v("XLS", i), _v("NS", i), "UTLS"])
    eq("e_ESS", ["ESS", "ETS", "UTLS"])
    eq("e_TCS", ["TCS", "ESS"] + [_v("XLS", i) for i in CONDENSABLE])
    eq("e_DLS", ["DLS", "TCS"] + [_v("XLS", i) for i in CONDENSABLE])
    eq("e_VLS", ["VLS", "UTLS", "DLS"])
    eq("e_VVS", ["VVS", "VLS"])
    for i in NONCOND:
        eq(_v("e_PPS", i), [_v("PPS", i), _v("NS", i), "TCS", "VVS"])
    for i in CONDENSABLE:
        eq(_v("e_PPS", i), [_v("PPS", i), "TCS", _v("XLS", i)])
    eq("e_PTS", ["PTS"] + [_v("PPS", i) for i in CIDX])
    for i in CIDX:
        eq(_v("e_XVS", i), [_v("XVS", i), _v("PPS", i), "PTS"])

    # ================================== stripper (liquid only, l.453-472)
    eq("e_UTLC", ["UTLC"] + [_v("NC", i) for i in CIDX])
    for i in CIDX:
        eq(_v("e_XLC", i), [_v("XLC", i), _v("NC", i), "UTLC"])
    eq("e_ESC", ["ESC", "ETC", "UTLC"])
    eq("e_TCC", ["TCC", "ESC"] + [_v("XLC", i) for i in CIDX])
    eq("e_DLC", ["DLC", "TCC"] + [_v("XLC", i) for i in CIDX])
    eq("e_VLC", ["VLC", "UTLC", "DLC"])

    # ============================ mixing zone / vapour header (l.429-492)
    eq("e_UTVV", ["UTVV"] + [_v("NV", i) for i in CIDX])
    for i in CIDX:
        eq(_v("e_XVV", i), [_v("XVV", i), _v("NV", i), "UTVV"])
    eq("e_ESV", ["ESV", "ETV", "UTVV"])
    eq("e_TCV", ["TCV", "ESV"] + [_v("XVV", i) for i in CIDX])
    eq("e_PTV", ["PTV", "UTVV", "TCV"])

    # =========================================== reaction (l.503-528)
    eq("e_RR1", ["RR1", "TCR", "PPR1", "PPR3", "PPR4", "VVR", "dev13"])
    eq("e_RR2", ["RR2", "TCR", "PPR1", "PPR3", "PPR5", "VVR", "dev13"])
    eq("e_RR3", ["RR3", "TCR", "PPR1", "PPR5", "VVR"])
    eq("e_RR4", ["RR4", "TCR", "PPR1", "PPR4", "VVR"])
    crxr = {1: ["RR1", "RR2", "RR3"], 3: ["RR1", "RR2"],
            4: ["RR1", "RR4"], 5: ["RR2", "RR3"],
            6: ["RR3", "RR4"], 7: ["RR1"], 8: ["RR2"]}
    for i, rr in crxr.items():
        eq(_v("e_CRXR", i), [_v("CRXR", i)] + rr)
    eq("e_RH", ["RH", "RR1", "RR2"])

    # ============================================== heat (l.663-680, 789)
    eq("e_UAR", ["UAR", "VLR", "AGSP"])
    eq("e_QUR", ["QUR", "UAR", "TWR", "TCR", "fIDV17"])
    eq("e_UAS", ["UAS", "FTM8"])
    eq("e_QUS", ["QUS", "UAS", "TWS", "TCR", "fIDV18"])   # TST(8) == TCR
    eq("e_QUC", ["QUC", "UAC", "TCC"])
    eq("e_TWR", ["dTWR", "FWR", "TCWR", "TWR", "QUR"])
    eq("e_TWS", ["dTWS", "FWS", "TCWS", "TWS", "QUS"])
    diff.append(("dTWR", "TWR"))
    diff.append(("dTWS", "TWS"))

    # ================================= stripper vapour split (l.614-651)
    eq("e_TMPFAC", ["TMPFAC", "TCC"])
    eq("e_VOVRL", ["VOVRL", "FTM4", "FTM11", "TMPFAC"])
    for i in CONDENSABLE:
        eq(_v("e_SFR", i), [_v("SFR", i), "VOVRL"])
    for i in CIDX:
        fin = [_v("FIN", i), _v("FCM%d_" % i, 11)]
        if i <= 3:                      # only A, B, C are present in stream 4
            fin.append(_v("FCM%d_" % i, 4))
        eq(_v("e_FIN", i), fin)
        v = [_v("FCM%d_" % i, 5), _v("FIN", i)]
        if i in CONDENSABLE:
            v.append(_v("SFR", i))
        eq("e_FCM%d_5" % i, v)
        eq("e_FCM%d_12" % i, [_v("FCM%d_" % i, 12), _v("FIN", i),
                              _v("FCM%d_" % i, 5)])
    eq("e_FTM5", ["FTM5"] + [_v("FCM%d_" % i, 5) for i in CIDX])
    for i in CIDX:
        eq("e_XST%d_5" % i, [_v("XST%d_" % i, 5), _v("FCM%d_" % i, 5), "FTM5"])

    # ==================================== component molar flows (l.604-613)
    # streams 1,2,3 are pure feeds: FCM(i,k) is proportional to FTM(k)
    for i in CIDX:
        if i <= 3:          # stream 4 carries only A, B, C
            eq("e_FCM%d_4" % i,
               [_v("FCM%d_" % i, 4), "FTM4", _v("XST%d_" % i, 4)])
        eq("e_FCM%d_6" % i, [_v("FCM%d_" % i, 6), _v("XVV", i), "FTM6"])
        eq("e_FCM%d_8" % i, [_v("FCM%d_" % i, 8), _v("XVR", i), "FTM8"])
        eq("e_FCM%d_9" % i, [_v("FCM%d_" % i, 9), _v("XVS", i), "FTM9"])
        eq("e_FCM%d_10" % i, [_v("FCM%d_" % i, 10), _v("XVS", i), "FTM10"])
        eq("e_FCM%d_11" % i, [_v("FCM%d_" % i, 11), _v("XLS", i), "FTM11"])
        eq("e_FCM%d_13" % i, [_v("FCM%d_" % i, 13), _v("XLC", i), "FTM13"])

    # ============================================= enthalpies (l.550-565)
    eq("e_HST1", ["HST1", "TST1"])
    eq("e_HST2", ["HST2", "TST2"])
    eq("e_HST3", ["HST3", "TST3"])
    eq("e_HST4", ["HST4", "TST4", "XST1_4", "XST2_4", "XST3_4"])
    eq("e_HST5", ["HST5", "TCC"] + [_v("XST%d_" % i, 5) for i in CIDX])
    eq("e_HST6", ["HST6", "TCV"] + [_v("XVV", i) for i in CIDX])
    eq("e_HST8", ["HST8", "TCR"] + [_v("XVR", i) for i in CIDX])
    eq("e_HST9b", ["HST9b", "TCS"] + [_v("XVS", i) for i in CIDX])
    eq("e_HST9", ["HST9", "HST9b", "CPDH", "FTM9"])
    eq("e_HST11", ["HST11", "TCS"] + [_v("XLS", i) for i in CIDX])
    eq("e_HST13", ["HST13", "TCC"] + [_v("XLC", i) for i in CIDX])

    # ============================================== balances (l.762-792)
    for i in CIDX:
        eq(_v("e_balR", i), [_v("dNR", i), _v("FCM%d_" % i, 6),
                             _v("FCM%d_" % i, 8)] +
           ([_v("CRXR", i)] if i != 2 else []))
        diff.append((_v("dNR", i), _v("NR", i)))
        eq(_v("e_balS", i), [_v("dNS", i), _v("FCM%d_" % i, 8),
                             _v("FCM%d_" % i, 9), _v("FCM%d_" % i, 10),
                             _v("FCM%d_" % i, 11)])
        diff.append((_v("dNS", i), _v("NS", i)))
        eq(_v("e_balC", i), [_v("dNC", i), _v("FCM%d_" % i, 12),
                             _v("FCM%d_" % i, 13)])
        diff.append((_v("dNC", i), _v("NC", i)))
        balv = [_v("dNV", i), _v("FCM%d_" % i, 5), _v("FCM%d_" % i, 9),
                _v("FCM%d_" % i, 6)]
        for k, comps in FEED_COMP.items():
            if i in comps:
                balv.append(_v("FTM", k))
        eq(_v("e_balV", i), balv)
        diff.append((_v("dNV", i), _v("NV", i)))
    eq("e_balR_E", ["dETR", "HST6", "FTM6", "HST8", "FTM8", "RH", "QUR"])
    diff.append(("dETR", "ETR"))
    eq("e_balS_E", ["dETS", "HST8", "FTM8", "HST9", "FTM9",
                    "HST9b", "FTM10", "HST11", "FTM11", "QUS"])
    diff.append(("dETS", "ETS"))
    eq("e_balC_E", ["dETC", "HST4", "FTM4", "HST11", "FTM11",
                    "HST5", "FTM5", "HST13", "FTM13", "QUC"])
    diff.append(("dETC", "ETC"))
    eq("e_balV_E", ["dETV", "HST1", "FTM1", "HST2", "FTM2", "HST3", "FTM3",
                    "HST5", "FTM5", "HST9", "FTM9", "HST6", "FTM6"])
    diff.append(("dETV", "ETV"))

    return E, diff


def build_model(sensors=None, name="TEP"):
    """Build an fdt.DiagnosisModel for a chosen XMEAS subset."""
    if sensors is None:
        sensors = list(SENSORS.keys())
    catalogue = dict(SENSORS)
    catalogue.update(OPTIONAL_SENSORS)
    E, diff = tep_equations()

    rels, eqnames = [], []
    for nm, vs in E:
        rels.append(list(vs))
        eqnames.append(nm)
    for dv, v in diff:
        rels.append(fdt.DiffConstraint(dv, v))
        eqnames.append("e_diff_" + v)
    for s in sensors:
        rels.append(list(catalogue[s]) + ["z" + s])
        eqnames.append("e_meas_" + s)

    known = ["z" + s for s in sensors] + list(XMV)
    known_set = set(known)

    unknown, seen = [], set()
    for r in rels:
        vs = r[:2] if fdt.IsDifferentialConstraint(r) else r
        for v in vs:
            if v in known_set or v.startswith("fIDV") or v in seen:
                continue
            seen.add(v)
            unknown.append(v)

    md = {"type": "VarStruc", "x": unknown, "f": FAULTS,
          "z": known, "rels": rels}
    model = fdt.DiagnosisModel(md, name=name)
    model.eqnames = eqnames
    return model


if __name__ == "__main__":
    m = build_model()
    m.Lint()
    print("ne = %d  nx = %d  nz = %d  nf = %d  redundancy = %d"
          % (m.ne(), m.nx(), m.nz(), m.nf(), m.Redundancy()))
