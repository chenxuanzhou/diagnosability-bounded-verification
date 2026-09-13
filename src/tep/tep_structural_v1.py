"""
Structural model of the Tennessee Eastman Process (Downs & Vogel, 1993).

Granularity: unit-level component / energy / momentum balances, valve
characteristics, VLE relations, and one measurement equation per XMEAS
channel.  Manipulated variables XMV(1..12) are treated as KNOWN inputs
(they are logged), so no controller equations are included -- this is the
standard open-loop-diagnosis convention.

Fault modes are the benchmark disturbances IDV(1..15) and IDV(21).
IDV(16..20) are left unmodelled: the original benchmark specifies them as
"unknown", so they cannot be mapped to a structural location without
reverse-engineering the Fortran source.  This exclusion is reported.

Documented simplifications (recorded in docs/tep_model_notes.md):
  S1 purge composition == separator overhead composition, so y9_i is not
     carried as a separate variable; XMEAS(29..36) measure y8_i directly.
  S2 recycle composition == separator overhead composition, likewise.
  S3 reactor VLE is lumped into one relation per component,
     x7_i = phi_i(Nr_i, Nr_tot, T_r, P_r), rather than an explicit
     two-phase split.
  S4 separator / stripper liquid holdup dominates the vapour holdup, so
     the liquid composition is taken as the inventory composition.
"""
from __future__ import annotations

import faultdiagnosistoolbox as fdt

COMP = ["A", "B", "C", "D", "E", "F", "G", "H"]

# ---------------------------------------------------------------- sensors
# sensor name -> unknown variable it observes
SENSORS: dict = {
    "XMEAS1": "F1",          # A feed (stream 1)
    "XMEAS2": "F2",          # D feed (stream 2)
    "XMEAS3": "F3",          # E feed (stream 3)
    "XMEAS4": "F4",          # A and C feed (stream 4)
    "XMEAS5": "F8",          # separator overhead / recycle flow (stream 8)
    "XMEAS6": "F6",          # reactor feed rate (stream 6)
    "XMEAS7": "P_r",         # reactor pressure
    "XMEAS8": "L_r",         # reactor level
    "XMEAS9": "T_r",         # reactor temperature
    "XMEAS10": "F9",         # purge rate (stream 9)
    "XMEAS11": "T_s",        # separator temperature
    "XMEAS12": "L_s",        # separator level
    "XMEAS13": "P_s",        # separator pressure
    "XMEAS14": "F10",        # separator underflow (stream 10)
    "XMEAS15": "L_t",        # stripper level
    "XMEAS16": "P_t",        # stripper pressure
    "XMEAS17": "F11",        # stripper underflow (stream 11)
    "XMEAS18": "T_t",        # stripper temperature
    "XMEAS19": "F_steam",    # stripper steam flow
    "XMEAS20": "W_comp",     # compressor work
    "XMEAS21": "T_cwr_out",  # reactor cooling water outlet temperature
    "XMEAS22": "T_cwc_out",  # condenser cooling water outlet temperature
}
for _k, _c in enumerate(COMP[:6]):          # reactor feed analysis, A..F
    SENSORS["XMEAS%d" % (23 + _k)] = "x6" + _c
for _k, _c in enumerate(COMP):              # purge analysis, A..H
    SENSORS["XMEAS%d" % (29 + _k)] = "y8" + _c
for _k, _c in enumerate(COMP[3:]):          # product analysis, D..H
    SENSORS["XMEAS%d" % (37 + _k)] = "x11" + _c

XMV = {
    "XMV1": "u_D", "XMV2": "u_E", "XMV3": "u_A", "XMV4": "u_AC",
    "XMV5": "u_rec", "XMV6": "u_purge", "XMV7": "u_sep", "XMV8": "u_prod",
    "XMV9": "u_steam", "XMV10": "u_cwr", "XMV11": "u_cwc", "XMV12": "u_agit",
}

# ------------------------------------------------------- fault dictionary
# IDV -> (description, literature difficulty tag)
# Difficulty tags follow the consensus of the TEP benchmarking literature
# (Chiang/Russell/Braatz 2001; Yin et al. 2012; Onel et al. 2019): faults
# reported with near-100% detection rate are "easy"; IDV 3, 9, 15 (and 21)
# are the group repeatedly reported as near-undetectable / excluded.
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
    "IDV21": ("Valve for stream 4 fixed at steady-state position", "hard"),
}
IDVS = list(IDV_INFO.keys())
FAULTS = ["f" + k for k in IDVS]   # symbol used inside the equations


def _eq(name, variables):
    return (name, list(variables))


def tep_equations():
    """Return (equations, differential-constraint pairs).

    equations : list of (name, [variable names]).
    diff      : list of (dvar, var) pairs turned into DiffConstraints.
    """
    E = []
    diff = []

    # ------------------------------------------------ feed specification
    # stream 4 composition has two degrees of freedom: A/C ratio and x_B
    # IDV8 drives BOTH composition degrees of freedom; fdt requires every
    # fault symbol in exactly one equation, so it enters through the
    # intermediate variable dev8 (structurally equivalent).
    E.append(_eq("e_dev8", ["dev8", "fIDV8"]))
    E.append(_eq("e_d_ac", ["d_ac", "fIDV1", "dev8"]))
    E.append(_eq("e_d_b", ["d_b", "fIDV2", "dev8"]))
    E.append(_eq("e_x4A", ["x4A", "d_ac", "d_b"]))
    E.append(_eq("e_x4B", ["x4B", "d_b"]))
    E.append(_eq("e_x4C", ["x4C", "d_ac", "d_b"]))
    # feed temperatures (the benchmark has no feed-temperature sensors)
    E.append(_eq("e_T1", ["T1"]))
    E.append(_eq("e_T2", ["T2", "fIDV3", "fIDV9"]))
    E.append(_eq("e_T3", ["T3"]))
    E.append(_eq("e_T4", ["T4", "fIDV10"]))
    # cooling water inlet temperatures (unmeasured)
    E.append(_eq("e_Tcwr_in", ["T_cwr_in", "fIDV4", "fIDV11"]))
    E.append(_eq("e_Tcwc_in", ["T_cwc_in", "fIDV5", "fIDV12"]))
    # C header supply pressure (unmeasured)
    E.append(_eq("e_Phdr4", ["P_hdr4", "fIDV7"]))

    # ------------------------------------------------------------ valves
    E.append(_eq("e_v1", ["F1", "u_A", "P_r", "fIDV6"]))
    E.append(_eq("e_v2", ["F2", "u_D", "P_r"]))
    E.append(_eq("e_v3", ["F3", "u_E", "P_r"]))
    E.append(_eq("e_v4", ["F4", "u_AC", "P_hdr4", "P_t", "fIDV21"]))
    E.append(_eq("e_v6", ["F9", "u_purge", "P_s"]))
    E.append(_eq("e_v7", ["F10", "u_sep", "P_s", "P_t"]))
    E.append(_eq("e_v8", ["F11", "u_prod", "P_t"]))
    E.append(_eq("e_v9", ["F_steam", "u_steam"]))
    E.append(_eq("e_v10", ["w_cwr", "u_cwr", "fIDV14"]))
    E.append(_eq("e_v11", ["w_cwc", "u_cwc", "fIDV15"]))

    # ---------------------------------------------- reactor feed mixing M1
    E.append(_eq("e_M_tot", ["F6", "F1", "F2", "F3", "F5", "F12"]))
    pure = {"A": "F1", "D": "F2", "E": "F3"}
    for c in COMP:
        v = ["F6", "x6" + c, "F5", "y8" + c, "F12", "y12" + c]
        if c in pure:
            v.append(pure[c])
        E.append(_eq("e_M_" + c, v))
    E.append(_eq("e_M_En", ["F6", "h6", "F1", "h1", "F2", "h2",
                            "F3", "h3", "F5", "h5", "F12", "h12"]))
    E.append(_eq("e_h1", ["h1", "T1"]))
    E.append(_eq("e_h2", ["h2", "T2"]))
    E.append(_eq("e_h3", ["h3", "T3"]))
    E.append(_eq("e_h6", ["h6", "T6"] + ["x6" + c for c in COMP]))

    # ----------------------------------------------------------- reactor
    # R1 A+C+D->G   R2 A+C+E->H   R3 A+E->F   R4 3D->2F
    nu = {
        "A": ["R1", "R2", "R3"],
        "B": [],
        "C": ["R1", "R2"],
        "D": ["R1", "R4"],
        "E": ["R2", "R3"],
        "F": ["R3", "R4"],
        "G": ["R1"],
        "H": ["R2"],
    }
    for c in COMP:
        E.append(_eq("e_R_" + c,
                     ["dNr" + c, "F6", "x6" + c, "F7", "x7" + c] + nu[c]))
        diff.append(("dNr" + c, "Nr" + c))
    rate_vars = [["x7A", "x7C", "x7D"], ["x7A", "x7C", "x7E"],
                 ["x7A", "x7E"], ["x7D"]]
    # IDV13 is a slow drift of the kinetics; it enters all four rate laws
    # through the intermediate variable k_drift.
    E.append(_eq("e_kdrift", ["k_drift", "fIDV13"]))
    for j, rv in enumerate(rate_vars, start=1):
        E.append(_eq("e_rate%d" % j, ["R%d" % j, "T_r", "P_r", "k_drift"] + rv))
    for c in COMP:
        E.append(_eq("e_R_x7" + c,
                     ["x7" + c, "Nr" + c, "Nr_tot", "T_r", "P_r"]))
    E.append(_eq("e_R_Ntot", ["Nr_tot"] + ["Nr" + c for c in COMP]))
    E.append(_eq("e_R_En", ["dUr", "F6", "h6", "F7", "h7",
                            "R1", "R2", "R3", "R4", "Q_r"]))
    diff.append(("dUr", "Ur"))
    E.append(_eq("e_R_U", ["Ur", "Nr_tot", "T_r"]))
    E.append(_eq("e_R_h7", ["h7", "T7"] + ["x7" + c for c in COMP]))
    E.append(_eq("e_R_T7", ["T7", "T_r"]))
    E.append(_eq("e_R_Vliq", ["V_rliq", "Nr_tot", "T_r", "P_r"]))
    E.append(_eq("e_R_Vvap", ["V_rvap", "V_rliq"]))
    E.append(_eq("e_R_L", ["L_r", "V_rliq"]))
    E.append(_eq("e_R_P", ["P_r", "Nr_tot", "T_r", "V_rvap"]))
    E.append(_eq("e_R_Qr", ["Q_r", "T_r", "T_cwr_avg", "u_agit"]))
    E.append(_eq("e_R_Qcw", ["Q_r", "w_cwr", "T_cwr_out", "T_cwr_in"]))
    E.append(_eq("e_R_Tavg", ["T_cwr_avg", "T_cwr_in", "T_cwr_out"]))

    # --------------------------------------------------------- condenser
    E.append(_eq("e_C_En", ["Q_c", "F7", "h7", "h7c"]))
    E.append(_eq("e_C_h7c", ["h7c", "T7c"] + ["x7" + c for c in COMP]))
    E.append(_eq("e_C_UA", ["Q_c", "T7", "T7c", "T_cwc_avg"]))
    E.append(_eq("e_C_Qcw", ["Q_c", "w_cwc", "T_cwc_out", "T_cwc_in"]))
    E.append(_eq("e_C_Tavg", ["T_cwc_avg", "T_cwc_in", "T_cwc_out"]))

    # --------------------------------------------------------- separator
    for c in COMP:
        E.append(_eq("e_S_" + c, ["dNs" + c, "F7", "x7" + c,
                                  "F8", "y8" + c, "F10", "x10" + c]))
        diff.append(("dNs" + c, "Ns" + c))
        E.append(_eq("e_S_x10" + c, ["x10" + c, "Ns" + c, "Ns_tot"]))
        E.append(_eq("e_S_vle" + c, ["y8" + c, "x10" + c, "T_s", "P_s"]))
    E.append(_eq("e_S_Ntot", ["Ns_tot"] + ["Ns" + c for c in COMP]))
    E.append(_eq("e_S_ysum", ["y8" + c for c in COMP]))
    E.append(_eq("e_S_En", ["dUs", "F7", "h7c", "F8", "h8", "F10", "h10"]))
    diff.append(("dUs", "Us"))
    E.append(_eq("e_S_U", ["Us", "Ns_tot", "T_s"]))
    E.append(_eq("e_S_h8", ["h8", "T_s"] + ["y8" + c for c in COMP]))
    E.append(_eq("e_S_h10", ["h10", "T_s"] + ["x10" + c for c in COMP]))
    E.append(_eq("e_S_L", ["L_s", "Ns_tot", "T_s"]))

    # ------------------------------------------------ compressor / purge
    E.append(_eq("e_K_split", ["F8", "F5", "F9"]))
    E.append(_eq("e_K_W", ["W_comp", "F5", "T_s", "P_s", "P_r", "u_rec"]))
    E.append(_eq("e_K_T5", ["T5", "T_s", "P_s", "P_r"]))
    E.append(_eq("e_K_h5", ["h5", "T5"] + ["y8" + c for c in COMP]))

    # ---------------------------------------------------------- stripper
    feed4 = {"A": "x4A", "B": "x4B", "C": "x4C"}
    for c in COMP:
        v = ["dNt" + c, "F10", "x10" + c, "F11", "x11" + c, "F12", "y12" + c]
        if c in feed4:
            v += ["F4", feed4[c]]
        E.append(_eq("e_T_" + c, v))
        diff.append(("dNt" + c, "Nt" + c))
        E.append(_eq("e_T_x11" + c, ["x11" + c, "Nt" + c, "Nt_tot"]))
        E.append(_eq("e_T_vle" + c, ["y12" + c, "x11" + c, "T_t", "P_t"]))
    E.append(_eq("e_T_Ntot", ["Nt_tot"] + ["Nt" + c for c in COMP]))
    E.append(_eq("e_T_ysum", ["y12" + c for c in COMP]))
    E.append(_eq("e_T_En", ["dUt", "F10", "h10", "F4", "h4",
                            "F11", "h11", "F12", "h12", "Q_steam"]))
    diff.append(("dUt", "Ut"))
    E.append(_eq("e_T_U", ["Ut", "Nt_tot", "T_t"]))
    E.append(_eq("e_T_Q", ["Q_steam", "F_steam"]))
    E.append(_eq("e_T_L", ["L_t", "Nt_tot", "T_t"]))
    E.append(_eq("e_T_h4", ["h4", "T4", "x4A", "x4B", "x4C"]))
    E.append(_eq("e_T_h11", ["h11", "T_t"] + ["x11" + c for c in COMP]))
    E.append(_eq("e_T_h12", ["h12", "T_t"] + ["y12" + c for c in COMP]))

    return E, diff


def build_model(sensors=None, name="TEP"):
    """Build a fdt.DiagnosisModel for a given sensor subset.

    sensors : list of XMEAS names to keep (None keeps all 41).
    Returns the model; model.eqnames holds the equation names in row order.
    """
    if sensors is None:
        sensors = list(SENSORS.keys())
    E, diff = tep_equations()

    rels, eqnames = [], []
    for nm, vars_ in E:
        rels.append(list(vars_))
        eqnames.append(nm)
    for dv, v in diff:
        rels.append(fdt.DiffConstraint(dv, v))
        eqnames.append("e_diff_" + v)
    for s in sensors:
        rels.append([SENSORS[s], "z" + s])
        eqnames.append("e_meas_" + s)

    known = ["z" + s for s in sensors] + list(XMV.values())
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
    print("ne = %d  nx = %d  nz = %d  nf = %d" % (m.ne(), m.nx(), m.nz(), m.nf()))
    print("redundancy =", m.Redundancy())
