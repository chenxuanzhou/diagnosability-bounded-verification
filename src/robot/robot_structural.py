"""
Structural model of the six-axis collaborative robot behind the voraus-AD
dataset, plus the mapping from the dataset's anomaly *categories* to
structural fault modes (Gate A).

Granularity is the one the experiment plan prescribes: per-axis electrical,
electromagnetic, gear, friction and drivetrain equations, a dense rigid-body
coupling row per joint, and one measurement equation per independently
sensed channel.

Sensor-set policy.  The dataset ships ~130 channels, but several are
algebraically derived from others by the drive firmware and therefore carry
no independent structural information:

  * motor_torque_i            = Kt_i * motor_iq_i
  * power_motor_el_i          = motor_voltage_i * motor_iq_i
  * power_motor_mech_i        = motor_torque_i * motor_velocity_i
  * power_load_mech_i         = joint_torque_i * joint_velocity_i
  * computed_torque_i,
    computed_inertia_i        = the controller's own inverse-dynamics model
  * target_*                  = reference trajectory, a known input

Including them would manufacture redundancy the hardware does not have, so
the default sensor set keeps only the independently sensed channels.  The
excluded list is reported with the results.  target_* signals enter as
known inputs, exactly as XMV does for the Tennessee Eastman model.

Assumption flagged for the paper: joint_position_i / joint_velocity_i are
taken to be link-side (secondary) encoder readings independent of the
motor-side encoder.  If on this hardware they are derived as
motor_position_i / N_i, the pair collapses to a single sensor; the
sensor-ablation experiment covers that case explicitly as the
"no link-side encoder" rung.
"""
from __future__ import annotations

import faultdiagnosistoolbox as fdt

NAXIS = 6
AX = list(range(1, NAXIS + 1))

# ------------------------------------------------------------ fault modes
# name -> (description, whether the voraus-AD dataset contains recordings)
FAULT_INFO = {}
for _i in AX:
    FAULT_INFO["friction%d" % _i] = (
        "Axis %d drivetrain friction increase (b_i, tau_c_i)" % _i, True)
for _i in AX:
    FAULT_INFO["addedmass%d" % _i] = (
        "Additional mass mounted on link %d" % _i, True)
for _i in AX:
    FAULT_INFO["commutation%d" % _i] = (
        "Axis %d motor commutation offset (torque constant error)" % _i, True)
for _i in AX:
    FAULT_INFO["extforce%d" % _i] = (
        "External contact torque on axis %d (collision / entanglement)" % _i,
        True)
FAULT_INFO["payload"] = ("Tool payload mass deviation", True)

FAULTS = ["f" + k for k in FAULT_INFO]
FAULT_NAMES = list(FAULT_INFO.keys())

# --------------------------------------------------------- Gate A mapping
# dataset category -> (structural fault modes, rationale, included?)
CATEGORY_MAP = {
    "AXIS_FRICTION": (
        ["friction%d" % i for i in AX],
        "Variants are axis-resolved (A1_A .. A6_B); the anomaly is a "
        "deviation of the friction parameters in the axis friction law.",
        True),
    "AXIS_WEIGHT": (
        ["addedmass%d" % i for i in AX],
        "Variants are axis-resolved with three magnitudes (A1_115G .. "
        "A6_500G); the anomaly is an inertial-parameter deviation of the "
        "corresponding link.",
        True),
    "MOTOR_COMMUTATION": (
        ["commutation%d" % i for i in AX],
        "Variants are axis-resolved (A1_HIGH .. A6_MEDIUM); a commutation "
        "offset scales the electromagnetic torque relation tau_m = Kt*iq.",
        True),
    "COLLISION_FOAM": (
        ["extforce%d" % i for i in AX],
        "Contact with a foam obstacle injects an external torque into the "
        "rigid-body row of whichever axes carry the contact wrench.",
        True),
    "COLLISION_CABLE": (
        ["extforce%d" % i for i in AX],
        "As COLLISION_FOAM; the obstacle differs, the structural location "
        "does not.",
        True),
    "COLLISION_CARTON": (
        ["extforce%d" % i for i in AX],
        "As COLLISION_FOAM.",
        True),
    "ENTANGLED": (
        ["extforce%d" % i for i in AX],
        "A snagged cable applies a sustained external wrench; same "
        "structural location as a collision.",
        True),
    "MISS_CAN": (
        ["payload"],
        "The gripper carries no workpiece, so the payload mass parameter "
        "deviates from nominal by the full can mass.",
        True),
    "LOSE_CAN": (
        ["payload"],
        "The workpiece is released mid-trajectory: the same payload "
        "parameter, with a step in time rather than at t=0.",
        True),
    "CAN_WEIGHT": (
        ["payload"],
        "The can is heavier or lighter than nominal: payload parameter "
        "deviation.",
        True),
    "INVALID_POSITION": (
        [], "EXCLUDED. The plant is healthy; the reference trajectory is "
            "wrong. This is a deviation of a known input, not of a model "
            "equation or parameter, so it has no structural fault location.",
        False),
    "WOBBLING_STATION": (
        [], "EXCLUDED. A compliant/moving base adds degrees of freedom that "
            "the fixed-base model does not contain. It is unmodelled "
            "dynamics, not a parameter deviation, and belongs to the "
            "model-mismatch experiment instead.",
        False),
}

# ------------------------------------------------------------- sensor set
# name -> unknown it observes
SENSORS = {}
for _i in AX:
    SENSORS["motor_position_%d" % _i] = "th_m%d" % _i
    SENSORS["motor_velocity_%d" % _i] = "w_m%d" % _i
    SENSORS["joint_position_%d" % _i] = "th_l%d" % _i
    SENSORS["joint_velocity_%d" % _i] = "w_l%d" % _i
    SENSORS["motor_iq_%d" % _i] = "I%d" % _i
    SENSORS["motor_voltage_%d" % _i] = "v%d" % _i
    SENSORS["torque_sensor_a_%d" % _i] = "tau_j%d" % _i
    SENSORS["torque_sensor_b_%d" % _i] = "tau_j%d" % _i
SENSORS["robot_current"] = "I_tot"

EXCLUDED_CHANNELS = {
    "motor_torque_i": "firmware estimate Kt*iq, not an independent sensor",
    "power_motor_el_i": "product of motor_voltage and motor_iq",
    "power_motor_mech_i": "product of motor_torque and motor_velocity",
    "power_load_mech_i": "product of joint torque and joint_velocity",
    "computed_torque_i": "controller inverse-dynamics output",
    "computed_inertia_i": "controller inertia model output",
    "motor_id_i": "d-axis current, regulated to zero, no structural role",
    "supply_voltage_i / brake_voltage_i": "DC-link and brake rails, "
                                          "not in the drivetrain equations",
    "io_current / system_current": "auxiliary supplies outside the drivetrain",
    "target_*": "reference trajectory, entered as a known input instead",
}

# No known inputs are needed: the motor voltage is measured, so the drive's
# control law never has to be written down.  This mirrors the Tennessee
# Eastman convention of treating the manipulated variables as logged inputs
# and keeps the analysis independent of the (proprietary) controller.
KNOWN_INPUTS = []

# Sensor groups for the E6 ablation ladder, ordered from the richest
# instrumentation down to a motor-side-only drive.
SENSOR_GROUPS = {
    "joint_torque": ["torque_sensor_a_%d" % i for i in AX] +
                    ["torque_sensor_b_%d" % i for i in AX],
    "link_encoders": ["joint_position_%d" % i for i in AX] +
                     ["joint_velocity_%d" % i for i in AX],
    "phase_current": ["motor_iq_%d" % i for i in AX],
    "motor_voltage": ["motor_voltage_%d" % i for i in AX],
    "motor_encoders": ["motor_position_%d" % i for i in AX] +
                      ["motor_velocity_%d" % i for i in AX],
    "bus_current": ["robot_current"],
}

# ---------------------------------------------------- candidate new sensors
# Instruments the arm does NOT have, used to separate the classes that an
# instrumentation upgrade can split from the ones that are irreducible.
OPTIONAL_SENSORS = {}
for _i in AX:
    OPTIONAL_SENSORS["NEW_motor_torque_%d" % _i] = "tau_m%d" % _i
    OPTIONAL_SENSORS["NEW_friction_obs_%d" % _i] = "tau_fric%d" % _i
    OPTIONAL_SENSORS["NEW_contact_ft_%d" % _i] = "tau_ext%d" % _i


def robot_equations():
    """Return (equations, diff_pairs) for the six-axis arm."""
    E, diff = [], []

    def eq(name, vs):
        E.append((name, list(vs)))

    for i in AX:
        s = str(i)
        # ---- electrical: v = R*I + L*dI/dt + Ke*w_m
        eq("e_elec" + s, ["v" + s, "I" + s, "dI" + s, "w_m" + s])
        diff.append(("dI" + s, "I" + s))
        # ---- electromagnetic torque: tau_m = Kt*I   (commutation fault)
        eq("e_taum" + s, ["tau_m" + s, "I" + s, "fcommutation" + s])
        # ---- friction: tau_fric = b*w_m + tau_c*sign(w_m)
        eq("e_fric" + s, ["tau_fric" + s, "w_m" + s, "ffriction" + s])
        # ---- motor-side dynamics
        eq("e_mdyn" + s, ["dw_m" + s, "tau_m" + s, "tau_fric" + s,
                          "tau_j" + s])
        diff.append(("dw_m" + s, "w_m" + s))
        # ---- kinematics
        eq("e_thm" + s, ["dth_m" + s, "w_m" + s])
        diff.append(("dth_m" + s, "th_m" + s))
        eq("e_gear" + s, ["w_m" + s, "w_l" + s])
        eq("e_thl" + s, ["dth_l" + s, "w_l" + s])
        diff.append(("dth_l" + s, "th_l" + s))
        eq("e_accl" + s, ["dw_l" + s, "w_l" + s])   # alpha definition
        diff.append(("dw_l" + s, "w_l" + s))
        # ---- fault entry points that span several rigid-body rows
        eq("e_dm" + s, ["dm" + s, "faddedmass" + s])
        eq("e_text" + s, ["tau_ext" + s, "fextforce" + s])
    eq("e_dmpl", ["dm_pl", "fpayload"])

    # ---- rigid-body coupling: row i of M(q)qdd + C(q,qd)qd + g(q) = tau_j - tau_ext
    # The mass matrix of a 6R arm is dense, so every row sees every joint
    # coordinate.  A mass added on link k loads joints 1..k, and the tool
    # payload loads all six.
    qvars = (["th_l%d" % k for k in AX] + ["w_l%d" % k for k in AX] +
             ["dw_l%d" % k for k in AX])
    for i in AX:
        # tau_fric enters the LINK balance as well as the motor balance.
        # The joint torque sensor sits between the gearbox and the link, so
        # it reads the torque delivered to the link; friction downstream of
        # it is subtracted after the measurement and therefore appears in
        # this row.  The first version of this model placed friction only on
        # the motor side.  The signature mismatch that exposed the error, and
        # the measurements that settle it, are in
        # src/robot/diagnose_isolation.py: a friction fault moves its own
        # rigid-body relation 4 to 7 times its threshold while moving its own
        # motor-side relation 0.3 to 1.8 times, and a commutation fault does
        # the opposite.  Correcting the placement raises signature precision
        # on the friction faults from 0.175 to 0.617.
        row = ["tau_j%d" % i, "tau_ext%d" % i, "tau_fric%d" % i] + qvars             + ["dm_pl"]
        row += ["dm%d" % k for k in AX if k >= i]
        eq("e_rbd%d" % i, row)

    # ---- total bus current
    eq("e_Itot", ["I_tot"] + ["I%d" % i for i in AX])

    return E, diff


def build_model(sensors=None, name="voraus6R"):
    """Build an fdt.DiagnosisModel for a chosen sensor subset."""
    if sensors is None:
        sensors = list(SENSORS.keys())
    catalogue = dict(SENSORS)
    catalogue.update(OPTIONAL_SENSORS)
    E, diff = robot_equations()

    rels, eqnames = [], []
    for nm, vs in E:
        rels.append(list(vs))
        eqnames.append(nm)
    for dv, v in diff:
        rels.append(fdt.DiffConstraint(dv, v))
        eqnames.append("e_diff_" + v)
    for s in sensors:
        rels.append([catalogue[s], "z_" + s])
        eqnames.append("e_meas_" + s)

    known = ["z_" + s for s in sensors] + list(KNOWN_INPUTS)
    known_set = set(known)
    fault_set = set(FAULTS)

    unknown, seen = [], set()
    for r in rels:
        vs = r[:2] if fdt.IsDifferentialConstraint(r) else r
        for v in vs:
            if v in known_set or v in fault_set or v in seen:
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
