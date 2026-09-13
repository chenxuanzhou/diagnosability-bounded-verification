"""
The proposition space, and why it is defined over fault LOCATIONS.

A proposition an operator acts on is a physical location and mode: "the
condenser cooling water valve is not delivering the commanded flow".  It is
not "...as a step rather than as a random drift".  The Tennessee Eastman
benchmark ships several disturbances that are the same physical deviation
in two temporal flavours -- IDV(3) and IDV(9) are both the D feed
temperature, IDV(4) and IDV(11) are both the reactor coolant inlet
temperature, IDV(5) and IDV(12) are both the condenser coolant inlet
temperature.  Treating those as separate propositions would be measuring
whether a method can tell a step from a random walk, which is a question
about temporal statistics and has nothing to do with the diagnosability
bound this paper is about.

So the proposition space is the 14 fault locations the reference
implementation actually exercises, plus "no fault".  Each location is a
union of the benchmark disturbances that act on the same model equation,
and the structural indiscernibility quotient is computed over locations.

This granularity also matters for honesty in the other direction.  Under a
location-level proposition space the surviving non-singleton classes are
NOT step-versus-drift pairs that a variance test could separate; they are
genuinely different physical causes that share a residual signature:

    {reactor coolant inlet temperature, reactor coolant valve}
    {condenser coolant inlet temperature, condenser coolant valve}

Neither the coolant flow nor the coolant inlet temperature is instrumented
on this plant, so the two causes enter the one available relation the same
way.  That is the ambiguity the operator actually faces, and the one a
maintenance record resolves in a sentence.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.tep import tep_constants as C                      # noqa: E402

NO_FAULT = "NF"

# location id -> (benchmark disturbances, operator-facing description)
LOCATIONS = {
    "L_feed4_AC": ([1], "A/C ratio in the A and C feed stream deviates"),
    "L_feed4_B": ([2], "B composition in the A and C feed stream deviates"),
    "L_feed4_multi": ([8], "A, B and C compositions in the A and C feed "
                           "stream all deviate"),
    "L_Dfeed_temp": ([3, 9], "D feed temperature deviates"),
    "L_Cfeed_temp": ([10], "A and C feed temperature deviates"),
    "L_reac_cw_temp": ([4, 11], "reactor cooling water inlet temperature "
                                "deviates"),
    "L_cond_cw_temp": ([5, 12], "condenser cooling water inlet temperature "
                                "deviates"),
    "L_Afeed_loss": ([6], "A feed supply is lost"),
    "L_Cheader": ([7], "C header pressure loss limits the A and C feed"),
    "L_kinetics": ([13], "reaction kinetics drift"),
    "L_reac_cw_valve": ([14], "reactor cooling water valve does not deliver "
                              "the commanded flow"),
    "L_cond_cw_valve": ([15], "condenser cooling water valve does not "
                              "deliver the commanded flow"),
    "L_steam_valve": ([16], "stripper steam valve and reboiler heat transfer "
                            "deviate"),
    "L_other_valves": ([19], "compressor recycle, separator underflow, "
                             "stripper product or stripper steam valve does "
                             "not deliver the commanded position"),
    # The three ramping disturbances.  They were wrongly excluded at first as
    # zero-amplitude; see the note in tep_constants.SSPAN.
    "L_reac_UA": ([17], "reactor heat transfer coefficient deviates"),
    "L_cond_UA": ([18], "condenser heat transfer coefficient deviates"),
    "L_reac_out_flow": ([20], "reactor to separator flow coefficient "
                              "deviates"),
}
LOCATION_IDS = list(LOCATIONS)
PROPOSITIONS = [NO_FAULT] + LOCATION_IDS

IDV_TO_LOCATION = {}
for _loc, (_idvs, _d) in LOCATIONS.items():
    for _k in _idvs:
        IDV_TO_LOCATION[_k] = _loc

# every disturbance the reference implementation exercises must be covered
assert set(IDV_TO_LOCATION) == set(C.ACTIVE_IDV), (
    set(IDV_TO_LOCATION) ^ set(C.ACTIVE_IDV))


def location_of(idv):
    return IDV_TO_LOCATION[int(idv)]


def catalogue_text():
    lines = ["%s: the plant is operating normally, no fault" % NO_FAULT]
    lines += ["%s: %s" % (k, v[1]) for k, v in LOCATIONS.items()]
    return "\n".join(lines)


def fsm_over_locations(fsm, fault_names):
    """Collapse an IDV-indexed FSM to a location-indexed one.

    A residual responds to a location if it responds to any disturbance at
    that location.  Column equality then defines the quotient over
    propositions.
    """
    import numpy as np
    cols = []
    for loc in LOCATION_IDS:
        idvs = LOCATIONS[loc][0]
        idx = [fault_names.index("fIDV%d" % k) for k in idvs]
        cols.append((fsm[:, idx].sum(1) > 0).astype(int))
    return np.stack(cols, axis=1)


def classes_from_fsm(fsm_loc):
    """Group locations with identical FSM columns."""
    groups = {}
    for j, loc in enumerate(LOCATION_IDS):
        groups.setdefault(tuple(fsm_loc[:, j]), []).append(loc)
    return sorted(groups.values(), key=lambda b: b[0])


def structural_classes():
    """Indiscernibility classes over locations, from the structural model.

    This is the theoretical bound, computed with no data and no residual
    bank: the finest partition of the proposition space that ANY verifier
    built on the installed sensor set could achieve.  Class-level accuracy
    is scored against it, because a method cannot be blamed for failing to
    split what is provably unsplittable.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.structural import equiv
    from src.tep import tep_structural as tep

    model = tep.build_model()
    res = equiv.quotient_summary(model, tep.FAULTS)
    blocks = []
    for b in res["blocks"]:
        locs = sorted({IDV_TO_LOCATION[int(f[4:])] for f in b
                       if int(f[4:]) in IDV_TO_LOCATION})
        if locs:
            blocks.append(locs)
    # merge blocks that share a location (a location can only be in one)
    merged = []
    for b in blocks:
        hit = [m for m in merged if set(m) & set(b)]
        if hit:
            hit[0][:] = sorted(set(hit[0]) | set(b))
        else:
            merged.append(sorted(b))
    return merged


def class_map():
    """location -> the set of locations it is indiscernible from."""
    out = {NO_FAULT: {NO_FAULT}}
    for b in structural_classes():
        for loc in b:
            out[loc] = set(b)
    return out
