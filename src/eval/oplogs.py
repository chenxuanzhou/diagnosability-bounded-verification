"""
Synthetic operations text, for the complementarity experiment.

The theory says the verifier is provably blind inside an indiscernibility
class.  So any method that does better than chance inside a class must be
reading information that never travelled through a sensor channel.  This
module manufactures exactly that information, under control, so the size of
the effect can be attributed rather than merely observed.

Design rules, all fixed before the experiment was run and recorded in
results/E8/protocol.json:

  * the vocabulary is drawn from the ISO 14224 failure-descriptor set
    (sticking, external leakage of utility medium, abnormal instrument
    reading, plugged or choked, fail to function on demand, overheating,
    vibration, structural deficiency);
  * every entry is generated from a template, never hand written per case;
  * each case is assigned an information level in advance:
      irrelevant  nothing in the log bears on the ambiguity
      weak        the log raises one branch of the ambiguity without
                  settling it
      strong      the log names an event that settles it
  * the mixing proportions are frozen: 40 % irrelevant, 30 % weak,
    30 % strong;
  * no entry ever names a catalogue id, restates a sensor reading, or
    mentions the word "fault"; entries carry maintenance history, utility
    status and work-order text, which is what the sensors cannot see;
  * the generator, its seed and every generated log are published.

The limitation is stated in the paper rather than left for a reviewer: this
text is synthetic.  It is used to isolate the complementarity mechanism at a
controlled information content.  Real operations logs are noisier and more
incomplete and would lower the absolute numbers, but they cannot change the
qualitative conclusion, because the conclusion rests on a proved structural
fact about where the gain must come from.
"""
from __future__ import annotations

import numpy as np

LEVELS = ("irrelevant", "weak", "strong")
MIX = {"irrelevant": 0.40, "weak": 0.30, "strong": 0.30}

# ISO 14224 style failure descriptors used by the templates
DESCRIPTORS = ["sticking", "external leakage, utility medium",
               "abnormal instrument reading", "plugged or choked",
               "fail to function on demand", "overheating", "vibration",
               "structural deficiency"]

FILLER = [
    "Shift handover: routine round completed on the {area} deck, no "
    "findings recorded.",
    "Work order {wo} closed: {item} replaced in the {area} utility room.",
    "Housekeeping note: spill kit restocked at station {n} near the {area}.",
    "Permit {wo} issued for scaffolding erection around the {area} "
    "platform, no process work.",
    "Calibration due list reviewed; next batch of field transmitters "
    "scheduled for quarter {n}.",
    "Operator note: radio channel {n} intermittent on the {area} side, "
    "telecoms informed.",
    "Work order {wo} raised: {desc} reported on the {item} in the "
    "{area} store room, non-process equipment.",
    "Contractor briefing held for the {area} painting campaign starting "
    "next week.",
]
AREAS = ["reactor", "separator", "stripper", "compressor house", "utility",
         "tank farm", "control room"]
ITEMS = ["lighting ballast", "hand rail", "gasket set", "hose reel",
         "eyewash unit", "door closer", "cable tray cover"]

# branch templates: what an entry looks like when it points at one cause
# family.  "loop" is reactor or condenser.
WEAK = {
    "temp": [
        "Utility engineering bulletin: cooling water supply temperature "
        "across the site has been running above seasonal average for "
        "several days; no equipment action raised.",
        "Environmental log: ambient wet bulb elevated since the weekend, "
        "cooling tower approach temperature noted as marginal.",
        "Utility shift log: one cooling tower cell taken out of service "
        "for cleaning, supply header affected site wide.",
    ],
    "valve": [
        "Maintenance planning: the control valve population on the "
        "cooling water services is included in this quarter's actuator "
        "inspection campaign.",
        "Reliability note: instrument air supply to the utility valve "
        "actuators showed low pressure alarms twice this month.",
        "Work order backlog review: several cooling water control valves "
        "carry open findings for {desc}.",
    ],
}
STRONG = {
    ("temp", "reactor"): [
        "Utility shift log {ts}: cooling tower fan serving the reactor "
        "jacket water circuit tripped and is locked out; supply "
        "temperature to the reactor jacket is running above set point "
        "until the motor is replaced.",
        "Work order {wo} ({ts}): heat exchanger on the reactor jacket "
        "water supply isolated for cleaning; the circuit is on the bypass "
        "and is being supplied warm.",
    ],
    ("valve", "reactor"): [
        "Work order {wo} ({ts}): operations report {desc} on the reactor "
        "cooling water control valve; the actuator does not follow the "
        "controller output smoothly. Replacement scheduled.",
        "Maintenance log {ts}: the reactor cooling water control valve was "
        "stroked during the last outage and failed the stroke test with "
        "{desc}; it was returned to service pending parts.",
    ],
    ("temp", "condenser"): [
        "Utility shift log {ts}: the cooling tower cell feeding the "
        "condenser water circuit is out of service for basin repair; "
        "supply temperature to the condenser is elevated.",
        "Work order {wo} ({ts}): the condenser cooling water supply is "
        "temporarily cross connected to the warm return header while the "
        "cold header is repaired.",
    ],
    ("valve", "condenser"): [
        "Work order {wo} ({ts}): {desc} reported on the condenser cooling "
        "water control valve; operations note the valve hangs and then "
        "jumps when the controller moves it.",
        "Maintenance log {ts}: positioner on the condenser cooling water "
        "control valve was found loose during the last inspection and has "
        "not yet been re-calibrated.",
    ],
}

# which branch each location belongs to, and on which loop
BRANCH = {
    "L_reac_cw_temp": ("temp", "reactor"),
    "L_reac_cw_valve": ("valve", "reactor"),
    "L_cond_cw_temp": ("temp", "condenser"),
    "L_cond_cw_valve": ("valve", "condenser"),
}


VALVE_DESC = ["sticking", "fail to function on demand",
              "structural deficiency", "abnormal instrument reading"]


def _fill(t, rng, desc_pool=None):
    return t.format(wo=int(rng.integers(4000, 6000)),
                    ts="2026-09-%02d %02d:%02d" % (rng.integers(1, 12),
                                                   rng.integers(0, 24),
                                                   rng.integers(0, 60)),
                    area=str(rng.choice(AREAS)), item=str(rng.choice(ITEMS)),
                    desc=str(rng.choice(desc_pool or DESCRIPTORS)),
                    n=int(rng.integers(1, 9)))


def assign_levels(n, seed=20260911):
    """Frozen level schedule: a fixed proportion, shuffled by a fixed seed."""
    rng = np.random.default_rng(seed)
    counts = {k: int(round(v * n)) for k, v in MIX.items()}
    counts["irrelevant"] += n - sum(counts.values())
    levels = sum(([k] * v for k, v in counts.items()), [])
    rng.shuffle(levels)
    return levels


def make_log(truth_location, level, seed, n_filler=3):
    """One shift log for a case."""
    rng = np.random.default_rng(seed)
    lines = [_fill(str(rng.choice(FILLER)), rng) for _ in range(n_filler)]
    if truth_location in BRANCH:
        branch, loop = BRANCH[truth_location]
        pool = VALVE_DESC if branch == "valve" else None
        if level == "weak":
            lines.append(_fill(str(rng.choice(WEAK[branch])), rng, pool))
        elif level == "strong":
            lines.append(_fill(str(rng.choice(STRONG[(branch, loop)])), rng,
                               pool))
    rng.shuffle(lines)
    return "\n".join("- " + s for s in lines)
