"""
The proposition space the agent must commit to, and the text shown to it.

The plan's contingency for a collapsing motivation is to force the agent
down to a component-level fault mode rather than letting it answer with a
vague category.  That is enforced here structurally: the answer must be one
entry of this catalogue, or "NF" for no fault.  There is no "process
anomaly" option to hide in.

Descriptions are the disturbance definitions from Downs & Vogel (1993).
They name the physical location but never the expected symptom pattern, so
the catalogue does not leak the answer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.tep import tep_constants as C                      # noqa: E402

NO_FAULT = "NF"

CATALOGUE = {
    "IDV1": "A/C feed ratio changes in the A and C feed stream, with the "
            "B composition held constant (step)",
    "IDV2": "B composition changes in the A and C feed stream, with the "
            "A/C ratio held constant (step)",
    "IDV3": "D feed temperature changes (step)",
    "IDV4": "Reactor cooling water inlet temperature changes (step)",
    "IDV5": "Condenser cooling water inlet temperature changes (step)",
    "IDV6": "A feed is lost from the A feed stream (step)",
    "IDV7": "C header pressure loss reduces the availability of the A and "
            "C feed stream (step)",
    "IDV8": "A, B and C compositions in the A and C feed stream vary "
            "randomly",
    "IDV9": "D feed temperature varies randomly",
    "IDV10": "C feed temperature varies randomly",
    "IDV11": "Reactor cooling water inlet temperature varies randomly",
    "IDV12": "Condenser cooling water inlet temperature varies randomly",
    "IDV13": "Reaction kinetics drift slowly",
    "IDV14": "The reactor cooling water valve sticks",
    "IDV15": "The condenser cooling water valve sticks",
    "IDV16": "The stripper steam valve and reboiler heat transfer vary "
             "randomly",
    "IDV19": "Valves on the compressor recycle, separator underflow, "
             "stripper product and stripper steam lines stick",
    "IDV17": "Reactor heat transfer coefficient varies",
    "IDV18": "Condenser heat transfer coefficient varies",
    "IDV20": "Reactor to separator flow coefficient varies",
}
PROPOSITIONS = [NO_FAULT] + list(CATALOGUE.keys())

# sanity: the catalogue must cover exactly the fault modes the reference
# implementation actually exercises
_expect = set("IDV%d" % k for k in C.ACTIVE_IDV)
assert set(CATALOGUE) == _expect, (set(CATALOGUE) ^ _expect)

CHANNELS = (["XMEAS%d" % i for i in range(1, 42)] +
            ["XMV%d" % i for i in range(1, 13)])

CHANNEL_DESC = {
    "XMEAS1": "A feed flow", "XMEAS2": "D feed flow", "XMEAS3": "E feed flow",
    "XMEAS4": "A and C feed flow", "XMEAS5": "recycle flow",
    "XMEAS6": "reactor feed rate", "XMEAS7": "reactor pressure",
    "XMEAS8": "reactor level", "XMEAS9": "reactor temperature",
    "XMEAS10": "purge rate", "XMEAS11": "separator temperature",
    "XMEAS12": "separator level", "XMEAS13": "separator pressure",
    "XMEAS14": "separator underflow", "XMEAS15": "stripper level",
    "XMEAS16": "stripper pressure", "XMEAS17": "stripper underflow",
    "XMEAS18": "stripper temperature", "XMEAS19": "stripper steam flow",
    "XMEAS20": "compressor work",
    "XMEAS21": "reactor cooling water outlet temperature",
    "XMEAS22": "condenser cooling water outlet temperature",
    "XMV1": "D feed valve", "XMV2": "E feed valve", "XMV3": "A feed valve",
    "XMV4": "A and C feed valve", "XMV5": "compressor recycle valve",
    "XMV6": "purge valve", "XMV7": "separator underflow valve",
    "XMV8": "stripper product valve", "XMV9": "stripper steam valve",
    "XMV10": "reactor cooling water valve",
    "XMV11": "condenser cooling water valve", "XMV12": "agitator speed",
}
for _k, _c in enumerate("ABCDEF"):
    CHANNEL_DESC["XMEAS%d" % (23 + _k)] = "reactor feed %s mol%%" % _c
for _k, _c in enumerate("ABCDEFGH"):
    CHANNEL_DESC["XMEAS%d" % (29 + _k)] = "purge gas %s mol%%" % _c
for _k, _c in enumerate("DEFGH"):
    CHANNEL_DESC["XMEAS%d" % (37 + _k)] = "product %s mol%%" % _c


def catalogue_text():
    lines = ["%s: %s" % (NO_FAULT, "no fault, the plant is operating "
                                   "normally")]
    lines += ["%s: %s" % (k, v) for k, v in CATALOGUE.items()]
    return "\n".join(lines)
