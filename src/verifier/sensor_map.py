"""
Which measurements each residual needs.

The sensor-ablation experiment removes instruments and asks what the
verifier can still do.  A residual survives only if every channel feeding it
survives, including the channels feeding the intermediate quantities it is
built on -- the separator VLE inversion, the stripper split, the reaction
rates.  The closure below is derived from the computation in
residuals_tep.py, quantity by quantity, so that removing a sensor removes
exactly the residuals that genuinely stop being computable.

XMV channels are always available: they are controller outputs written by
the DCS, not field instruments, so an instrumentation downgrade does not
remove them.
"""
from __future__ import annotations

ANALYSER_PURGE = ["XMEAS%d" % k for k in range(29, 37)]
ANALYSER_FEED = ["XMEAS%d" % k for k in range(23, 29)]
ANALYSER_PROD = ["XMEAS%d" % k for k in range(37, 42)]

# intermediate quantity -> channels it needs (transitively expanded below)
NEEDS = {
    "XLS": ANALYSER_PURGE + ["XMEAS13", "XMEAS11"],
    "DLS": ["XLS"],
    "DLC": ANALYSER_PROD + ["XMEAS18"],
    "FTM11": ["XMEAS14", "DLS"],
    "FTM13": ["XMEAS17", "DLC"],
    "FTM8": ["XMEAS5", "XMEAS10", "FTM11"],
    "XST8": ["FTM8", "XLS"] + ANALYSER_PURGE,
    "XMWS8": ["XST8"],
    "XMWS9": ANALYSER_PURGE,
    "FTM5": ["XMEAS4", "FTM11", "FTM13", "XLS"] + ANALYSER_PROD,
    "XST5": ["FTM5"],
    "XVV": ANALYSER_FEED + ["XST5", "FTM5", "XMEAS5", "XMEAS6"],
    "XMWS6": ["XVV"],
    "RR": ["XST8", "FTM8", "XVV", "XMEAS6"],
    "INVENTORY": ["XST8", "XLS", "XLC_FROM_PROD", "XVV", "XMEAS8",
                  "XMEAS12", "XMEAS15", "XMEAS7", "XMEAS13", "XMEAS16",
                  "XMEAS9", "XMEAS11", "XMEAS18"],
    "XLC_FROM_PROD": ANALYSER_PROD,
    "QUR": ["XMEAS21", "XMEAS9", "XMEAS8"],
    "QUS": ["XMEAS22", "XMEAS9", "FTM8"],
}

RESIDUAL_NEEDS = {
    "r_v1_Dfeed": ["XMEAS2"],
    "r_v2_Efeed": ["XMEAS3"],
    "r_v3_Afeed": ["XMEAS1"],
    "r_v4_ACfeed": ["XMEAS4"],
    "r_v5_recycle": ["XMEAS5", "XMEAS16", "XMEAS13", "XMWS9"],
    "r_v6_purge": ["XMEAS10", "XMEAS13", "XMWS9"],
    "r_v7_sepflow": ["FTM11"],
    "r_v8_product": ["FTM13"],
    "r_steam": ["XMEAS19", "XMEAS18"],
    "r_cpdh": ["XMEAS20", "XMEAS16", "XMEAS13", "XMEAS11", "XMWS9"],
    "r_f6_reacfeed": ["XMEAS6", "XMEAS16", "XMEAS7", "XMWS6"],
    "r_f8_reacout": ["FTM8", "XMEAS7", "XMEAS13", "XMWS8"],
    "r_cwR": ["QUR"],
    "r_cwS": ["QUS"],
    "r_balA": ["XMEAS1", "XMEAS3", "XMEAS4", "XMEAS10", "FTM13",
               "XMWS9", "XLC_FROM_PROD", "INVENTORY"],
    "r_balB": ["XMEAS2", "XMEAS1", "XMEAS4", "XMEAS10", "XMWS9",
               "INVENTORY"],
    "r_balC": ["XMEAS4", "XMEAS10", "FTM13", "XMWS9", "XLC_FROM_PROD",
               "INVENTORY"],
    "r_balD": ["XMEAS2", "XMEAS3", "XMEAS10", "FTM13", "XMWS9",
               "XLC_FROM_PROD", "INVENTORY"],
    "r_dec2": ["r_balA", "r_balB"],
    "r_energy": ["XMEAS2", "XMEAS3", "XMEAS1", "FTM5", "XST5", "XMEAS5",
                 "XST8", "FTM8", "RR", "QUR", "INVENTORY", "XMEAS20"],
    "r_sep_En": ["FTM8", "XST8", "XMEAS5", "XMEAS20", "FTM11", "XLS",
                 "QUS", "INVENTORY"],
    "r_strip_En": ["XMEAS4", "FTM11", "FTM5", "FTM13", "XMEAS19",
                   "XLS", "XLC_FROM_PROD", "INVENTORY"],
    "r_kin1": ["RR", "XMEAS9", "XST8", "XMEAS7", "XMEAS8"],
    "r_kin2": ["RR", "XMEAS9", "XST8", "XMEAS7", "XMEAS8"],
    "r_kin3": ["RR", "XMEAS9", "XST8", "XMEAS7", "XMEAS8"],
    "r_kin4": ["RR", "XMEAS9", "XST8", "XMEAS7", "XMEAS8"],
}


def _expand(item, seen=None):
    seen = seen or set()
    if item in seen:
        return set()
    seen.add(item)
    if item.startswith("XMEAS"):
        return {item}
    src = NEEDS.get(item) or RESIDUAL_NEEDS.get(item)
    if src is None:
        return set()
    out = set()
    for x in src:
        out |= _expand(x, seen)
    return out


CHANNELS_REQUIRED = {r: sorted(_expand(r)) for r in RESIDUAL_NEEDS}


def available_residuals(sensors):
    """Residuals still computable when only `sensors` remain."""
    keep = set(sensors)
    return [r for r, need in CHANNELS_REQUIRED.items()
            if set(need) <= keep]


if __name__ == "__main__":
    for r, need in CHANNELS_REQUIRED.items():
        print("%-16s %2d channels" % (r, len(need)))
