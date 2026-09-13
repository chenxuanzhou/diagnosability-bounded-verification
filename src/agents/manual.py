"""
A plant fault manual, and keyword retrieval over it, for the RAG baseline.

The entries are written from flowsheet reasoning about the Tennessee
Eastman process: what a given deviation does to inventories, pressures and
the loops that fight it.  None of them was written by looking at the
experimental results in this repository, because an entry that encoded the
measured signature would turn the retrieval baseline into an oracle and the
comparison would be meaningless.  Each entry names the channels an engineer
would look at, which is what retrieval matches against.

The manual is deliberately good.  B1 is meant to be the strongest realistic
non-structural baseline, not a straw man.
"""
from __future__ import annotations

import numpy as np

# location id -> (channels an engineer would look at, manual text)
ENTRIES = {
    "L_feed4_AC": (
        ["XMEAS4", "XMEAS1", "XMV3", "XMV4", "XMEAS29", "XMEAS31", "XMEAS23",
         "XMEAS25"],
        "A/C ratio in the A and C feed stream.\n"
        "The A and C feed is the plant's main source of both reactant A and "
        "reactant C. If the ratio between them shifts while the total flow "
        "holds, the reactor is fed a different A-to-C proportion than the "
        "recipe assumes. A is non-condensable and leaves mainly through the "
        "purge, so a surplus of A accumulates in the gas loop and the A "
        "feed controller backs off its own valve to compensate. Look at the "
        "reactor feed and purge analyses for A and C, and at the A feed "
        "valve position relative to the A feed flow."),
    "L_feed4_B": (
        ["XMEAS30", "XMEAS24", "XMEAS10", "XMV6", "XMEAS7", "XMEAS13"],
        "B composition in the A and C feed stream.\n"
        "B is inert. It enters only with the feeds and leaves only with the "
        "purge, so any change in how much B arrives shows up as an "
        "accumulation of inert in the recycle loop. Pressure rises, the "
        "purge controller opens to hold it, and the purge analysis for B "
        "moves. The reactor feed analysis for B moves with it. Check the "
        "purge rate and its B fraction against the feed rates."),
    "L_feed4_multi": (
        ["XMEAS29", "XMEAS30", "XMEAS31", "XMEAS23", "XMEAS24", "XMEAS25",
         "XMEAS10", "XMEAS7"],
        "A, B and C compositions in the A and C feed stream all varying.\n"
        "When more than one feed component wanders at once the loop "
        "inventories move together and no single analyser tells the whole "
        "story. Expect several feed and purge analyses to be restless at "
        "the same time, with the pressure and purge loops working "
        "continuously. Distinguish this from a single-component shift by "
        "checking whether the A, B and C analyses move independently of "
        "each other or in a fixed proportion."),
    "L_Dfeed_temp": (
        ["XMEAS2", "XMV1", "XMEAS9", "XMV10", "XMEAS21"],
        "D feed temperature.\n"
        "The D feed enters the reactor feed mixing zone. A change in its "
        "temperature changes the enthalpy arriving at the reactor, and the "
        "reactor temperature controller absorbs it by trimming the cooling "
        "water valve. The reactor temperature itself barely moves, because "
        "it is controlled; the evidence is in the cooling duty needed to "
        "hold it. There is no thermometer on this feed, so the deviation "
        "cannot be read directly. Check the reactor cooling water valve and "
        "outlet temperature against the D feed rate."),
    "L_Cfeed_temp": (
        ["XMEAS4", "XMEAS18", "XMEAS19", "XMV9", "XMEAS17"],
        "A and C feed temperature.\n"
        "This stream enters the base of the stripper as stripping vapour, "
        "so its temperature lands first on the stripper energy balance. The "
        "stripper temperature controller compensates with steam, and the "
        "steam demand moves before the stripper temperature does. There is "
        "no thermometer on this feed. Check the stripper steam flow and "
        "valve against the A and C feed rate."),
    "L_reac_cw_temp": (
        ["XMEAS21", "XMV10", "XMEAS9", "XMEAS8"],
        "Reactor cooling water inlet temperature.\n"
        "Warmer or cooler cooling water changes the temperature driving "
        "force across the reactor jacket. The reactor temperature loop "
        "restores the reactor temperature by moving the cooling water "
        "valve, so the reactor temperature stays put while the valve and "
        "the cooling water outlet temperature move. Neither the inlet "
        "temperature nor the cooling water flow is instrumented on this "
        "plant, so a change in supply temperature and a valve that is not "
        "delivering its commanded flow enter the one available relation in "
        "the same way. Treat them as a pair until an external record "
        "settles which it is."),
    "L_cond_cw_temp": (
        ["XMEAS22", "XMV11", "XMEAS11", "XMEAS12", "XMEAS13"],
        "Condenser cooling water inlet temperature.\n"
        "The condenser sets how much of the reactor effluent condenses into "
        "the separator. A change in cooling water supply temperature shifts "
        "the separator temperature and the vapour-liquid split, and the "
        "separator level and pressure loops respond. As on the reactor "
        "loop, neither the coolant inlet temperature nor the coolant flow "
        "is measured, so this is indistinguishable from a condenser cooling "
        "water valve that is not delivering its commanded flow."),
    "L_Afeed_loss": (
        ["XMEAS1", "XMV3", "XMEAS7", "XMEAS29", "XMEAS16"],
        "Loss of the A feed.\n"
        "The A feed flow collapses while its valve stays open or opens "
        "further, which is the clearest signature in the plant: the "
        "measured flow and the commanded valve position stop agreeing. "
        "Reactant A is then short in the reactor, conversion falls, and "
        "pressures move as the gas loop composition changes. This one is "
        "unambiguous; check the A feed flow against the A feed valve."),
    "L_Cheader": (
        ["XMEAS4", "XMV4", "XMEAS16", "XMEAS25"],
        "C header pressure loss.\n"
        "The supply header behind the A and C feed cannot deliver, so the "
        "stream flows less than the valve position calls for. As with a "
        "lost A feed, the tell is the disagreement between the measured "
        "flow and the commanded valve. It differs from a stuck valve only "
        "in where the restriction sits, and the plant has no instrument "
        "upstream of the valve to tell those apart."),
    "L_kinetics": (
        ["XMEAS37", "XMEAS38", "XMEAS40", "XMEAS41", "XMEAS9", "XMEAS7",
         "XMEAS23"],
        "Reaction kinetics drift.\n"
        "If the rate constants drift, the same reactor conditions convert "
        "reactants at a different rate. Product composition moves, "
        "unconverted reactant builds up in the recycle loop, and the "
        "pressure and purge loops work harder. The signature is a mismatch "
        "between what the reactor is fed and what comes out as product, "
        "rather than any single instrument going out of range."),
    "L_reac_cw_valve": (
        ["XMV10", "XMEAS21", "XMEAS9"],
        "Reactor cooling water valve not delivering its commanded flow.\n"
        "A sticking valve makes the controller hunt: the commanded position "
        "moves restlessly while the delivered flow lags or jumps. The "
        "reactor temperature is held, so look at the valve command's "
        "restlessness rather than at the temperature. Because the cooling "
        "water flow is not measured, this looks the same as a change in "
        "cooling water supply temperature; maintenance history is usually "
        "what separates them."),
    "L_cond_cw_valve": (
        ["XMV11", "XMEAS22", "XMEAS11"],
        "Condenser cooling water valve not delivering its commanded flow.\n"
        "As on the reactor loop: the commanded position and the delivered "
        "flow part company, the separator temperature loop compensates, and "
        "with no flow meter and no inlet thermometer this is "
        "indistinguishable from a coolant supply temperature change."),
    "L_steam_valve": (
        ["XMEAS19", "XMV9", "XMEAS18", "XMEAS17"],
        "Stripper steam valve and reboiler heat transfer.\n"
        "Steam flow to the stripper reboiler is measured, so a mismatch "
        "between the steam valve command and the delivered duty is directly "
        "visible. The stripper temperature loop compensates and the product "
        "flow follows. Check the steam flow against the steam valve "
        "position and the stripper temperature."),
    "L_other_valves": (
        ["XMV5", "XMV7", "XMV8", "XMV9", "XMEAS5", "XMEAS14", "XMEAS17",
         "XMEAS19"],
        "Sticking on the compressor recycle, separator underflow, stripper "
        "product or stripper steam valves.\n"
        "Several loops hunt at once because more than one valve fails to "
        "follow its command. The flows those valves set stop tracking their "
        "commanded positions, and the affected controllers become "
        "restless. Compare each of those flows with its own valve command."),
    "NF": (
        [],
        "Normal operation.\n"
        "Every channel sits within its usual range and the controllers are "
        "quiet. A handful of channels drifting by a standard deviation or "
        "two, with no flow disagreeing with its valve and no analysis out "
        "of range, is ordinary variation rather than a fault."),
}


def retrieve(z, ratio, k=4, channels=None):
    """Return the k manual entries whose channels look most disturbed.

    Retrieval is deliberately naive, the way a keyword-matched manual
    lookup in a real plant is: it scores an entry by how strongly the
    channels the entry names deviate, in mean or in spread.
    """
    from src.agents import catalogue as CAT
    channels = channels or CAT.CHANNELS
    idx = {c: i for i, c in enumerate(channels)}
    dev = np.maximum(np.abs(z), np.abs(np.log2(np.maximum(ratio, 1e-6))) * 2)
    scored = []
    for loc, (chs, text) in ENTRIES.items():
        if not chs:
            scored.append((0.0, loc, text))
            continue
        v = [dev[idx[c]] for c in chs if c in idx]
        scored.append((float(np.mean(sorted(v)[-3:])) if v else 0.0, loc,
                       text))
    scored.sort(key=lambda t: -t[0])
    return [t[2] for t in scored[:k]]
