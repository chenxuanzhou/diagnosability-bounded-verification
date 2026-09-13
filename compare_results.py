"""Compare regenerated result files with the stored ones.

    python compare_results.py STORED_DIR REGENERATED_DIR [--out FILE]

For every JSON file present in both trees, every value is compared leaf by
leaf: numbers within a relative tolerance, everything else for equality.
Each file is reported as identical, as differing (with the number of
differing values and the largest absolute numeric difference), or as having
a different structure.  Latency results (E10) and wall-clock timings depend
on the hardware, so they are listed separately and are not counted as
mismatches.  The script only reports; it never fails the run.
"""
from __future__ import annotations

import argparse
import json
import math
import os

REL_TOL = 1e-6
HARDWARE_DEPENDENT = ("E10" + os.sep, "E10/")
TIMING_KEYS = ("wall_s", "seconds", "latency", "median_ms", "p95_ms",
               "median_s", "p95_s", "elapsed")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from leaves(x[k], "%s/%s" % (path, k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, "%s[%d]" % (path, i))
    else:
        yield path, x


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare(stored, regenerated):
    a, b = dict(leaves(stored)), dict(leaves(regenerated))
    only = len(set(a) ^ set(b))
    n = differ = timing = 0
    worst = 0.0
    for key in set(a) & set(b):
        va, vb = a[key], b[key]
        n += 1
        if any(t in key for t in TIMING_KEYS):
            timing += 1
            continue
        if is_number(va) and is_number(vb):
            if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
                continue
            d = abs(va - vb)
            if d > REL_TOL * max(1.0, abs(va), abs(vb)):
                differ += 1
                worst = max(worst, d)
        elif va != vb:
            differ += 1
    return n, differ, worst, only, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stored")
    ap.add_argument("regenerated")
    ap.add_argument("--out")
    ap.add_argument("--since",
                    help="marker file created before the run; files not modified after it were "
                         "only copied, not regenerated, and are listed but not compared")
    args = ap.parse_args()
    since = os.path.getmtime(args.since) if args.since else None

    lines, counts = [], {"identical": 0, "different": 0, "hardware": 0, "not_regenerated": 0}
    for root, _, files in os.walk(args.regenerated):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            regen = os.path.join(root, name)
            rel = os.path.relpath(regen, args.regenerated)
            stored = os.path.join(args.stored, rel)
            if since is not None and os.path.getmtime(regen) <= since:
                counts["not_regenerated"] += 1
                lines.append("%-48s not regenerated in this run" % rel)
                continue
            if not os.path.exists(stored):
                lines.append("%-48s new file, no stored counterpart" % rel)
                continue
            with open(stored, encoding="utf-8") as fa, open(regen, encoding="utf-8") as fb:
                n, differ, worst, only, timing = compare(json.load(fa), json.load(fb))
            if rel.startswith(HARDWARE_DEPENDENT):
                counts["hardware"] += 1
                lines.append("%-48s hardware-dependent timings, not compared" % rel)
            elif differ == 0 and only == 0:
                counts["identical"] += 1
                note = " (%d timing values ignored)" % timing if timing else ""
                lines.append("%-48s identical, %d values%s" % (rel, n, note))
            else:
                counts["different"] += 1
                lines.append("%-48s DIFFERENT: %d of %d values, max abs diff %.3g, %d keys only on one side"
                             % (rel, differ, n, worst, only))
    summary = ("compared with stored results (relative tolerance %g): %d identical, %d different, "
               "%d hardware-dependent, %d not regenerated in this run"
               % (REL_TOL, counts["identical"], counts["different"], counts["hardware"],
                  counts["not_regenerated"]))
    text = "\n".join(sorted(lines) + ["", summary])
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
