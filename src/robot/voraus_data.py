"""
Loading the voraus-AD recordings, restricted to the independently sensed
channels the structural model uses.

The dataset ships 137 columns at 100 Hz.  Most of the per-axis channels are
derived by the drive firmware from others (motor torque is Kt times the
q-axis current, the power channels are products, the computed torque and
inertia are the controller's own model output), so loading them would only
manufacture redundancy the hardware does not have.  What is loaded is the
set declared in robot_structural.SENSORS plus the labels.

Each recording is one pick-and-place motion.  Recordings are grouped by the
anomaly flag: the normal ones are split into disjoint fit, calibration and
test blocks by recording index, so no recording contributes to more than one
role.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.robot import robot_structural as RS                # noqa: E402

RAW = os.path.join("data", "raw", "voraus", "voraus-ad-100hz.parquet")
OUT = os.path.join("data", "voraus")
os.makedirs(OUT, exist_ok=True)

LABEL_COLS = ["time", "sample", "anomaly", "category", "setting", "action",
              "active"]
SIGNAL_COLS = []
for _i in RS.AX:
    SIGNAL_COLS += ["motor_position_%d" % _i, "motor_velocity_%d" % _i,
                    "joint_position_%d" % _i, "joint_velocity_%d" % _i,
                    "motor_iq_%d" % _i, "motor_voltage_%d" % _i,
                    "torque_sensor_a_%d" % _i, "torque_sensor_b_%d" % _i]
SIGNAL_COLS += ["robot_current"]

# Category ids, from voraus_ad.Category in the official repository
CATEGORY_NAMES = {
    0: "AXIS_FRICTION", 1: "AXIS_WEIGHT", 2: "COLLISION_FOAM",
    3: "COLLISION_CABLE", 4: "COLLISION_CARTON", 5: "MISS_CAN",
    6: "LOSE_CAN", 7: "CAN_WEIGHT", 8: "ENTANGLED", 9: "INVALID_POSITION",
    10: "MOTOR_COMMUTATION", 11: "WOBBLING_STATION", 12: "NORMAL_OPERATION",
}


def build(max_samples=None):
    import pyarrow.parquet as pq
    cols = LABEL_COLS + SIGNAL_COLS
    tbl = pq.read_table(RAW, columns=cols)
    df = tbl.to_pandas()
    del tbl
    print("loaded %s rows, %d columns" % (len(df), df.shape[1]))

    samples, meta = [], []
    for sid, g in df.groupby("sample", sort=True):
        g = g.sort_values("time")
        samples.append(g[SIGNAL_COLS].to_numpy(dtype=np.float32))
        meta.append({"sample": int(sid),
                     "anomaly": bool(g["anomaly"].iloc[0]),
                     "category": int(g["category"].iloc[0]),
                     "category_name": CATEGORY_NAMES.get(
                         int(g["category"].iloc[0]), "?"),
                     "setting": int(g["setting"].iloc[0]),
                     "n": int(len(g))})
        if max_samples and len(samples) >= max_samples:
            break
    print("recordings: %d ; anomalous %d ; normal %d"
          % (len(meta), sum(m["anomaly"] for m in meta),
             sum(not m["anomaly"] for m in meta)))
    lens = [m["n"] for m in meta]
    print("length: min %d median %d max %d"
          % (min(lens), int(np.median(lens)), max(lens)))
    from collections import Counter
    print("categories: %s"
          % dict(Counter(m["category_name"] for m in meta)))

    # store as a ragged list in one npz, plus the metadata
    np.savez_compressed(os.path.join(OUT, "recordings.npz"),
                        **{"r%05d" % i: a for i, a in enumerate(samples)})
    json.dump({"columns": SIGNAL_COLS, "meta": meta},
              open(os.path.join(OUT, "recordings_meta.json"), "w"))
    print("wrote %s" % os.path.join(OUT, "recordings.npz"))


def load():
    z = np.load(os.path.join(OUT, "recordings.npz"))
    info = json.load(open(os.path.join(OUT, "recordings_meta.json")))
    arrs = [z["r%05d" % i] for i in range(len(info["meta"]))]
    return arrs, info["meta"], info["columns"]


if __name__ == "__main__":
    build()
