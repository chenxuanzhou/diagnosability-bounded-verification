"""Count MTES / MSO sets for both structural models (Table I column)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.robot.robot_structural import build_model as rb
from src.tep.tep_structural import build_model as tb

out = {}
for nm, bm in (("robot", rb), ("tep", tb)):
    m = bm()
    rec = {"ne": m.ne(), "nx": m.nx(), "nz": m.nz(), "nf": m.nf(),
           "redundancy": int(m.Redundancy())}
    for meth in ("MTES", "MSO"):
        t0 = time.time()
        try:
            sets = getattr(m, meth)()
            rec[meth.lower() + "_count"] = len(sets)
            rec[meth.lower() + "_seconds"] = round(time.time() - t0, 1)
        except Exception as e:
            rec[meth.lower() + "_count"] = None
            rec[meth.lower() + "_error"] = repr(e)[:200]
        print(nm, meth, rec.get(meth.lower() + "_count"), flush=True)
    out[nm] = rec
os.makedirs("results/E1", exist_ok=True)
json.dump(out, open("results/E1/set_counts.json", "w"), indent=2)
print("done", flush=True)
