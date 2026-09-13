"""
Residual generators for the six-axis robot.

The Tennessee Eastman residuals could be written down exactly because the
plant's own source code was available.  Here the physical parameters -- the
torque constants, gear ratios, friction coefficients, link inertias -- are
not published, so the relations are FITTED, on nominal recordings only.

What keeps that from becoming an ordinary black-box detector is the
restriction: each predictor may read only the variables that appear in the
structural relation it stands for, and nothing else.  A fault that the
structure decouples from a relation cannot change the value of any variable
that relation reads in a way that breaks it, so the residual stays quiet.
The fault signature therefore still follows from the structure rather than
from the data, exactly as with an analytically derived residual.  This is
the standard data-driven-residual-on-a-structural-support construction.

Three relation families:

  r_mdyn_i   the motor-side drivetrain of axis i.  Eliminating the
      electromagnetic torque and the friction torque between the three
      per-axis equations leaves one relation among the q-axis current, the
      motor velocity and acceleration, and the joint torque.  It carries
      both the friction and the commutation fault of that axis, which is
      why those two are indiscernible.

  r_rbd_i    row i of the rigid-body coupling.  It carries the added-mass
      faults of every link from i outward, the external contact torque on
      axis i, and the tool payload.

  r_gear_i, r_elec_i, r_tsens_i, r_bus
      relations that carry no fault in the model: the gear ratio between
      motor and link encoders, the electrical equation of the winding, the
      agreement of the two joint-torque sensors, and the bus current
      against the sum of the axis currents.  They are kept because a
      relation that must stay quiet under every fault is the cheapest check
      there is that the fitted relations have not simply learned the
      trajectory.
"""
from __future__ import annotations

import numpy as np

AX = list(range(1, 7))
FS = 100.0                       # sampling rate, Hz

RESIDUAL_NAMES = (["r_mdyn_%d" % i for i in AX]
                  + ["r_rbd_%d" % i for i in AX]
                  + ["r_gear_%d" % i for i in AX]
                  + ["r_elec_%d" % i for i in AX]
                  + ["r_tsens_%d" % i for i in AX]
                  + ["r_bus"])

# residual -> the fault modes it carries, read off the structural model
FAULT_SUPPORT = {}
for _i in AX:
    FAULT_SUPPORT["r_mdyn_%d" % _i] = ["friction%d" % _i,
                                       "commutation%d" % _i]
    # friction_i is here as well as on the motor side: the joint torque
    # sensor reads the torque delivered to the link, so friction downstream
    # of it enters this balance.  See src/robot/diagnose_isolation.py for the
    # measurements that corrected this placement.
    FAULT_SUPPORT["r_rbd_%d" % _i] = (["addedmass%d" % k for k in AX
                                       if k >= _i]
                                      + ["extforce%d" % _i, "payload",
                                         "friction%d" % _i])
    FAULT_SUPPORT["r_gear_%d" % _i] = []
    FAULT_SUPPORT["r_elec_%d" % _i] = []
    FAULT_SUPPORT["r_tsens_%d" % _i] = []
FAULT_SUPPORT["r_bus"] = []


class Channels:
    """Column lookup for one recording array."""

    def __init__(self, columns):
        self.idx = {c: i for i, c in enumerate(columns)}

    def __call__(self, name, X):
        return X[:, self.idx[name]]


def _deriv(x, fs=FS, half=5):
    """Central least-squares slope over a short window, per second."""
    n = len(x)
    t = np.arange(-half, half + 1) / fs
    denom = (t ** 2).sum()
    pad = np.concatenate([np.repeat(x[:1], half), x, np.repeat(x[-1:], half)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    return ((win - win.mean(1, keepdims=True)) * t).sum(1) / denom


def features(X, ch):
    """Everything both relation families read, for one recording."""
    d = {}
    for i in AX:
        d["wm%d" % i] = ch("motor_velocity_%d" % i, X)
        d["thm%d" % i] = ch("motor_position_%d" % i, X)
        d["wl%d" % i] = ch("joint_velocity_%d" % i, X)
        d["thl%d" % i] = ch("joint_position_%d" % i, X)
        d["I%d" % i] = ch("motor_iq_%d" % i, X)
        d["v%d" % i] = ch("motor_voltage_%d" % i, X)
        a = ch("torque_sensor_a_%d" % i, X)
        b = ch("torque_sensor_b_%d" % i, X)
        d["tja%d" % i] = a
        d["tjb%d" % i] = b
        d["tj%d" % i] = 0.5 * (a + b)
        d["dwm%d" % i] = _deriv(d["wm%d" % i])
        d["dwl%d" % i] = _deriv(d["wl%d" % i])
        d["dI%d" % i] = _deriv(d["I%d" % i])
    d["Itot"] = ch("robot_current", X)
    return d


def raw_inputs(kind, i, d):
    """The variables a relation is structurally allowed to read.

    This is the only place the structural restriction is enforced, and it is
    what makes the fitted relation a residual generator rather than a
    detector: r_mdyn_i sees the current, velocity and acceleration of its
    own axis and nothing else; r_rbd_i sees the link coordinates of every
    axis, because the mass matrix of a serial arm is dense, but no
    motor-side quantity at all.
    """
    if kind == "mdyn":
        return np.column_stack([d["I%d" % i], d["wm%d" % i],
                                d["dwm%d" % i]])
    if kind == "rbd":
        cols = []
        for k in AX:
            cols += [np.sin(d["thl%d" % k]), np.cos(d["thl%d" % k]),
                     d["wl%d" % k], d["dwl%d" % k]]
        return np.column_stack(cols)
    if kind == "gear":
        return np.column_stack([d["wl%d" % i]])
    if kind == "elec":
        return np.column_stack([d["I%d" % i], d["dI%d" % i],
                                d["wm%d" % i]])
    if kind == "tsens":
        return np.column_stack([d["tjb%d" % i]])
    if kind == "bus":
        return np.column_stack([d["I%d" % k] for k in AX])
    raise ValueError(kind)


# Random Fourier feature maps, one per relation.  The relations are
# genuinely nonlinear -- gear efficiency depends on the direction of power
# flow, and the rigid-body torque is a trigonometric polynomial of the joint
# angles whose coefficients need the arm's kinematics, which are not
# published.  Rather than guess the kinematics, each relation is fitted in a
# random Fourier basis over its own admissible inputs.  The basis is drawn
# once from a fixed seed and published with the code, so the relation is
# reproducible; the structural restriction is untouched, because the basis
# only ever sees the admissible variables.
_RFF = {}
NFEAT = {"mdyn": 400, "rbd": 1200, "gear": 60, "elec": 200, "tsens": 60,
         "bus": 200}


def _rff(kind, i, Z, fit_stats=None):
    key = (kind, i)
    if key not in _RFF:
        rng = np.random.default_rng(hash(key) % (2 ** 31))
        m = NFEAT[kind]
        _RFF[key] = {"W": rng.standard_normal((Z.shape[1], m)),
                     "b": rng.uniform(0, 2 * np.pi, m), "mu": None,
                     "sd": None}
    st = _RFF[key]
    if fit_stats is not None:
        st["mu"], st["sd"] = fit_stats
    if st["mu"] is None:
        st["mu"] = Z.mean(0)
        st["sd"] = np.maximum(Z.std(0), 1e-6)
    Zs = (Z - st["mu"]) / st["sd"]
    P = np.sqrt(2.0 / st["W"].shape[1]) * np.cos(Zs @ st["W"] + st["b"])
    return np.column_stack([np.ones(len(Z)), Zs, P])


def design(kind, i, d):
    return _rff(kind, i, raw_inputs(kind, i, d))


def target(kind, i, d):
    return {"mdyn": lambda: d["tj%d" % i],
            "rbd": lambda: d["tj%d" % i],
            "gear": lambda: d["wm%d" % i],
            "elec": lambda: d["v%d" % i],
            "tsens": lambda: d["tja%d" % i],
            "bus": lambda: d["Itot"]}[kind]()


SPECS = ([("mdyn", i) for i in AX] + [("rbd", i) for i in AX]
         + [("gear", i) for i in AX] + [("elec", i) for i in AX]
         + [("tsens", i) for i in AX] + [("bus", 0)])


class RobotResiduals:
    """Ridge predictors, one per structural relation, fitted on nominal data."""

    def __init__(self, columns, lam=1e-3):
        self.ch = Channels(columns)
        self.lam = lam
        self.coef = {}

    def fit(self, recordings, subsample=4):
        # standardise each relation's inputs on the fitting block, once
        for kind, i in SPECS:
            zs = np.concatenate([raw_inputs(kind, i,
                                            features(X, self.ch))[::16]
                                 for X in recordings[:60]])
            _rff(kind, i, zs[:1], fit_stats=(zs.mean(0),
                                             np.maximum(zs.std(0), 1e-6)))
        for kind, i in SPECS:
            XtX = None
            Xty = None
            for X in recordings:
                d = features(X, self.ch)
                A = design(kind, i, d)[::subsample]
                y = target(kind, i, d)[::subsample]
                ok = np.isfinite(A).all(1) & np.isfinite(y)
                A, y = A[ok], y[ok]
                if XtX is None:
                    XtX = A.T @ A
                    Xty = A.T @ y
                else:
                    XtX += A.T @ A
                    Xty += A.T @ y
            n = XtX.shape[0]
            scale = np.trace(XtX) / n
            self.coef[(kind, i)] = np.linalg.solve(
                XtX + self.lam * scale * np.eye(n), Xty)
        return self

    def residuals(self, X):
        d = features(X, self.ch)
        out = {}
        for kind, i in SPECS:
            A = design(kind, i, d)
            y = target(kind, i, d)
            pred = A @ self.coef[(kind, i)]
            name = "r_bus" if kind == "bus" else "r_%s_%d" % (kind, i)
            out[name] = y - pred
        return out

    def window_features(self, X):
        """Mean and dispersion of every residual over the whole recording.

        A recording is one complete pick-and-place motion, so the window is
        the motion rather than a fixed number of samples: the arm passes
        through the same trajectory every time, which is what makes the
        nominal distribution comparable across recordings.
        """
        r = self.residuals(X)
        mu, disp = [], []
        for k in RESIDUAL_NAMES:
            v = r[k]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                mu.append(0.0)
                disp.append(0.0)
                continue
            m = np.median(v)
            mu.append(float(np.mean(v)))
            disp.append(float(np.median(np.abs(v - m))))
        return np.array(mu + disp)
