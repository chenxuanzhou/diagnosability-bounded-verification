"""
Structural diagnosability primitives shared by both case studies.

The central object is the *structural indiscernibility quotient*: the
partition of the fault-mode set induced by identical columns of the fault
signature matrix.  Two fault modes in the same block cannot be told apart
by ANY residual generator built from the given equation set and sensor
set, which is exactly the upper bound R1 asserts.

We obtain the partition from the structural isolability matrix rather than
by enumerating MSO sets, because the two agree exactly and the isolability
matrix is computable on models where MSO enumeration is intractable:

    im[i, j] = 1  <=>  fault j is a possible diagnosis when fault i acts
                       (i.e. j cannot be isolated from i)

Fault i and fault j have identical FSM columns  <=>  im[i, j] = im[j, i] = 1.
Mutual non-isolability is an equivalence relation, so its blocks are the
quotient set.
"""
from __future__ import annotations

import itertools

import numpy as np


def isolability_matrix(model, causality="mixed"):
    """Structural isolability matrix, rows/cols ordered as model.f."""
    kw = {} if causality == "mixed" else {"causality": causality}
    return np.asarray(model.IsolabilityAnalysis(permute=False, **kw))


def equivalence_classes(im, names):
    """Blocks of the mutual-non-isolability relation, i.e. the quotient set.

    Returns a list of lists of fault names, each block sorted by the order
    in `names`, blocks sorted by their first member.
    """
    n = len(names)
    mutual = (im == 1) & (im.T == 1)
    seen, blocks = set(), []
    for i in range(n):
        if i in seen:
            continue
        block = [j for j in range(n) if mutual[i, j]]
        # guard: mutual non-isolability must be transitive on a consistent
        # isolability matrix; verify rather than assume.
        for a, b in itertools.combinations(block, 2):
            if not mutual[a, b]:
                raise AssertionError(
                    "non-transitive indiscernibility at %s/%s" %
                    (names[a], names[b]))
        seen.update(block)
        blocks.append([names[j] for j in block])
    return blocks


def class_of(blocks):
    """fault name -> index of its block."""
    return {f: k for k, blk in enumerate(blocks) for f in blk}


def detectable(model):
    """Names of structurally detectable faults."""
    det, _ = model.DetectabilityAnalysis()
    return list(det)


def quotient_summary(model, names, causality="mixed"):
    """Everything E1 needs for one (model, sensor-set) pair."""
    im = isolability_matrix(model, causality)
    blocks = equivalence_classes(im, names)
    det = set(detectable(model))
    sizes = [len(b) for b in blocks]
    return {
        "isolability": im,
        "blocks": blocks,
        "n_classes": len(blocks),
        "max_class_size": max(sizes) if sizes else 0,
        "n_singletons": sum(1 for s in sizes if s == 1),
        "detectable": [f for f in names if f in det],
        "undetectable": [f for f in names if f not in det],
        # fraction of fault modes that sit inside a non-singleton block:
        # the population the verifier is provably blind inside.
        "indiscernible_mass": sum(s for s in sizes if s > 1) / float(len(names)),
    }


def render_blocks(blocks, strip="f"):
    out = []
    for b in blocks:
        out.append("{" + ", ".join(x[len(strip):] if x.startswith(strip) else x
                                   for x in b) + "}")
    return "  ".join(out)
