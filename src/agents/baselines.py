"""
The baseline ladder, B0 through B7.

B0  bare LLM, zero shot, over a textual summary of the window
B1  LLM with retrieval over a fault manual
B2  LLM with self-consistency, k independent samples and a majority vote
B3  LLM with numerical tool calls over the raw window
B4  LLM handed the output of a learned anomaly detector
B5  LLM plus the structural admissibility verification layer  (this paper)
B6  residual verifier alone, no LLM
B7  oracle: the verifier's candidate set, scored as if an ideal chooser
    picked correctly inside it

B2, B3 and B4 exist to answer the three reviewer objections that a stronger
prompt, more samples, or a detector bolted on would have solved the problem.
B6 answers whether the LLM is needed at all.  B7 is the ceiling the bound
allows.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.agents import catalogue as CAT                     # noqa: E402
from src.agents import llm as LLM                           # noqa: E402
from src.agents import manual as MAN                        # noqa: E402
from src.eval import propositions as P                      # noqa: E402

ANSWER_RULES = (
    'Answer with one JSON object and nothing else:\n'
    '{"fault": "<one catalogue id>", "confidence": <number between 0 and 1>,'
    ' "reason": "<one sentence>"}\n'
    'The id must be copied exactly from the catalogue. Do not invent ids, '
    'do not answer with a category, and do not return more than one id.'
)

SYSTEM = (
    "You are a fault diagnosis assistant for the Tennessee Eastman chemical "
    "process: a reactor, a condenser, a vapour-liquid separator, a "
    "recycle compressor and a stripper, under plantwide feedback control.\n\n"
    "You will be shown a three-hour window of plant data, summarised per "
    "channel as the window mean, its deviation from normal operation in "
    "units of the normal standard deviation, and the ratio of the window's "
    "spread to the normal spread. Because the plant is under feedback "
    "control, a deviation often shows up in a manipulated variable rather "
    "than in the measurement being controlled.\n\n"
    "Decide which single entry of the fault catalogue best explains the "
    "window.\n\nFault catalogue:\n" + P.catalogue_text() + "\n\n"
    + ANSWER_RULES
)

VALID = set(P.PROPOSITIONS)


def parse_answer(text):
    """Pull the committed id out of a model reply, tolerantly."""
    if not text:
        return None, {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            f = str(obj.get("fault", "")).strip()
            if f in VALID:
                return f, obj
        except Exception:                                   # noqa: BLE001
            pass
    # fall back to the first catalogue id that appears verbatim
    for tok in sorted(VALID, key=len, reverse=True):
        if re.search(r"\b%s\b" % re.escape(tok), text):
            return tok, {}
    return None, {}


def _chat(model, messages, seed, tag, max_tokens=900):
    kw = {}
    if LLM.MODELS[model][0] == "ollama":
        kw = {"temperature": 0.0, "seed": seed}
    return LLM.chat(model, messages, max_tokens=max_tokens, log_tag=tag, **kw)


# --------------------------------------------------------------- B0
def b0_bare(case, pres, model, seed):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Plant data window:\n"
                                        + pres.text(case["window"])}]
    txt, meta = _chat(model, msgs, seed, "B0/%s" % case["id"])
    ans, obj = parse_answer(txt)
    return {"answer": ans, "raw": txt, "obj": obj, "meta": meta}


# --------------------------------------------------------------- B1
def b1_rag(case, pres, model, seed, k=4):
    _, z, ratio = pres.stats(case["window"])
    entries = MAN.retrieve(z, ratio, k=k)
    ctx = "\n\n".join(entries)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": ("Relevant entries retrieved from the plant fault "
                         "manual:\n\n" + ctx
                         + "\n\nPlant data window:\n"
                         + pres.text(case["window"]))}]
    txt, meta = _chat(model, msgs, seed, "B1/%s" % case["id"])
    ans, obj = parse_answer(txt)
    return {"answer": ans, "raw": txt, "obj": obj, "meta": meta,
            "retrieved": entries}


# --------------------------------------------------------------- B2
def b2_self_consistency(case, pres, model, seed, k=3):
    votes, raws, metas = [], [], []
    for i in range(k):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Plant data window:\n"
                                            + pres.text(case["window"])}]
        kw = {}
        if LLM.MODELS[model][0] == "ollama":
            kw = {"temperature": 0.8, "seed": seed * 1000 + i}
        txt, meta = LLM.chat(model, msgs, max_tokens=900,
                             log_tag="B2/%s/%d" % (case["id"], i), **kw)
        a, _ = parse_answer(txt)
        votes.append(a)
        raws.append(txt)
        metas.append(meta)
    good = [v for v in votes if v]
    if good:
        vals, cnt = np.unique(good, return_counts=True)
        ans = str(vals[int(np.argmax(cnt))])
        agree = float(cnt.max() / len(votes))
    else:
        ans, agree = None, 0.0
    return {"answer": ans, "raw": raws, "obj": {"votes": votes,
                                                "agreement": agree},
            "meta": metas[-1]}


# --------------------------------------------------------------- B3
TOOL_SPEC = (
    "You may call numerical tools on the raw window before answering. "
    "To call tools, reply with one JSON object:\n"
    '{"tool_calls": [{"name": "<tool>", "args": {...}}, ...]}\n'
    "Available tools:\n"
    "  channel_series(name, every) -> the sampled values of one channel\n"
    "  correlate(a, b) -> Pearson correlation between two channels over "
    "the window\n"
    "  trend(name) -> least-squares slope of one channel per hour, and its "
    "standard error\n"
    "  ratio(a, b) -> mean of channel a divided by mean of channel b\n"
    "You may call tools at most twice, then you must answer.\n"
)


def _tool_exec(W, name, args):
    idx = {c: i for i, c in enumerate(CAT.CHANNELS)}
    try:
        if name == "channel_series":
            i = idx[args["name"]]
            every = int(args.get("every", 6))
            v = W[::every, i]
            return [round(float(x), 4) for x in v[:20]]
        if name == "correlate":
            a, b = idx[args["a"]], idx[args["b"]]
            x, y = W[:, a], W[:, b]
            if x.std() < 1e-12 or y.std() < 1e-12:
                return "undefined, a channel is constant"
            return round(float(np.corrcoef(x, y)[0, 1]), 4)
        if name == "trend":
            i = idx[args["name"]]
            t = np.arange(len(W)) * 0.05
            A = np.vstack([t, np.ones_like(t)]).T
            coef, res, *_ = np.linalg.lstsq(A, W[:, i], rcond=None)
            se = float(np.sqrt(res[0] / max(len(W) - 2, 1)) /
                       (t.std() * np.sqrt(len(W)))) if len(res) else 0.0
            return {"slope_per_hour": round(float(coef[0]), 5),
                    "std_error": round(se, 5)}
        if name == "ratio":
            a, b = idx[args["a"]], idx[args["b"]]
            d = float(W[:, b].mean())
            return round(float(W[:, a].mean() / d), 5) if abs(d) > 1e-12 \
                else "undefined"
    except Exception as e:                                  # noqa: BLE001
        return "error: %s" % type(e).__name__
    return "unknown tool"


def b3_tools(case, pres, model, seed, max_rounds=2):
    W = case["window"]
    msgs = [{"role": "system", "content": SYSTEM + "\n\n" + TOOL_SPEC},
            {"role": "user", "content": "Plant data window:\n"
                                        + pres.text(W)}]
    transcript = []
    for rnd in range(max_rounds + 1):
        txt, meta = _chat(model, msgs, seed, "B3/%s/%d" % (case["id"], rnd))
        transcript.append(txt)
        ans, obj = parse_answer(txt)
        if ans:
            return {"answer": ans, "raw": transcript, "obj": obj,
                    "meta": meta, "rounds": rnd}
        calls = []
        m = re.search(r"\{.*\}", txt or "", re.S)
        if m:
            try:
                calls = json.loads(m.group(0)).get("tool_calls", [])[:4]
            except Exception:                               # noqa: BLE001
                calls = []
        if not calls or rnd == max_rounds:
            return {"answer": ans, "raw": transcript, "obj": obj,
                    "meta": meta, "rounds": rnd}
        out = {("%s(%s)" % (c.get("name"), json.dumps(c.get("args", {})))):
               _tool_exec(W, c.get("name"), c.get("args", {}))
               for c in calls}
        msgs.append({"role": "assistant", "content": txt})
        msgs.append({"role": "user",
                     "content": "Tool results:\n"
                                + json.dumps(out, default=str)
                                + "\n\nNow answer. " + ANSWER_RULES})
    return {"answer": None, "raw": transcript, "obj": {}, "meta": meta}


# --------------------------------------------------------------- B4
def b4_detector(case, pres, model, seed, detector):
    rep = detector.report(case["window"])
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": ("A learned anomaly detector (principal component "
                         "model fitted on normal operation only) reports:\n"
                         + rep + "\n\nPlant data window:\n"
                         + pres.text(case["window"]))}]
    txt, meta = _chat(model, msgs, seed, "B4/%s" % case["id"])
    ans, obj = parse_answer(txt)
    return {"answer": ans, "raw": txt, "obj": obj, "meta": meta,
            "detector": rep}


# --------------------------------------------------------------- B5
def b5_verified(case, pres, model, seed, verifier, trig, retry=True):
    """LLM proposal, then structural admissibility, then one guided retry.

    The verifier never names the answer.  On a rejection it returns only the
    admissible set, and the model has to choose within it; on an ambiguous
    set the case is escalated rather than guessed.
    """
    first = b0_bare(case, pres, model, seed)
    claim = first["answer"]
    D, rule = verifier.candidates(trig)
    if claim is None:
        return {"answer": None, "decision": "no_answer", "candidates": D,
                "rule": rule, "rounds": 1, "first": first,
                "meta": first["meta"], "raw": first["raw"]}
    decision, D, rule = verifier.verify(claim, trig)

    if decision == "reject" and retry:
        adm = "\n".join("%s: %s" % (d, P.LOCATIONS[d][1]
                                    if d in P.LOCATIONS else "no fault")
                        for d in D)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Plant data window:\n"
                                            + pres.text(case["window"])},
                {"role": "assistant", "content": first["raw"] or ""},
                {"role": "user",
                 "content": ("A physical consistency check on the plant's "
                             "redundancy relations rules out that answer. "
                             "The only fault modes consistent with the "
                             "residual pattern are:\n" + adm
                             + "\n\nChoose one of these. " + ANSWER_RULES)}]
        txt, meta = _chat(model, msgs, seed, "B5retry/%s" % case["id"])
        second, _ = parse_answer(txt)
        if second is not None:
            claim = second
            decision, D, rule = verifier.verify(claim, trig)
        return {"answer": claim if decision != "escalate" else None,
                "decision": decision, "candidates": D, "rule": rule,
                "rounds": 2, "first": first, "meta": meta, "raw": txt}
    return {"answer": claim if decision != "escalate" else None,
            "decision": decision, "candidates": D, "rule": rule,
            "rounds": 1, "first": first, "meta": first["meta"],
            "raw": first["raw"]}


# --------------------------------------------------------------- B5L
def b5_verified_with_log(case, pres, model, seed, verifier, trig, log_text):
    """The full proposed pipeline: verifier, then the agent on what is left.

    B5 escalates whenever the admissible set has more than one member,
    which is what the verification operator is defined to do, and it is why
    B5's numbers do not move when the language model is changed: with only
    the sensor channels in play the verifier determines the outcome and the
    model is decorative.  That is the honest reading of B5 and it is the
    answer to "is the LLM necessary" as far as sensor data goes.

    This baseline is the pipeline the paper actually proposes.  When the
    admissible set is a singleton it behaves exactly like B5.  When it is
    not, the case is handed to the agent together with the operations log --
    information that never travelled through a sensor -- and the agent
    chooses inside the admissible set.  It can never choose outside it, so
    the bound is still respected; what it can do is resolve an ambiguity the
    verifier is provably unable to resolve.
    """
    first = b0_bare(case, pres, model, seed)
    claim = first["answer"]
    D, rule = verifier.candidates(trig)
    if len(D) == 1:
        decision = "accept" if claim == D[0] else "reject"
        return {"answer": D[0], "decision": decision, "candidates": D,
                "rule": rule, "rounds": 1, "first": first,
                "meta": first["meta"], "raw": first["raw"], "used_log": False}

    nl = chr(10)
    adm = nl.join("%s: %s" % (d, P.LOCATIONS[d][1] if d in P.LOCATIONS
                              else "the plant is operating normally")
                  for d in D)
    system = (
        "You are supporting an operator on the Tennessee Eastman plant."
        + nl + nl +
        "A physical consistency check on the plant's redundancy relations "
        "has narrowed the cause to the candidates below and can go no "
        "further: on this instrumentation they produce the same residual "
        "pattern, so no further measurement from the installed sensors can "
        "separate them." + nl + nl
        + "Candidates:" + nl + adm + nl + nl
        + "Operations shift log for the last 24 hours:" + nl + log_text
        + nl + nl
        + "Choose the single most likely candidate. " + ANSWER_RULES)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Plant data window:" + nl
                                        + pres.text(case["window"])}]
    txt, meta = _chat(model, msgs, seed, "B5L/%s" % case["id"])
    ans, _ = parse_answer(txt)
    if ans not in D:
        ans = None
    return {"answer": ans, "decision": "resolved" if ans else "escalate",
            "candidates": D, "rule": rule, "rounds": 2, "first": first,
            "meta": meta, "raw": txt, "used_log": True}


# --------------------------------------------------------------- B6, B7
def b6_residual_only(case, verifier, trig, rng):
    """No LLM: the verifier's candidate set, with a uniform pick inside it."""
    D, rule = verifier.candidates(trig)
    pick = str(rng.choice(D)) if D else None
    return {"answer": pick, "candidates": D, "rule": rule}


def b7_oracle(case, verifier, trig):
    """Ceiling: an ideal chooser inside the admissible set."""
    D, rule = verifier.candidates(trig)
    truth = case["location"]
    return {"answer": truth if truth in D else (D[0] if D else None),
            "candidates": D, "rule": rule, "truth_in_set": truth in D}
