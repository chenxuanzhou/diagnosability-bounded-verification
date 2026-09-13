"""
LLM client used by every agent baseline.

Two providers:
  moonshot  Kimi models over the OpenAI-compatible endpoint (closed, hosted)
  ollama    locally served open-weight models

Protocol notes that go into the paper.

1.  The Kimi models reject any temperature other than 1, so the plan's
    "temperature = 0 as the main setting" is not available for the closed
    family.  Rather than run the two families under different sampling
    regimes, every model is evaluated the same way: N independent seeds per
    case, reporting mean and confidence interval.  The self-consistency
    baseline draws k samples within one seed, which is a different axis.
2.  Both Kimi models emit reasoning tokens before the answer.  Those are
    billed as completion tokens, so max_tokens has to leave room for them
    or the answer comes back empty.
3.  Every call is appended to a JSONL log: model id, call date, prompt,
    raw response, token usage and latency.  The plan requires the raw
    outputs to be published, and latency feeds the timing table.

The API key is read from key.txt in the project root.  It is never logged,
never printed, and never sent anywhere except the Moonshot endpoint.
"""
from __future__ import annotations

import datetime
import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(ROOT, "logs", "llm")
os.makedirs(LOGDIR, exist_ok=True)

MOONSHOT_BASE = "https://api.moonshot.cn/v1"
OLLAMA_BASE = "http://127.0.0.1:11434"

# model registry: alias -> (provider, model id, family, capability tier,
#                            extra request fields)
# The two Kimi models are reasoning models.  Leaving the reasoning pass on
# costs thousands of completion tokens per call, so it is treated as an
# explicit capability tier rather than as a hidden default: the -think
# aliases reason, the plain aliases answer directly.
_NOTHINK = {"thinking": {"type": "disabled"}}
MODELS = {
    "kimi-k3-think": ("moonshot", "kimi-k3", "Kimi", "frontier-reasoning", {}),
    "kimi-k3":       ("moonshot", "kimi-k3", "Kimi", "frontier-direct", _NOTHINK),
    "kimi-k2.6-think": ("moonshot", "kimi-k2.6", "Kimi", "prev-reasoning", {}),
    "kimi-k2.6":     ("moonshot", "kimi-k2.6", "Kimi", "prev-direct", _NOTHINK),
    # qwen3.5 is a reasoning model too: without think=false it spends the
    # whole prediction budget on reasoning and returns empty content, the
    # same failure the Kimi models have.
    "qwen3.5-9b":    ("ollama", "qwen3.5:9b", "Qwen", "open-9b",
                      {"think": False}),
    "qwen2.5-7b":    ("ollama", "qwen2.5:7b", "Qwen", "open-7b", {}),
    "llama3.1-8b":   ("ollama", "llama3.1:8b", "Llama", "open-8b", {}),
}

# Moonshot list prices, CNY per million tokens.  Used only for budget
# tracking; the authoritative spend figure is the account balance, which
# budget_probe.py reads directly.
# Calibrated against the account balance over the first 1200 calls of the
# main grid rather than taken from a price list, so the projection matches
# what is actually billed.  The authoritative spend figure is always the
# balance, which every run records before and after.
PRICE_CNY = {
    "kimi-k3": {"in": 12.0, "in_cached": 3.0, "out": 48.0},
    "kimi-k3-think": {"in": 12.0, "in_cached": 3.0, "out": 48.0},
    "kimi-k2.6": {"in": 8.0, "in_cached": 2.0, "out": 32.0},
    "kimi-k2.6-think": {"in": 8.0, "in_cached": 2.0, "out": 32.0},
}

_KEY = None


def _key():
    global _KEY
    if _KEY is None:
        p = os.path.join(ROOT, "key.txt")
        with open(p) as fh:
            _KEY = fh.read().strip()
    return _KEY


class Usage:
    """Running token and cost accounting for one process."""

    def __init__(self):
        self.calls = 0
        self.prompt = 0
        self.cached = 0
        self.completion = 0
        self.reasoning = 0
        self.seconds = 0.0
        self.by_model = {}

    def add(self, model, u, dt):
        self.calls += 1
        p = int(u.get("prompt_tokens", 0) or 0)
        c = int(u.get("completion_tokens", 0) or 0)
        ca = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        rr = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
        self.prompt += p
        self.cached += ca
        self.completion += c
        self.reasoning += rr
        self.seconds += dt
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "cached": 0,
                                             "out": 0, "reason": 0, "sec": 0.0})
        m["calls"] += 1
        m["in"] += p
        m["cached"] += ca
        m["out"] += c
        m["reason"] += rr
        m["sec"] += dt

    def cost_cny(self):
        total = 0.0
        for model, m in self.by_model.items():
            pr = PRICE_CNY.get(model)
            if not pr:
                continue
            fresh = max(m["in"] - m["cached"], 0)
            total += (fresh * pr["in"] + m["cached"] * pr["in_cached"]
                      + m["out"] * pr["out"]) / 1e6
        return total

    def summary(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt,
                "cached_tokens": self.cached,
                "completion_tokens": self.completion,
                "reasoning_tokens": self.reasoning,
                "seconds": round(self.seconds, 1),
                "est_cost_cny": round(self.cost_cny(), 4),
                "by_model": self.by_model}


USAGE = Usage()


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def balance_cny():
    """Remaining Moonshot account balance in CNY, or None on failure."""
    try:
        req = urllib.request.Request(
            MOONSHOT_BASE + "/users/me/balance",
            headers={"Authorization": "Bearer " + _key()})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["data"]["available_balance"]
    except Exception:
        return None


def chat(alias, messages, max_tokens=2048, temperature=None, seed=None,
         log_tag="", timeout=300, retries=8):
    """One chat completion.

    Returns (text, meta).  meta carries usage, latency, model id and the
    call date; the raw response is written to the JSONL log.
    """
    if alias not in MODELS:
        raise KeyError("unknown model alias %r" % alias)
    provider, model_id, family, tier, extra = MODELS[alias]
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            if provider == "moonshot":
                payload = {"model": model_id, "messages": messages,
                           "max_tokens": max_tokens}
                payload.update(extra)
                # Kimi models accept only temperature = 1; passing anything
                # else is rejected, so the field is simply omitted.
                out = _post(MOONSHOT_BASE + "/chat/completions", payload,
                            {"Authorization": "Bearer " + _key(),
                             "Content-Type": "application/json"}, timeout)
                msg = out["choices"][0]["message"]
                text = msg.get("content") or ""
                usage = out.get("usage", {})
                finish = out["choices"][0].get("finish_reason")
            else:
                opts = {"num_predict": max_tokens}
                if temperature is not None:
                    opts["temperature"] = temperature
                if seed is not None:
                    opts["seed"] = int(seed)
                payload = {"model": model_id, "messages": messages,
                           "stream": False, "options": opts}
                payload.update(extra)
                out = _post(OLLAMA_BASE + "/api/chat", payload,
                            {"Content-Type": "application/json"}, timeout)
                msg = out.get("message", {}) or {}
                text = msg.get("content") or ""
                if not text and msg.get("thinking"):
                    # a reasoning model that ran out of budget before
                    # answering; the reasoning text still carries the id
                    text = msg["thinking"]
                usage = {"prompt_tokens": out.get("prompt_eval_count", 0),
                         "completion_tokens": out.get("eval_count", 0)}
                finish = out.get("done_reason")
            dt = time.time() - t0
            USAGE.add(alias, usage, dt)
            meta = {"alias": alias, "model_id": model_id, "family": family,
                    "tier": tier, "usage": usage, "latency_s": round(dt, 3),
                    "finish_reason": finish,
                    "called_at": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(timespec="seconds"),
                    "attempt": attempt}
            _log(log_tag, messages, text, meta, out)
            return text, meta
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            last_err = "HTTP %s %s" % (e.code, body)
            if e.code in (429, 500, 502, 503, 504):
                # the hosted endpoint sheds load under concurrency; back off
                # generously rather than losing the run
                time.sleep(min(2 ** attempt * 5, 180) * (1.0 + 0.3 * attempt))
                continue
            break
        except Exception as e:                      # noqa: BLE001
            last_err = repr(e)
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError("chat failed for %s: %s" % (alias, last_err))


def _log(tag, messages, text, meta, raw):
    rec = {"tag": tag, "meta": meta, "messages": messages,
           "response": text, "raw": raw}
    day = datetime.date.today().isoformat()
    with open(os.path.join(LOGDIR, "calls-%s.jsonl" % day), "a",
              encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ollama_available():
    try:
        with urllib.request.urlopen(OLLAMA_BASE + "/api/tags", timeout=10) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
