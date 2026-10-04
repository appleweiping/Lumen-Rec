"""FAKE GPU scorer for the amendment-2 CPU dry run (round 2). Same CLI as `python -m src.confrec.pyes_scorer`.

It runs the REAL src.confrec.pyes_scorer.run of the sandbox copy of the repo (cwd) -- prompt rendering with the
variant bank (system messages for V5/V7 through chat_ids(tok, user, system)), the chat-template guards, chunked parts,
merge, report.json, resume checks, the real load_vllm wiring, the yes/no AND the digit readout -- against a fake
`vllm` module installed in sys.modules. The fake tokenizer is character level (id = ord(c) + 300; "Yes" = 1, "No" = 2)
with a chat template that supports a system message and `enable_thinking` (Qwen3-like: the generation prompt ends with
the closed empty think block when thinking is off).

The fake model decodes each prompt back to (system, user), looks the prompt up in an index built from the panel with the
same rendering the scorer uses, and returns a top-50 logprob dict encoding a planted logit

    like     L = s(panel, variant) * (2 label - 1) + 0.35 z_pop + user offset + brand familiarity
                 + 0.9 noise(user, item, like, title) [+ 0.5 variant noise for V != V0]
                 + 0.12 (displayed rating of the last history event - 3)
    dislike  L = -s * (2 label - 1) + ...      like_para = like + 0.6 noise
    next     L = -2 + 1.6 (2 label - 1) + 0.3 z + ...
    T1       "(This user later rated this item r/5.)" is read from the prompt: L = 1.0 (r - 3) + noise (a readout that
             follows the stated rating; scenario t1_broken ignores it)
    T2       "Average rating by other users before this date: m/5 (n ratings)" is read: L += 1.1 (m - 3.5)
    T0_probe item-only: L = 1.6 (probe_ref_prior_mean - 3.5) + noise (the panel row carries the reference)
    digits   the same L through P(r) ~ exp(-(r - mu)^2 / 2), mu = 3 + 1.2 tanh(L / 2); mass 0.99 on the 5 digits
    (no history)  L = 0.4 z + 0.6 noise

The signal s comes from the scenario file ($FAKE_SCENARIO json): {"signal": {"<domain>[_confirm]": {"<variant>": s}},
"default_signal": 0.12, "mass": {"<domain>/<variant>": total Yes+No mass}, "default_mass": 0.99999, "cens2": rate,
"cens1": rate, "t1_broken": bool, "llama_signal": s}. The data file name selects the domain (ml1m / toys / games /
sports / kuairec) and a "confirm" panel its own signal row. The fake asserts that the prompt text carries the markers
of the --variant it was asked to render (system message, threshold question, liked/disliked split, genre / category
brackets, persona line, history window), i.e. that the scorer's rendering honours --variant.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import types
from pathlib import Path

REPO = Path(os.environ.get("FAKE_REPO") or os.getcwd())
sys.path.insert(0, str(REPO))

from src.confrec import pyes_scorer as ps  # noqa: E402
from src.confrec.prompting import SYSTEM_SEP, panel_kind_of, prompt_key, resolve_hist_len  # noqa: E402
from src.confrec.pseudonymize import brand_regex, stop_reason  # noqa: E402

YES, NO, OFF = 1, 2, 300
FILLERS = list(range(3, 51))   # 48 filler ids: 50 returned logprobs with Yes and No
SCEN = (json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text(encoding="utf-8"))
        if os.environ.get("FAKE_SCENARIO") else {})
SENT_RE = re.compile(r"^(?:<\|im_start\|>system\n(.*?)<\|im_end\|>\n)?<\|im_start\|>user\n(.*)<\|im_end\|>\n"
                     r"<\|im_start\|>assistant\n(?:<think>\n\n</think>\n\n)?$", re.S)
T1_RE = re.compile(r"\(This user later rated this item (\d)/5\.\)")
T2_RE = re.compile(r"Average rating by other users before this date: (?:([\d.]+)/5 \((\d+) ratings\)|no ratings yet)")
RATED_RE = re.compile(r"\(rated (\d)/5\)")
LINE_RE = re.compile(r"^- .*$", re.M)
# variant -> (system message, threshold question, liked/disliked split, [meta] brackets, persona line)
MARKERS = {"V0": (False, False, False, False, True), "V1": (False, True, False, False, True),
           "V2": (False, False, True, False, True), "V3": (False, False, False, False, True),
           "V4": (False, False, False, True, True), "V5": (True, False, False, False, False),
           "V7": (True, True, False, True, False)}


def h01(*parts) -> float:
    return int(hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def gauss(*parts) -> float:
    u1, u2 = max(h01(*parts, "a"), 1e-12), h01(*parts, "b")
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


class Tok:
    chat_template = "{# Qwen3-like #}{% if enable_thinking is defined and enable_thinking is false %}<think>{% endif %}"

    def __len__(self):
        return OFF + 0x110000

    def decode(self, ids):
        return "".join("Yes" if i == YES else "No" if i == NO else chr(i - OFF) if i >= OFF else "" for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msg)
        out += "<|im_start|>assistant\n"
        if not enable_thinking:
            out += "<think>\n\n</think>\n\n"
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) + OFF for c in text]}


class LP:
    def __init__(self, v):
        self.logprob = v


def domain_key(data) -> str:
    n = Path(data).name.lower()
    d = ("ml1m" if "ml1m" in n else "toys" if "toys" in n else "games" if "games" in n else
         "sports" if "sports" in n else "kuairec" if "kuai" in n else "x")
    return d + ("_confirm" if "confirm" in n else "")


def signal(dkey: str, variant: str, llama: bool) -> float:
    if llama:
        return float(SCEN.get("llama_signal", 0.20))
    return float(((SCEN.get("signal") or {}).get(dkey) or {}).get(variant, SCEN.get("default_signal", 0.12)))


def check_markers(variant, system, user, rec, hist) -> None:
    exp = MARKERS.get(variant)
    if exp is None:
        return
    got = (system is not None, "stars or higher" in user, "Items this user liked" in user,
           bool(re.search(r"^- .* \[(?:Genres|Categories): .*\] \(rated \d/5\)$", user, re.M)),
           user.startswith("You are an expert recommendation system."))
    # meta brackets only when some history event of the window has a non-empty history_meta
    if variant in ("V4", "V7"):
        hm = (rec.get("history_meta") or [])[-hist:] if hist else []
        exp = exp[:3] + (any(hm),) + exp[4:]
    assert got == exp, f"{variant}: rendered markers {got} != expected {exp}\n{user[:400]}"
    if "Items this user liked" not in user:
        block = user.split("User history (oldest to newest):\n", 1)[1].split("\n\nCandidate item:", 1)[0]
        n = 0 if block.startswith("- (no history") else len(LINE_RE.findall(block))
        assert n == min(hist, len(rec["history"])), f"{variant}: {n} history lines, window {hist}"


def build_index(args, questions, kind, hist):
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if args.n_users:
        rows = rows[: args.n_users]
    dkey = domain_key(args.data)
    llama = "llama" in Path(args.model).name.lower()
    brands = {str(b).strip() for r in rows for f in ("candidate_brands", "history_brands") for b in r.get(f) or []
              if stop_reason(b) is None}
    rx = brand_regex(brands) if brands else None
    pops = [math.log1p(float(x)) for r in rows if "candidate_popularity" in r for x in r["candidate_popularity"]]
    mu = sum(pops) / len(pops) if pops else 0.0
    sd = (sum((x - mu) ** 2 for x in pops) / len(pops)) ** 0.5 if pops else 1.0
    s = signal(dkey, args.variant, llama)
    t1_broken = bool(SCEN.get("t1_broken"))

    def z_of(r, i):
        if "candidate_popularity" in r:
            return (math.log1p(float(r["candidate_popularity"][i])) - mu) / (sd or 1.0)
        g = (r.get("candidate_popularity_groups") or ["mid"] * len(r["candidate_item_ids"]))[i]
        return {"head": 1.0, "mid": 0.0, "tail": -1.0}.get(str(g).lower(), 0.0)

    acc, item_prior, n_checked = {}, {}, 0
    salt = (args.variant, "llama" if llama else "qwen")
    for r in rows:
        u = str(r["user_id"])
        uo = 0.5 * gauss("user", u)
        labels = ps.labels_of(r)
        for i, q, p in ps.record_requests(r, questions, hist, args.variant, kind, args.readout):
            title, item = str(r["candidate_titles"][i]), str(r["candidate_item_ids"][i])
            user, system = str(p), p.system
            if kind == "rated" and args.variant != "T0_probe":
                check_markers(args.variant, system, user, r, hist)
                n_checked += 1
            z = z_of(r, i)
            fam = 0.5 * z if rx is not None and rx.search(title) else 0.0
            y = 2 * labels[i] - 1
            e = gauss(u, item, q, title)
            vn = 0.0 if args.variant == "V0" and not llama else 0.5 * gauss(u, item, q, title, *salt)
            shown = [int(x) for x in RATED_RE.findall(user.split("\n\nCandidate item:", 1)[0])]
            last = shown[-1] if shown else 3
            t1, t2 = T1_RE.search(user), T2_RE.search(user)
            if args.variant == "T0_probe":
                pm = r.get("probe_ref_prior_mean")
                L = (1.6 * (float(pm) - 3.5) if pm is not None else 0.0) + 0.6 * gauss("t0", item)
            elif hist == 0 and kind == "rated":
                L = 0.4 * z + 0.6 * gauss("nohist", item, q, title)
            elif q == "next":
                L = -2.0 + 1.6 * y + 0.3 * z + uo + 0.9 * e
            elif t1 is not None and not t1_broken:
                L = 1.0 * (int(t1.group(1)) - 3) + 0.4 * gauss("t1", u, item) + uo
            elif q in ("like", "like_para", "quality", "rating"):
                base = s * y + 0.35 * z + uo + fam + 0.9 * gauss(u, item, "like", title) + vn + 0.12 * (last - 3)
                if t2 is not None and t2.group(1) is not None:
                    base += 1.1 * (float(t2.group(1)) - 3.5)
                L = base if q != "like_para" else base + 0.6 * e
            elif q in ("dislike", "dislike_para"):
                L = -s * y + 0.35 * z + uo + fam - 0.4 + 0.9 * e + vn
            else:
                L = e
            acc.setdefault(prompt_key(p), []).append(L)
            item_prior.setdefault(title, 0.35 * z + 0.5 * fam)
    return {k: sum(v) / len(v) for k, v in acc.items()}, item_prior, dkey, n_checked


def install_fake_vllm(index, item_prior, args, dkey) -> dict:
    tok = Tok()
    stats = {"generated": 0, "indexed": 0, "swap": 0}
    title_rx = re.compile(r"\nTitle: (.*?)(?:\nDescription: |\n\n)", re.S)
    mass_total = float(((SCEN.get("mass") or {}).get(f"{dkey}/{args.variant}", SCEN.get("default_mass", 0.99999))))
    cens1, cens2 = float(SCEN.get("cens1", 0.0)), float(SCEN.get("cens2", 0.0))
    digits = args.readout == "digits"

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class LLM:
        def __init__(self, **kw):
            self.kw = kw
            assert kw["dtype"] == "float16" and kw["max_logprobs"] == 50, kw
            print("FAKE LLM kwargs:", {k: kw[k] for k in ("model", "max_model_len", "dtype", "max_logprobs",
                                                          "enable_prefix_caching")}, flush=True)

        def get_tokenizer(self):
            return tok

        def generate(self, prompts, sp, use_tqdm=False, **kw):
            assert sp.kw["logprobs"] == 50 and sp.kw["max_tokens"] == 1 and sp.kw["temperature"] == 0.0
            out = []
            for pr in prompts:
                text = tok.decode(pr["prompt_token_ids"])
                m = SENT_RE.match(text)
                assert m, f"unparseable chat text {text[:200]!r}"
                system, user = m.group(1), m.group(2)
                key = user if system is None else system + SYSTEM_SEP + user
                if key in index:
                    L = index[key]
                    stats["indexed"] += 1
                else:   # swap prompt: an item under another user's history
                    mt = title_rx.search(user)
                    L = item_prior.get(mt.group(1) if mt else "", 0.0) + 0.8 * gauss("swap", key)
                    stats["swap"] += 1
                stats["generated"] += 1
                hz = h01("cens", key)
                top = {t: LP(math.log((1 - mass_total + 1e-9) / len(FILLERS)) - 0.01 * k)
                       for k, t in enumerate(FILLERS)}
                floor = min(v.logprob for v in top.values())
                if digits:
                    mu_r = 3.0 + 1.2 * math.tanh(L / 2)
                    w = {r: math.exp(-((r - mu_r) ** 2) / 2) for r in range(1, 6)}
                    tot = sum(w.values())
                    for r, x in w.items():
                        top[ord(str(r)) + OFF] = LP(math.log(0.99 * x / tot))
                    if hz < cens2:
                        for r in range(1, 6):
                            top.pop(ord(str(r)) + OFF, None)
                else:
                    py = mass_total / (1 + math.exp(-L))
                    ly, ln = math.log(max(py, 1e-300)), math.log(max(mass_total - py, 1e-300))
                    if ly >= floor:
                        top[YES] = LP(ly)
                    if ln >= floor and not (cens1 and hz < cens1):
                        top[NO] = LP(ln)
                    if cens2 and hz < cens2:
                        top.pop(YES, None)
                        top.pop(NO, None)
                out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[top])]))
            return out

    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.30.0+fake"
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    inputs = types.ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass
    inputs.TokensPrompt = TokensPrompt
    vllm.inputs = inputs
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = lambda *a: ("lora",) + a
    lora.request = req
    vllm.lora = lora
    sys.modules.update({"vllm": vllm, "vllm.inputs": inputs, "vllm.lora": lora, "vllm.lora.request": req})
    return stats


def main() -> None:
    args = ps.parse_args(sys.argv[1:])
    questions = [q for q in args.questions.split(",") if q]
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    kind = panel_kind_of(rows[0])
    hist = resolve_hist_len(args.variant, kind, args.hist_len)
    del rows
    index, prior, dkey, n_checked = build_index(args, questions, kind, hist)
    stats = install_fake_vllm(index, prior, args, dkey)
    print(f"FAKE scorer: variant {args.variant} readout {args.readout} kind {kind} hist {hist} domain {dkey} "
          f"signal {signal(dkey, args.variant, 'llama' in Path(args.model).name.lower())} "
          f"scenario {os.environ.get('FAKE_SCENARIO')} markers checked {n_checked}", flush=True)
    ps.run(args)   # default loader = the real ps.load_vllm, importing the fake vllm
    print("FAKE scorer done:", stats, flush=True)


if __name__ == "__main__":
    main()
