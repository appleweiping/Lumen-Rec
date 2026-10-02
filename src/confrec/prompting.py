"""Prompt rendering and Yes/No token reading shared by `pyes_scorer` and `train_lora_yesno`.

Both paths build the user message with `build_prompt` and tokenize it with `chat_ids`, so the token ids vLLM
scores are exactly the ids a LoRA adapter was trained on (amendment 1, C0):

    from src.confrec.prompting import build_prompt, chat_ids, read_yes_no, yes_no_ids
    ids = chat_ids(tok, build_prompt(history, title, text, "like", hist_len=10))
    yes_ids, no_ids = yes_no_ids(tok)                      # resolved once per tokenizer, logged by the scorer
    lp_yes, lp_no, mass, censored = read_yes_no(vllm_output.outputs[0].logprobs[0], yes_ids, no_ids)
"""
from __future__ import annotations

import math
import re

QUESTIONS = {
    "next": "Will this user purchase the candidate item next?",
    "like": "Would this user like the candidate item?",
    "dislike": "Would this user dislike the candidate item?",
    "like_para": "Is the candidate item a good match for this user's taste?",
    "dislike_para": "Would this user be disappointed by the candidate item?",
}
BODY = ("You are an expert recommendation system.\n\n"
        "User history (oldest to newest):\n{hist}\n\n"
        "Candidate item:\nTitle: {title}{desc}\n\n"
        "{question} Answer with only Yes or No.")
MAX_TITLE_CHARS = 200
_RATED = re.compile(r"^(.*)( \(rated \d+/5\))$", re.S)  # rated-panel history line "<title> (rated r/5)"


def cap_history_line(h, max_title_chars: int = MAX_TITLE_CHARS) -> str:
    """Cap the title part of a history line; a rated line keeps its " (rated r/5)" suffix."""
    h = str(h)
    m = _RATED.match(h)
    if m:
        return m.group(1)[:max_title_chars] + m.group(2)
    return h[:max_title_chars]


def build_prompt(history: list[str], title: str, text: str, question: str, hist_len: int,
                 max_title_chars: int = MAX_TITLE_CHARS) -> str:
    hist = history[-hist_len:] if hist_len > 0 else []
    hist_block = ("\n".join(f"- {cap_history_line(h, max_title_chars)}" for h in hist) if hist
                  else "- (no history available)")
    meta = text[:200] if text else ""
    desc = f"\nDescription: {meta}" if meta else ""
    return BODY.format(hist=hist_block, title=str(title)[:max_title_chars], desc=desc,
                       question=QUESTIONS[question])


def chat_ids(tok, prompt: str) -> list[int]:
    """Chat-template the prompt (thinking off) and tokenize without extra special tokens.

    The template already inserts BOS where the model needs it; tokenizing its text with the default
    add_special_tokens=True (what vLLM does for a text prompt) gives Llama-3 a second BOS."""
    msg = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def yes_no_ids(tok) -> tuple[frozenset, frozenset]:
    """All vocab ids whose decoded text, stripped and lower-cased, is "yes" / "no" (one scan of the vocab)."""
    yes, no = set(), set()
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if s == "yes":
            yes.add(i)
        elif s == "no":
            no.add(i)
    return frozenset(yes), frozenset(no)


def _logsumexp(v: list[float]) -> float:
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def read_yes_no(top: dict, yes_ids, no_ids) -> tuple[float, float, float, int]:
    """(lp_yes, lp_no, mass, censored) from a top-k dict {token_id: obj with .logprob}.

    lp_yes / lp_no = log of the summed probability of the yes / no ids present (finite logprobs only);
    mass = P(yes) + P(no) over the present ids. censored: 0 both sides present; 1 one side absent, imputed with
    the smallest returned logprob `floor` (amendment 1, C0); 2 both absent -> (nan, nan, 0, 2).
    For censored=1 the floor upper-bounds each absent id, not the absent side's sum: that side's log probability is
    <= floor + log(n ids of that side), and the present side sums only its ids in the top-k. So the true logit is
    >= lp_yes - floor - log|no_ids| when No is absent, and <= floor + log|yes_ids| - lp_no when Yes is absent."""
    ys, ns, seen = [], [], []
    for tid, lp in top.items():
        v = float(lp.logprob)
        if not math.isfinite(v):
            continue
        seen.append(v)
        if tid in yes_ids:
            ys.append(v)
        elif tid in no_ids:
            ns.append(v)
    if not ys and not ns:
        return math.nan, math.nan, 0.0, 2
    mass = sum(math.exp(v) for v in ys + ns)
    floor = min(seen)
    return (_logsumexp(ys) if ys else floor, _logsumexp(ns) if ns else floor, mass, 0 if ys and ns else 1)
