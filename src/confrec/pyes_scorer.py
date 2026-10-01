"""Token-level P(Yes) scorer for LLM recommendation panels (vLLM), multi-question.

Input rows (`*.jsonl`) — either a same-candidate next-item panel (Lumen `ranking_test.jsonl`:
`positive_item_index`) or a rated panel (`candidate_labels`, `candidate_ratings`):
    user_id, source_event_id, history (list[str]), candidate_item_ids, candidate_titles, candidate_texts, ...

For every (user, candidate, question) the confidence is read from the next-token distribution:
    logit = log P(Yes) - log P(No),  p_yes = sigmoid(logit)
where P(Yes)/P(No) sum the top-k tokens whose stripped lower-cased text is "yes"/"no". All questions for a
candidate share the prompt prefix (instruction + history + candidate), so vLLM prefix caching amortises it.

`--swap_k K` additionally scores every unique candidate item under K other users' histories with the
first question (user-marginalised item prior π(i), used by the evidence decomposition).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from pathlib import Path

import numpy as np

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


def build_prompt(history: list[str], title: str, text: str, question: str, hist_len: int) -> str:
    hist = history[-hist_len:] if hist_len > 0 else []
    hist_block = "\n".join(f"- {h}" for h in hist) if hist else "- (no history available)"
    meta = text[:200] if text else ""
    desc = f"\nDescription: {meta}" if meta else ""
    return BODY.format(hist=hist_block, title=title, desc=desc, question=QUESTIONS[question])


def yes_no_logprobs(top: dict) -> tuple[float, float, float]:
    """Return (log P(yes), log P(no), P(yes)+P(no)) from a vLLM top-k logprob dict."""
    p_yes = p_no = 0.0
    for lp in top.values():
        tok = (lp.decoded_token or "").strip().lower()
        if tok == "yes":
            p_yes += math.exp(lp.logprob)
        elif tok == "no":
            p_no += math.exp(lp.logprob)
    return math.log(max(p_yes, 1e-12)), math.log(max(p_no, 1e-12)), p_yes + p_no


def labels_of(rec: dict) -> list[int]:
    if "candidate_labels" in rec:
        return [int(x) for x in rec["candidate_labels"]]
    pos = rec["positive_item_index"]
    return [int(i == pos) for i in range(len(rec["candidate_item_ids"]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--questions", default="next", help="comma list of " + ",".join(QUESTIONS))
    ap.add_argument("--hist_len", type=int, default=10, help="0 = no-history ablation")
    ap.add_argument("--n_users", type=int, default=None)
    ap.add_argument("--swap_k", type=int, default=0, help="K other-user histories per unique item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk_logprobs", type=int, default=20)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=3072)
    ap.add_argument("--chunk_users", type=int, default=1000)
    ap.add_argument("--lora", default=None, help="PEFT adapter dir from train_lora_yesno (scored via vLLM LoRA)")
    args = ap.parse_args()

    questions = [q for q in args.questions.split(",") if q]
    backbone = args.backbone or Path(args.model).name
    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.n_users:
        records = records[: args.n_users]

    from vllm import LLM, SamplingParams

    lora_kw = dict(enable_lora=True, max_lora_rank=64) if args.lora else {}
    llm = LLM(model=args.model, tokenizer=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, dtype="bfloat16",
              seed=args.seed, **lora_kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.topk_logprobs)
    gen_kw = {}
    if args.lora:
        from vllm.lora.request import LoRARequest
        gen_kw["lora_request"] = LoRARequest("adapter", 1, args.lora)

    def fmt(p: str) -> str:
        msg = [{"role": "user", "content": p}]
        try:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, n_prompts, mass_sum = time.time(), 0, 0.0
    with gzip.open(out_dir / "scores.csv.gz", "wt", newline="", encoding="utf-8") as fz:
        w = csv.writer(fz)
        w.writerow(["source_event_id", "user_id", "item_id", "cand_idx", "label", "question",
                    "lp_yes", "lp_no", "logit", "yes_no_mass"])
        for c0 in range(0, len(records), args.chunk_users):
            chunk = records[c0: c0 + args.chunk_users]
            prompts, meta = [], []
            for rec in chunk:
                texts = rec.get("candidate_texts") or [""] * len(rec["candidate_titles"])
                labels = labels_of(rec)
                for i, (title, text) in enumerate(zip(rec["candidate_titles"], texts)):
                    for q in questions:
                        prompts.append(fmt(build_prompt(rec["history"], title, text, q, args.hist_len)))
                        meta.append((rec, i, labels[i], q))
            outs = llm.generate(prompts, sp, use_tqdm=False, **gen_kw)
            for (rec, i, lab, q), o in zip(meta, outs):
                lpy, lpn, mass = yes_no_logprobs(o.outputs[0].logprobs[0])
                mass_sum += mass
                w.writerow([rec.get("source_event_id", rec["user_id"]), rec["user_id"],
                            rec["candidate_item_ids"][i], i, lab, q, f"{lpy:.6f}", f"{lpn:.6f}",
                            f"{lpy - lpn:.6f}", f"{mass:.6f}"])
            n_prompts += len(prompts)
            print(f"[{c0 + len(chunk)}/{len(records)} users] {n_prompts / (time.time() - t0):.0f} prompts/s",
                  flush=True)

    n_main = n_prompts
    swap_info = {}
    if args.swap_k > 0:
        rng = random.Random(args.seed)
        items = {}
        for rec in records:
            texts = rec.get("candidate_texts") or [""] * len(rec["candidate_titles"])
            for iid, title, text in zip(rec["candidate_item_ids"], rec["candidate_titles"], texts):
                items.setdefault(iid, (title, text, rec["user_id"]))
        prompts, meta = [], []
        for iid, (title, text, owner) in items.items():
            donors = [r for r in rng.sample(records, min(len(records), args.swap_k + 1))
                      if r["user_id"] != owner][: args.swap_k]
            for d in donors:
                prompts.append(fmt(build_prompt(d["history"], title, text, questions[0], args.hist_len)))
                meta.append((iid, d["user_id"]))
        outs = llm.generate(prompts, sp, use_tqdm=False, **gen_kw)
        with gzip.open(out_dir / "swap_prior.csv.gz", "wt", newline="", encoding="utf-8") as fz:
            w = csv.writer(fz)
            w.writerow(["item_id", "donor_user_id", "question", "logit"])
            for (iid, donor), o in zip(meta, outs):
                lpy, lpn, _ = yes_no_logprobs(o.outputs[0].logprobs[0])
                w.writerow([iid, donor, questions[0], f"{lpy - lpn:.6f}"])
        n_prompts += len(prompts)
        swap_info = {"swap_k": args.swap_k, "swap_items": len(items), "swap_prompts": len(prompts)}

    report = dict(n_users=len(records), n_prompts=n_prompts, inference_time_s=time.time() - t0,
                  mean_yes_no_mass=mass_sum / max(1, n_main), data_path=args.data, model=args.model,
                  backbone=backbone, questions=questions, hist_len=args.hist_len, seed=args.seed,
                  lora=args.lora,
                  scorer="token_pyes_multiq", **swap_info)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
