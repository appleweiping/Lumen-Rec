"""LoRA fine-tuning of a yes/no LLM recommender (TALLRec-style) with optional *mirror tuning*.

Each rated training example (user history, candidate, label) becomes
  standard : "Would this user like the candidate item?"    -> "Yes" if label else "No"
  mirror   : + "Would this user dislike the candidate item?" -> "No" if label else "Yes"
The loss is cross-entropy on the single answer token only (prompt tokens masked), i.e. a proper scoring rule
on P(Yes) restricted to {Yes, No}. Prompt token ids come from `prompting.build_prompt` + `prompting.chat_ids`,
the exact ids `pyes_scorer` sends to vLLM, so the adapter is scored on the sequence it was trained on. The answer
ids are tok("Yes") / tok("No"), asserted to be single tokens inside the scorer's yes/no id sets. Over-long
prompts are never truncated: examples with len(prompt) + 1 > --max_len are skipped and counted
(<out>/train_report.json).

    python -m src.confrec.train_lora_yesno --train panels/toys_rated_train.jsonl --model <Qwen3-8B> \
        --out runs/toys_qwen_mirror_s0 --mode mirror --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.confrec.prompting import build_prompt, chat_ids, yes_no_ids
from src.confrec.stats import strict_json


def answer_id(tok, word: str, allowed) -> int:
    ids = tok(word, add_special_tokens=False)["input_ids"]
    if len(ids) != 1 or ids[0] not in allowed:
        raise ValueError(f"{word!r} tokenizes to {ids}, not one id from the scorer's set {sorted(allowed)}")
    return ids[0]


class YesNoSet(Dataset):
    def __init__(self, rows, tok, mode, hist_len, max_len):
        yes, no = yes_no_ids(tok)
        self.answer = {"Yes": answer_id(tok, "Yes", yes), "No": answer_id(tok, "No", no)}
        self.items, self.n_skipped, self.max_prompt_len = [], 0, 0
        for r in rows:
            texts = r.get("candidate_texts") or [""] * len(r["candidate_titles"])
            for title, text, y in zip(r["candidate_titles"], texts, r["candidate_labels"]):
                qa = [("like", "Yes" if y else "No")]
                if mode == "mirror":
                    qa.append(("dislike", "No" if y else "Yes"))
                for q, ans in qa:
                    p_ids = chat_ids(tok, build_prompt(r["history"], title, text, q, hist_len))
                    self.max_prompt_len = max(self.max_prompt_len, len(p_ids))
                    if len(p_ids) + 1 > max_len:
                        self.n_skipped += 1
                        continue
                    self.items.append((p_ids + [self.answer[ans]], len(p_ids)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, n_prompt = self.items[i]
        labels = [-100] * n_prompt + ids[n_prompt:]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for k, b in enumerate(batch):  # left padding keeps the answer token at the end
        n = len(b["input_ids"])
        ids[k, m - n:] = torch.tensor(b["input_ids"])
        lab[k, m - n:] = torch.tensor(b["labels"])
        att[k, m - n:] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["standard", "mirror"], default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist_len", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_examples", type=int, default=None)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    random.Random(a.seed).shuffle(rows)
    ds = YesNoSet(rows, tok, a.mode, a.hist_len, a.max_len)
    n_built = len(ds)
    if a.max_examples:
        ds.items = ds.items[: a.max_examples]
    print(f"examples: {n_built} built, {ds.n_skipped} skipped (prompt + answer > max_len={a.max_len}), "
          f"{len(ds)} used; longest prompt {ds.max_prompt_len} tokens", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_report.json").write_text(json.dumps(strict_json(dict(
        n_examples=len(ds), n_examples_built=n_built, n_skipped_overlength=ds.n_skipped, max_len=a.max_len,
        max_prompt_len=ds.max_prompt_len, mode=a.mode, hist_len=a.hist_len, answer_ids=ds.answer,
        train=a.train, model=a.model)), indent=2))
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bsz,
                             gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
                             learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
                             logging_steps=20, save_strategy="no", report_to=[], seed=a.seed,
                             remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=lambda b: collate(b, tok.pad_token_id)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (out / "train_config.json").write_text(json.dumps(strict_json(
        {**vars(a), "n_examples": len(ds), "n_skipped_overlength": ds.n_skipped}), indent=2))


if __name__ == "__main__":
    main()
