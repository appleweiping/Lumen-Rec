"""LoRA fine-tuning of a yes/no LLM recommender (TALLRec-style) with optional *mirror tuning*.

Each rated training example (user history, candidate, label) becomes
  standard : "Would this user like the candidate item?"    -> "Yes" if label else "No"
  mirror   : + "Would this user dislike the candidate item?" -> "No" if label else "Yes"
The loss is cross-entropy on the single answer token only (prompt tokens masked), i.e. a proper scoring rule
on P(Yes) restricted to {Yes, No}. Prompts are byte-identical to `pyes_scorer.build_prompt`, so the adapter is
scored by the same scorer (vLLM `--enable-lora` or a merged checkpoint).

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

from src.confrec.pyes_scorer import build_prompt


class YesNoSet(Dataset):
    def __init__(self, rows, tok, mode, hist_len, max_len):
        self.items = []
        for r in rows:
            texts = r.get("candidate_texts") or [""] * len(r["candidate_titles"])
            for title, text, y in zip(r["candidate_titles"], texts, r["candidate_labels"]):
                qa = [("like", "Yes" if y else "No")]
                if mode == "mirror":
                    qa.append(("dislike", "No" if y else "Yes"))
                for q, ans in qa:
                    msg = [{"role": "user", "content": build_prompt(r["history"], title, text, q, hist_len)}]
                    try:
                        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=False)
                    except TypeError:
                        prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-(max_len - 1):]
                    a_ids = tok(ans, add_special_tokens=False)["input_ids"][:1]  # first answer token only
                    self.items.append((p_ids + a_ids, len(p_ids)))

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
    if a.max_examples:
        ds.items = ds.items[: a.max_examples]
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
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "train_config.json").write_text(json.dumps({**vars(a), "n_examples": len(ds)}, indent=2))


if __name__ == "__main__":
    main()
