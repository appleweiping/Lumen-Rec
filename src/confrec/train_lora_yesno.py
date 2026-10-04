"""LoRA fine-tuning of a yes/no LLM recommender (TALLRec-style) with optional *mirror tuning*.

Each rated training example (user history, candidate, label) becomes, under prompt variant `--variant`
(prompting.VARIANTS, amendment 2 G9: V* or V0; default V0):
  standard : the variant's like question    -> "Yes" if label else "No"
  mirror   : + the variant's dislike question -> "No" if label else "Yes"
(V0: "Would this user like / dislike the candidate item?"; V1/V7: the threshold family "Will this user rate the
candidate item 4 stars or higher / 2 stars or lower (on a 1-5 scale)?".)
The loss is cross-entropy on the single answer token only (prompt tokens masked), i.e. a proper scoring rule
on P(Yes) restricted to {Yes, No}; `lora_trainer.last_token_loss` computes it from the last two positions' logits
only (equal to the full-sequence loss, tests/test_confrec_lora_trainer.py). Prompt token ids come from
`prompting.render_record` + `prompting.chat_ids`
(system message included for V5/V7), the exact ids `pyes_scorer --variant` sends to vLLM, so the adapter is scored
on the sequence it was trained on; train_config.json records the variant and pyes_scorer --lora refuses another.
`--hist_len` defaults to the variant's registered history window (V0: 10, V3/V7: 20). Before any model is loaded the
panel is checked as pyes_scorer checks it (`check_training_panel`): one panel kind; a variant other than V0 only on a
rated panel (next-item prompts stay frozen at V0, amendment 2 G0); a registered window; and a V3/V7 run is refused
on a panel whose rows never hold more than 10 history events (it would train on the 10-event rendering while the
CONFIRM evaluation renders 20). train_report.json / train_config.json record the variant, the window used
(hist_len_used, which pyes_scorer --lora checks) and the panel's longest history (max_history_len_in_panel).
The answer ids are tok("Yes") / tok("No"), asserted to be single tokens inside the scorer's yes/no id sets.
Over-long prompts are never truncated: examples with len(prompt) + 1 > --max_len are skipped and counted
(<out>/train_report.json).

    python -m src.confrec.train_lora_yesno --train panels/toys_rated_train.jsonl --model <Qwen3-8B> \
        --out runs/toys_qwen_mirror_s0 --mode mirror --seed 0 [--variant V3]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.confrec.lora_trainer import last_token_trainer
from src.confrec.prompting import (GATE_VARIANTS, chat_ids, panel_kind_of, render_record, resolve_hist_len,
                                   short_history_error, yes_no_ids)
from src.confrec.stats import strict_json


def answer_id(tok, word: str, allowed) -> int:
    ids = tok(word, add_special_tokens=False)["input_ids"]
    if len(ids) != 1 or ids[0] not in allowed:
        raise ValueError(f"{word!r} tokenizes to {ids}, not one id from the scorer's set {sorted(allowed)}")
    return ids[0]


class YesNoSet(Dataset):
    """hist_len None = the variant's registered history window (pyes_scorer's default)."""

    def __init__(self, rows, tok, mode, hist_len, max_len, variant="V0"):
        if variant not in GATE_VARIANTS:
            raise ValueError(f"variant {variant!r} is not a yes/no gate variant {GATE_VARIANTS}")
        yes, no = yes_no_ids(tok)
        self.answer = {"Yes": answer_id(tok, "Yes", yes), "No": answer_id(tok, "No", no)}
        self.variant = variant
        self.items, self.n_skipped, self.max_prompt_len = [], 0, 0
        qs = ["like", "dislike"] if mode == "mirror" else ["like"]
        for r in rows:
            labels = r["candidate_labels"]
            assert len(labels) == len(r["candidate_titles"]), r.get("source_event_id")
            for i, q, system, user in render_record(r, qs, variant, hist_len=hist_len):
                y = labels[i]
                ans = ("Yes" if y else "No") if q == "like" else ("No" if y else "Yes")
                p_ids = chat_ids(tok, user, system)
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


def check_training_panel(rows, variant: str, hist_len: int | None, name: str = "training panel") -> dict:
    """The scorer's panel rules, applied before any model is loaded (SystemExit on violation). Returns panel_kind,
    hist_len_used (the effective window) and max_history_len_in_panel."""
    kinds = sorted({panel_kind_of(r) for r in rows})
    if len(kinds) != 1:
        raise SystemExit(f"{name}: expected rows of one panel kind, got {kinds or 'no rows'}")
    kind = kinds[0]
    if kind != "rated" and variant != "V0":
        raise SystemExit(f"{name} is a next-item panel: its prompt stays frozen at V0 (amendment 2 G0); --variant "
                         f"{variant} is unregistered there")
    try:
        hist_len_used = resolve_hist_len(variant, kind, hist_len)
    except ValueError as e:
        raise SystemExit(f"{name}: {e}") from None
    max_hist = max(len(r.get("history") or []) for r in rows)
    short = short_history_error(variant, kind, hist_len_used, max_hist)
    if short:
        raise SystemExit(f"{name}: --{short}")
    return {"panel_kind": kind, "hist_len_used": hist_len_used, "max_history_len_in_panel": max_hist}


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


def training_arguments(a) -> dict:
    """The `TrainingArguments` keywords of the G9 recipe (a separate function so a test can hand exactly these keywords
    to the installed transformers: a renamed option, such as warmup_ratio in transformers 5, then fails in CI, not at
    the scheduled start of a 5-GPU-h stage). warmup_steps is a ratio when it is a float below 1 (transformers 5)."""
    return dict(output_dir=a.out, per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.grad_accum,
                num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03, bf16=True,
                logging_steps=20, save_strategy="no", report_to=[], seed=a.seed, remove_unused_columns=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["standard", "mirror"], default="standard")
    ap.add_argument("--variant", choices=list(GATE_VARIANTS), default="V0",
                    help="prompt variant (amendment 2 G9: V*, or V0 under F0); pyes_scorer --lora must use the same")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist_len", type=int, default=None,
                    help="default: the variant's registered history window (V0 10, V3/V7 20); 0 = no history")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_examples", type=int, default=None)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    random.Random(a.seed).shuffle(rows)
    panel = check_training_panel(rows, a.variant, a.hist_len, a.train)   # before any model / tokenizer load
    hist_len_used = panel["hist_len_used"]

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = YesNoSet(rows, tok, a.mode, a.hist_len, a.max_len, a.variant)
    n_built = len(ds)
    if a.max_examples:
        ds.items = ds.items[: a.max_examples]
    print(f"examples: {n_built} built, {ds.n_skipped} skipped (prompt + answer > max_len={a.max_len}), "
          f"{len(ds)} used; longest prompt {ds.max_prompt_len} tokens; variant {a.variant}, hist_len "
          f"{hist_len_used} (longest panel history {panel['max_history_len_in_panel']})", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_report.json").write_text(json.dumps(strict_json(dict(
        n_examples=len(ds), n_examples_built=n_built, n_skipped_overlength=ds.n_skipped, max_len=a.max_len,
        max_prompt_len=ds.max_prompt_len, mode=a.mode, variant=a.variant, hist_len=hist_len_used,
        panel_kind=panel["panel_kind"], max_history_len_in_panel=panel["max_history_len_in_panel"],
        answer_ids=ds.answer, train=a.train, model=a.model)), indent=2))
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    args = TrainingArguments(**training_arguments(a))
    last_token_trainer(Trainer)(model=model, args=args, train_dataset=ds,
                                data_collator=lambda b: collate(b, tok.pad_token_id)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (out / "train_config.json").write_text(json.dumps(strict_json(
        {**vars(a), "loss": "last_token", "hist_len_used": hist_len_used, "panel_kind": panel["panel_kind"],
         "max_history_len_in_panel": panel["max_history_len_in_panel"], "n_examples": len(ds),
         "n_skipped_overlength": ds.n_skipped}), indent=2))


if __name__ == "__main__":
    main()
