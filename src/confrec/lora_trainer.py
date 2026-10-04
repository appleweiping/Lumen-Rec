"""Memory-lean trainer for the yes/no LoRA runs (Gate-FT, fine-tuned grid).

`train_lora_yesno.YesNoSet` builds sequences `prompt + answer` (left padded, labels = -100 everywhere except the single
answer token). The stock `Trainer` loss then materialises B x L x |V| logits (|V| ~ 152k for Qwen3: about 3 GB in fp32
for B = 8, L = 600) only to read one position per example. `last_token_trainer` asks the model for the logits of the
last two positions (`logits_to_keep=2`) and takes the cross-entropy of the answer token from position -2, which is the
same number as the shifted full-sequence loss with the prompt masked (tests/test_confrec_lora_trainer.py).

Under gradient accumulation the stock causal-LM loss is `sum(CE) / num_items_in_batch` (the label count of the whole
accumulation window) and the Trainer then skips its own division by the number of accumulation steps; a plain
micro-batch mean would therefore be summed over the window (a gradient `grad_accum` times too large, which changes
what gradient clipping does). `last_token_loss` follows the stock rule, and the test trains the same tiny LoRA model
with both trainers and compares the updated weights.

This module imports no `transformers`: the trainer class is built around the `Trainer` class the caller passes in, so
the CPU tests that fake `transformers` and the GPU server (real one) share the same code.
"""
from __future__ import annotations

import torch.nn.functional as F


def last_token_loss(model, input_ids, attention_mask, labels, return_outputs: bool = False, num_items_in_batch=None):
    """Cross-entropy of the final token given the rest; labels must be -100 everywhere except the last position.
    With `num_items_in_batch` (the Trainer's label count of the accumulation window) the loss is the stock
    `sum / num_items_in_batch`; without it, the batch mean."""
    if labels.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError(f"labels {tuple(labels.shape)} must match input_ids {tuple(input_ids.shape)}")
    if bool((labels[:, -1] == -100).any()) or bool((labels[:, :-1] != -100).any()):
        raise ValueError("every example must end in exactly one answer token (left padding, prompt labels masked)")
    out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=2)
    logits = out.logits[:, -2, :].float()            # the position that predicts the last token
    if num_items_in_batch is None:
        loss = F.cross_entropy(logits, labels[:, -1])
    else:
        loss = F.cross_entropy(logits, labels[:, -1], reduction="sum") / num_items_in_batch
    return (loss, out) if return_outputs else loss


def last_token_trainer(trainer_base):
    """`trainer_base` is `transformers.Trainer`; returns the subclass whose loss is `last_token_loss`."""

    class LastTokenTrainer(trainer_base):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return last_token_loss(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"],
                                   return_outputs, num_items_in_batch)

    return LastTokenTrainer
