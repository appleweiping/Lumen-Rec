import re
import zlib

import pytest
import torch

from src.confrec.prompting import chat_ids
from src.confrec.pyes_scorer import record_requests
from src.confrec.split_panel import bucket
from src.confrec.train_lora_yesno import YesNoSet, collate


class FakeTok:
    """Llama-3-like tokenizer: template text starts with BOS; __call__ adds BOS only with add_special_tokens=True."""
    BOS = "<|begin_of_text|>"
    FIXED = (BOS, "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "Yes", "No", " yes", " No")

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.FIXED)}

    def __len__(self):
        return 1000 + 20000

    def decode(self, ids):
        inv = dict(enumerate(self.FIXED))
        return "".join(inv.get(i, f"<w{i}>") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return f"{self.BOS}<|start_header_id|>user<|end_header_id|>\n\n{msg[0]['content']}<|eot_id|>" \
               "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def __call__(self, text, add_special_tokens=True):
        ids = [0] if add_special_tokens else []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [self.vocab[piece]] if piece in self.FIXED else \
                [self.vocab.get(w, 1000 + zlib.crc32(w.encode()) % 20000) for w in piece.split()]
        return {"input_ids": ids}


def _rows():
    return [{"history": ["A (rated 5/5)"], "candidate_titles": ["X", "Y"], "candidate_texts": ["", ""],
             "candidate_item_ids": ["x", "y"], "candidate_labels": [1, 0]}]


def test_standard_vs_mirror_examples_and_labels():
    tok = FakeTok()
    std = YesNoSet(_rows(), tok, "standard", 10, 512)
    mir = YesNoSet(_rows(), tok, "mirror", 10, 512)
    assert len(std) == 2 and len(mir) == 4
    yes, no = tok("Yes", add_special_tokens=False)["input_ids"][0], tok("No", add_special_tokens=False)["input_ids"][0]
    answers = [ex["labels"][-1] for ex in (mir[i] for i in range(4))]
    # X liked: like->Yes, dislike->No ; Y disliked: like->No, dislike->Yes
    assert answers == [yes, no, no, yes]
    for i in range(4):  # only the answer token is supervised
        assert sum(t != -100 for t in mir[i]["labels"]) == 1


def test_trainer_prompt_ids_equal_scorer_ids_single_bos():
    tok = FakeTok()
    row = _rows()[0]
    for mode, qs in (("standard", ["like"]), ("mirror", ["like", "dislike"])):
        ds = YesNoSet([row], tok, mode, 10, 512)
        scorer = [chat_ids(tok, p) for _, _, p in record_requests(row, qs, 10)]
        trainer = [ids[:n] for ids, n in ds.items]
        assert trainer == scorer
        assert all(ids.count(0) == 1 and ids[0] == 0 for ids in trainer)


def test_overlength_examples_are_skipped_not_left_truncated():
    tok = FakeTok()
    row = dict(_rows()[0], history=["A (rated 5/5)", " ".join(["w"] * 150) + " (rated 2/5)"],
               candidate_titles=["X", "Y has a longer title"])
    full = YesNoSet([row], tok, "mirror", 10, 4096)
    n_prompt = sorted(n for _, n in full.items)
    ds = YesNoSet([row], tok, "mirror", 10, n_prompt[0] + 1)          # only the shortest prompt(s) fit
    assert ds.n_skipped == sum(n + 1 > n_prompt[0] + 1 for n in n_prompt) > 0
    assert len(ds) + ds.n_skipped == len(full) and ds.max_prompt_len == full.max_prompt_len
    assert all(ids[0] == 0 and len(ids) <= n_prompt[0] + 1 for ids, _ in ds.items)   # BOS/header kept


def test_answer_token_must_be_one_scorer_id():
    class SplitYes(FakeTok):
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [4, 1500]} if text == "Yes" else super().__call__(text, add_special_tokens)
    with pytest.raises(ValueError):
        YesNoSet(_rows(), SplitYes(), "standard", 10, 512)


def test_collate_left_pads_and_masks():
    tok = FakeTok()
    ds = YesNoSet(_rows(), tok, "mirror", 10, 512)
    b = collate([ds[0], ds[1]], pad_id=0)
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    assert torch.all(b["labels"][:, -1] != -100)          # answer is the last position for every row
    assert torch.all(b["attention_mask"][:, -1] == 1)


def test_user_split_is_deterministic_and_disjoint():
    r = [0.8, 0.1, 0.1]
    a = [bucket(f"u{i}", 0, r) for i in range(5000)]
    assert a == [bucket(f"u{i}", 0, r) for i in range(5000)]
    frac = [a.count(k) / 5000 for k in range(3)]
    assert abs(frac[0] - 0.8) < 0.03 and abs(frac[1] - 0.1) < 0.03
