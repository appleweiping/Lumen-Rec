import torch

from src.confrec.split_panel import bucket
from src.confrec.train_lora_yesno import YesNoSet, collate


class FakeTok:
    """Whitespace tokenizer with a chat template that appends an assistant header."""
    vocab = {}

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return msg[0]["content"] + " <assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]}


def _rows():
    return [{"history": ["A (rated 5/5)"], "candidate_titles": ["X", "Y"], "candidate_texts": ["", ""],
             "candidate_labels": [1, 0]}]


def test_standard_vs_mirror_examples_and_labels():
    tok = FakeTok()
    std = YesNoSet(_rows(), tok, "standard", 10, 512)
    mir = YesNoSet(_rows(), tok, "mirror", 10, 512)
    assert len(std) == 2 and len(mir) == 4
    yes, no = tok("Yes")["input_ids"][0], tok("No")["input_ids"][0]
    answers = [ex["labels"][-1] for ex in (mir[i] for i in range(4))]
    # X liked: like->Yes, dislike->No ; Y disliked: like->No, dislike->Yes
    assert answers == [yes, no, no, yes]
    for i in range(4):  # only the answer token is supervised
        assert sum(t != -100 for t in mir[i]["labels"]) == 1


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
