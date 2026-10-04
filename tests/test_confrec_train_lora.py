import json
import re
import sys
import types
import zlib
from types import SimpleNamespace

import pytest
import torch

from src.confrec import pyes_scorer as ps
from src.confrec import train_lora_yesno as tl
from src.confrec.prompting import GATE_VARIANTS, THRESHOLD_QUESTIONS, VARIANTS, chat_ids
from src.confrec.pyes_scorer import record_requests
from src.confrec.split_panel import bucket
from src.confrec.train_lora_yesno import YesNoSet, check_training_panel, collate


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


# ---------------------------------------------------------------- amendment 2: prompt variants (G9 trains on V*)
class SysTok(FakeTok):
    """Renders every chat message with its role (FakeTok keeps only msg[0]), so a system message changes the ids."""
    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        body = "".join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>" for m in msg)
        return f"{self.BOS}{body}<|start_header_id|>assistant<|end_header_id|>\n\n"


def _rated_rows(n_hist=12):
    rows = []
    for u in range(3):
        hist = [(f"Film {u}-{k}", (k * 3 + u) % 5 + 1) for k in range(n_hist)]
        rows.append({"user_id": f"u{u}", "source_event_id": f"u{u}::1", "source": "ml1m", "domain_kind": "movie",
                     "history": [f"{t} (rated {r}/5)" for t, r in hist], "history_ratings": [float(r) for _, r in hist],
                     "history_item_ids": [f"f{u}-{k}" for k in range(n_hist)],
                     "history_meta": [f"Genres: G{k}" for k in range(n_hist)],
                     "candidate_item_ids": [f"x{u}", f"y{u}", f"z{u}"], "candidate_titles": ["X", "Y", "Z"],
                     "candidate_texts": ["Genres: Drama", "", "Genres: Comedy"], "candidate_labels": [1, 0, 1]})
    return rows


@pytest.mark.parametrize("variant", GATE_VARIANTS)
def test_trainer_ids_equal_scorer_ids_per_variant(variant):
    tok, rows = SysTok(), _rated_rows()
    yes, no = tok.vocab["Yes"], tok.vocab["No"]
    for mode, qs in (("standard", ["like"]), ("mirror", ["like", "dislike"])):
        ds = YesNoSet(rows, tok, mode, None, 4096, variant)
        reqs = [(r, i, q, p) for r in rows for i, q, p in record_requests(r, qs, None, variant)]
        assert [ids[:n] for ids, n in ds.items] == [chat_ids(tok, p) for *_, p in reqs]
        want = [(yes if r["candidate_labels"][i] else no) if q == "like" else (no if r["candidate_labels"][i] else yes)
                for r, i, q, _ in reqs]
        assert [ids[n] for ids, n in ds.items] == want
        if VARIANTS[variant]["rated"].system is not None:   # V5 / V7 train with the system message
            assert all(ids[:n] != chat_ids(tok, str(p)) for (ids, n), (*_, p) in zip(ds.items, reqs))
    v0 = YesNoSet(rows, tok, "standard", None, 4096, "V0")     # every other variant trains on different ids
    assert (v0.items == YesNoSet(rows, tok, "standard", None, 4096, variant).items) == (variant == "V0")


def test_threshold_variants_train_the_threshold_dislike_question():
    tok, row = SysTok(), _rated_rows()[0]
    for variant in ("V1", "V7"):
        p = [p for _, q, p in record_requests(row, ["dislike"], None, variant)][0]
        assert THRESHOLD_QUESTIONS["dislike"] in p
        ds = YesNoSet([row], tok, "mirror", None, 4096, variant)
        assert ds.items[1][0][:ds.items[1][1]] == chat_ids(tok, p)
    with pytest.raises(ValueError, match="not a yes/no gate variant"):
        YesNoSet([row], tok, "standard", None, 4096, "T0_probe")
    with pytest.raises(ValueError, match="unregistered"):
        YesNoSet([row], tok, "standard", 10, 4096, "V3")       # V3 registers 20 events
    assert len(YesNoSet([row], tok, "standard", 0, 4096, "V3")) == 3   # 0 = no-history arm is allowed


class _CaptureLLM:
    def __init__(self, tok):
        self.tok, self.sent = tok, []

    def generate(self, prompts, sp, use_tqdm=False, **kw):
        self.sent += [p["prompt_token_ids"] for p in prompts]
        top = {self.tok.vocab["Yes"]: SimpleNamespace(logprob=-0.5), self.tok.vocab["No"]: SimpleNamespace(logprob=-1.)}
        return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]) for _ in prompts]


@pytest.mark.parametrize("variant", ["V0", "V5", "V7"])
def test_scorer_sends_exactly_the_trained_ids(tmp_path, variant):
    tok, rows = SysTok(), _rated_rows()
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    llm = _CaptureLLM(tok)

    def load(args):
        return SimpleNamespace(llm=llm, tok=tok, sp=None, gen_kw={}, version="fake",
                               to_prompt=lambda ids: {"prompt_token_ids": ids})
    ps.run(ps.parse_args(["--data", str(data), "--output", str(tmp_path / "o"), "--model", "m", "--variant", variant,
                          "--questions", "like,dislike"]), load)
    ds = YesNoSet(rows, tok, "mirror", None, 4096, variant)
    assert llm.sent == [ids[:n] for ids, n in ds.items]


# ---------------------------------------------------------------- the scorer's panel rules, before any model load
def test_check_training_panel_applies_the_scorer_rules():
    assert check_training_panel(_rated_rows(12), "V7", None) == {"panel_kind": "rated", "hist_len_used": 20,
                                                                 "max_history_len_in_panel": 12}
    assert check_training_panel(_rated_rows(10), "V0", None)["hist_len_used"] == 10
    assert check_training_panel(_rated_rows(10), "V3", 0)["hist_len_used"] == 0      # no-history arm
    for v in ("V3", "V7"):   # a 10-event panel would train the 10-event rendering; CONFIRM renders 20
        with pytest.raises(SystemExit, match="--hist_len 20"):
            check_training_panel(_rated_rows(10), v, None)
    with pytest.raises(SystemExit, match="unregistered"):
        check_training_panel(_rated_rows(12), "V3", 10)
    nxt = _rows()                                                    # no history_ratings: a next-item panel
    assert check_training_panel(nxt, "V0", 10)["panel_kind"] == "next_item"
    with pytest.raises(SystemExit, match="frozen at V0"):
        check_training_panel(nxt, "V5", None)
    with pytest.raises(SystemExit, match="one panel kind"):
        check_training_panel(_rated_rows(12) + nxt, "V0", None)
    with pytest.raises(SystemExit, match="no rows"):
        check_training_panel([], "V0", None)


class TrainTok(SysTok):
    pad_token_id = 0

    def save_pretrained(self, out):
        pass


def _fake_training_stack(monkeypatch):
    """transformers / peft stand-ins: main() runs end to end on CPU without a model."""
    seen = {"tokenizer_loads": 0, "trained": 0}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            seen["tokenizer_loads"] += 1
            return TrainTok()

    class Model:
        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

        def save_pretrained(self, out):
            seen["saved"] = out

    class Trainer:
        def __init__(self, model, args, train_dataset, data_collator):
            self.batch = data_collator([train_dataset[0], train_dataset[1]])

        def train(self):
            seen["trained"] += 1

    tf = types.ModuleType("transformers")
    tf.AutoTokenizer, tf.Trainer = AutoTokenizer, Trainer
    tf.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda path, **kw: Model())
    tf.TrainingArguments, tf.set_seed = (lambda **kw: SimpleNamespace(**kw)), (lambda s: None)
    peft = types.ModuleType("peft")
    peft.LoraConfig, peft.get_peft_model = (lambda **kw: SimpleNamespace(**kw)), (lambda model, cfg: model)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return seen


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_main_records_the_window_and_panel_history_and_the_scorer_enforces_them(tmp_path, monkeypatch):
    seen = _fake_training_stack(monkeypatch)
    h12 = _write(tmp_path / "h12.jsonl", _rated_rows(12))
    out = tmp_path / "v3"
    tl.main(["--train", h12, "--model", "m", "--out", str(out), "--mode", "mirror", "--variant", "V3"])
    rep = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
    assert (rep["variant"], rep["hist_len"], rep["panel_kind"], rep["max_history_len_in_panel"]) == ("V3", 20, "rated",
                                                                                                     12)
    assert (cfg["variant"], cfg["hist_len_used"], cfg["max_history_len_in_panel"]) == ("V3", 20, 12)
    assert seen["trained"] == 1 and rep["n_examples"] == 18
    ps.check_lora_variant(out, "V3", 20)
    ps.check_lora_variant(out, "V3", 0)                              # the no-history arm scores any adapter
    with pytest.raises(SystemExit, match="trained with prompt variant V3"):
        ps.check_lora_variant(out, "V0", 10)
    out0 = tmp_path / "v0"
    tl.main(["--train", h12, "--model", "m", "--out", str(out0), "--variant", "V0"])
    assert json.loads((out0 / "train_config.json").read_text(encoding="utf-8"))["hist_len_used"] == 10
    with pytest.raises(SystemExit, match="trained on the last 10 history events, scored with 5"):
        ps.check_lora_variant(out0, "V0", 5)


def test_main_refuses_a_short_panel_before_loading_anything(tmp_path, monkeypatch):
    seen = _fake_training_stack(monkeypatch)
    h10 = _write(tmp_path / "h10.jsonl", _rated_rows(10))
    for v in ("V3", "V7"):
        with pytest.raises(SystemExit, match="--hist_len 20"):
            tl.main(["--train", h10, "--model", "m", "--out", str(tmp_path / "o"), "--variant", v])
    nxt = _write(tmp_path / "next.jsonl", _rows())
    with pytest.raises(SystemExit, match="frozen at V0"):
        tl.main(["--train", nxt, "--model", "m", "--out", str(tmp_path / "o"), "--variant", "V5"])
    assert seen["tokenizer_loads"] == 0 and not (tmp_path / "o").exists()
