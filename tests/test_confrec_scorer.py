"""pyes_scorer / prompting without vLLM: prompt text, token ids, Yes/No reading, donors, chunked resume."""
import csv
import gzip
import json
import math
import re
import sys
import types
import zlib
from collections import namedtuple
from types import SimpleNamespace

import pytest

from src.confrec import pyes_scorer as ps
from src.confrec.prompting import BODY, QUESTIONS, build_prompt, chat_ids, read_yes_no, yes_no_ids

LP = namedtuple("LP", "logprob")


class FakeLlamaTok:
    """Llama-3-like: the chat template text starts with the BOS string and __call__ prepends a BOS id only when
    add_special_tokens=True (the HF default, used by vLLM for text prompts). Word ids are hash-stable, so two
    instances agree regardless of the order in which text is tokenized."""
    BOS = "<|begin_of_text|>"
    FIXED = (BOS, "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
             "Yes", "No", " yes", "YES", " No", "Nope", "Yesterday")

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.FIXED)}
        self.inv = dict(enumerate(self.FIXED))

    def __len__(self):
        return 1000 + 20000

    def _word(self, w):
        if w in self.vocab:
            return self.vocab[w]
        i = 1000 + zlib.crc32(w.encode()) % 20000
        self.inv[i] = f"<w{i}>"
        return i

    def decode(self, ids):
        return "".join(self.inv.get(i, f"<w{i}>") for i in ids)

    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        return (f"{self.BOS}<|start_header_id|>user<|end_header_id|>\n\n{msg[0]['content']}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab[self.BOS]] if add_special_tokens else []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [self.vocab[piece]] if piece in self.FIXED else [self._word(w) for w in piece.split()]
        return {"input_ids": ids}


class NoThinkingKwTok(FakeLlamaTok):
    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True):
        return super().apply_chat_template(msg, tokenize, add_generation_prompt)


# verbatim copy of the pre-amendment pyes_scorer prompt (reference for byte identity)
OLD_QUESTIONS = {
    "next": "Will this user purchase the candidate item next?",
    "like": "Would this user like the candidate item?",
    "dislike": "Would this user dislike the candidate item?",
    "like_para": "Is the candidate item a good match for this user's taste?",
    "dislike_para": "Would this user be disappointed by the candidate item?",
}
OLD_BODY = ("You are an expert recommendation system.\n\n"
            "User history (oldest to newest):\n{hist}\n\n"
            "Candidate item:\nTitle: {title}{desc}\n\n"
            "{question} Answer with only Yes or No.")


def old_build_prompt(history, title, text, question, hist_len):
    hist = history[-hist_len:] if hist_len > 0 else []
    hist_block = "\n".join(f"- {h}" for h in hist) if hist else "- (no history available)"
    meta = text[:200] if text else ""
    desc = f"\nDescription: {meta}" if meta else ""
    return OLD_BODY.format(hist=hist_block, title=title, desc=desc, question=OLD_QUESTIONS[question])


def test_build_prompt_identical_to_previous_for_uncapped_inputs():
    assert QUESTIONS == OLD_QUESTIONS and BODY == OLD_BODY and ps.QUESTIONS is QUESTIONS
    t199 = "x" * 199
    cases = [
        ([f"Movie {k} (rated {k % 5 + 1}/5)" for k in range(15)], "Toy Story (1995)", "", 10),
        (["Plain title A", "Plain title B {braces}"], "Lego set", "d" * 450, 10),
        ([t199 + " (rated 3/5)", t199 + "y"], "z" * 200, "short desc", 5),
        ([], "Café ünïcode — title", "désc", 10),
        (["h1", "h2"], "c", None, 0),
    ]
    for hist, title, text, hl in cases:
        for q in OLD_QUESTIONS:
            assert build_prompt(hist, title, text, q, hl) == old_build_prompt(hist, title, text, q, hl)


def test_build_prompt_caps_titles_keeps_rating_suffix():
    p = build_prompt(["A" * 300 + " (rated 4/5)", "B" * 250, "short (rated 2/5)"], "T" * 260, "d" * 300, "like", 10)
    assert "- " + "A" * 200 + " (rated 4/5)\n" in p and "A" * 201 not in p
    assert "- " + "B" * 200 + "\n" in p and "B" * 201 not in p
    assert "- short (rated 2/5)\n\nCandidate item" in p
    assert "Title: " + "T" * 200 + "\nDescription: " + "d" * 200 + "\n\n" in p and "T" * 201 not in p


def test_chat_ids_single_bos_and_thinking_fallback():
    tok = FakeLlamaTok()
    bos = tok.vocab[tok.BOS]
    p = build_prompt(["Heat (rated 5/5)"], "Ronin", "a thriller", "like", 10)
    ids = chat_ids(tok, p)
    assert ids[0] == bos and ids.count(bos) == 1
    text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    assert tok(text)["input_ids"][:2] == [bos, bos]  # the old text-prompt path doubled BOS
    assert chat_ids(NoThinkingKwTok(), p) == ids


def test_yes_no_ids_by_decoded_text():
    tok = FakeLlamaTok()
    yes, no = yes_no_ids(tok)
    assert yes == {tok.vocab[t] for t in ("Yes", " yes", "YES")}
    assert no == {tok.vocab[t] for t in ("No", " No")}


def test_read_yes_no_codes():
    Y, y2, N, O = 1, 2, 3, 9
    lp_y, lp_n, mass, c = read_yes_no({Y: LP(math.log(.6)), y2: LP(math.log(.1)), N: LP(math.log(.2)),
                                       O: LP(math.log(.05))}, {Y, y2}, {N})
    assert c == 0 and lp_y == pytest.approx(math.log(.7)) and lp_n == pytest.approx(math.log(.2))
    assert mass == pytest.approx(.9)
    lp_y, lp_n, mass, c = read_yes_no({Y: LP(math.log(.9)), O: LP(math.log(.01)), 7: LP(-math.inf)}, {Y}, {N})
    assert c == 1 and lp_n == pytest.approx(math.log(.01)) and mass == pytest.approx(.9)  # finite floor
    lp_y, lp_n, mass, c = read_yes_no({N: LP(math.log(.8)), O: LP(math.log(.02))}, {Y}, {N})
    assert c == 1 and lp_y == pytest.approx(math.log(.02)) and lp_n == pytest.approx(math.log(.8))
    lp_y, lp_n, mass, c = read_yes_no({O: LP(-.1), 8: LP(-3.)}, {Y}, {N})
    assert c == 2 and math.isnan(lp_y) and math.isnan(lp_n) and mass == 0.0


def _recs():
    recs = [{"user_id": f"u{j}", "history": [f"h{j} (rated 4/5)"], "history_item_ids": [f"h{j}"],
             "candidate_item_ids": [f"c{j}"], "candidate_titles": [f"C{j}"]} for j in range(30)]
    recs[0]["candidate_item_ids"] = ["X"]
    recs[1]["history_item_ids"] = ["X"]
    recs.append({"user_id": "u5", "history": [], "candidate_item_ids": ["X", "c99"],  # 2nd row of u5, no hist ids
                 "candidate_titles": ["X", "c99"]})
    return recs


def test_pick_donors_excludes_every_holder_and_is_deterministic():
    recs = _recs()
    holders = ps.item_holders(recs)
    assert holders["X"] == {"u0", "u1", "u5"}
    for seed in range(25):
        d = ps.pick_donors("X", recs, holders, 8, seed)
        users = [recs[j]["user_id"] for j in d]
        assert len(d) == 8 and len(set(users)) == 8 and not set(users) & {"u0", "u1", "u5"}
    assert ps.pick_donors("X", recs, holders, 8, 3) == ps.pick_donors("X", recs, holders, 8, 3)
    small = recs[:4]  # u2, u3 eligible only
    assert sorted(recs[j]["user_id"] for j in ps.pick_donors("X", small, ps.item_holders(small), 8, 0)) == ["u2", "u3"]
    st = ps.donor_stats({"X": [0, 1], "c2": [3]}, holders, 2)
    assert st["donors_min"] == 1 and st["items_fewer_than_k"] == 1 and st["excluded_users_max"] == 3


def _read_rows(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_chunk_parts_pending_and_merge(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    assert ps.chunk_bounds(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert ps.pending_chunks(parts, "scores", 3) == [0, 1, 2]
    row = lambda k, c: [f"e{k}", f"u{k}", "i", 0, 1, "like", "-0.1", "-2.0", "1.9", "0.9", c]
    ps.write_csv_gz(ps.part_path(parts, "scores", 1), ps.SCORE_COLS, [row(1, 1), row(1, 0)])
    (parts / "scores-00002.csv.gz.tmp").write_bytes(b"killed mid-write")
    assert ps.pending_chunks(parts, "scores", 3) == [0, 2]
    assert not list(parts.glob("scores-00001*.tmp"))
    ps.write_csv_gz(ps.part_path(parts, "scores", 0), ps.SCORE_COLS, [row(0, 3)])
    ps.write_csv_gz(ps.part_path(parts, "scores", 2), ps.SCORE_COLS, [row(2, 0)])
    info = ps.merge_parts([ps.part_path(parts, "scores", k) for k in range(3)], tmp_path / "scores.csv.gz")
    rows = _read_rows(tmp_path / "scores.csv.gz")
    assert rows[0] == ps.SCORE_COLS and [r[0] for r in rows[1:]] == ["e0", "e1", "e1", "e2"]
    assert info["n_rows"] == 4 and info["censored"] == {"0": 2, "1": 1, "2": 0, "3": 1}
    ps.write_csv_gz(parts / "bad.csv.gz", ps.SWAP_COLS, [])
    with pytest.raises(ValueError):
        ps.merge_parts([ps.part_path(parts, "scores", 0), parts / "bad.csv.gz"], tmp_path / "x.csv.gz")


def test_resume_config_mismatch_refuses(tmp_path):
    ps.check_resume_config(tmp_path, {"questions": ["like"], "hist_len": 10})
    ps.check_resume_config(tmp_path, {"questions": ["like"], "hist_len": 10})
    with pytest.raises(SystemExit):
        ps.check_resume_config(tmp_path, {"questions": ["like", "dislike"], "hist_len": 10})
    other = tmp_path / "other"
    other.mkdir()
    ps.write_csv_gz(other / "scores-00000.csv.gz", ps.SCORE_COLS, [])
    with pytest.raises(SystemExit):
        ps.check_resume_config(other, {"questions": ["like"]})


# ---------------------------------------------------------------- end-to-end with a fake vLLM
class FakeLLM:
    def __init__(self, fail_after=None):
        self.tok, self.calls, self.fail_after, self.sent = FakeLlamaTok(), 0, fail_after, []

    def generate(self, prompts, sp, use_tqdm=False, **kw):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated preemption")
        self.calls += 1
        Y, N = self.tok.vocab["Yes"], self.tok.vocab["No"]
        out = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            self.sent.append(ids)
            h = sum(ids) % 97
            top = {Y: LP(-0.2 - h / 100), N: LP(-1.5 + h / 200), 500: LP(-4.0)}
            if h % 4 == 0:
                top.pop(N)
            out.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[top])]))
        return out


def loader(llm):
    def load(args):
        return SimpleNamespace(llm=llm, tok=llm.tok, sp=None, gen_kw={}, version="fake-1",
                               to_prompt=lambda ids: {"prompt_token_ids": ids})
    return load


def _panel(tmp_path):
    pool = [f"i{k}" for k in range(8)]
    rows = []
    for u in range(6):
        cands = [pool[(u + j) % 8] for j in range(3)]
        hid = pool[(u + 5) % 8]
        hist = ([" ".join(["w"] * 99) + " (rated 4/5)"] * 10) if u == 3 else [f"Item {hid} (rated 4/5)"]
        rows.append({"user_id": f"u{u}", "source_event_id": f"e{u}", "history": hist, "history_item_ids": [hid],
                     "candidate_item_ids": cands, "candidate_titles": [f"Title {c}" for c in cands],
                     "candidate_texts": [f"about {c}" for c in cands], "candidate_labels": [1, 0, 1]})
    p = tmp_path / "panel.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p, rows


def _argv(data, out, *extra):
    return ["--data", str(data), "--output", str(out), "--model", "fake/model", "--questions", "like,dislike",
            "--hist_len", "10", "--swap_k", "2", "--chunk_users", "2", "--chunk_items", "4",
            "--max_model_len", "500", *extra]


def test_cli_defaults_follow_amendment():
    a = ps.parse_args(["--data", "d", "--output", "o", "--model", "m"])
    assert (a.dtype, a.topk_logprobs, a.chunk_users, a.max_model_len, a.no_resume) == ("float16", 50, 100, 3072, False)


def test_run_sends_token_ids_flags_censoring_and_excludes_donors(tmp_path):
    data, rows = _panel(tmp_path)
    llm = FakeLLM()
    rep = ps.run(ps.parse_args(_argv(data, tmp_path / "out")), loader(llm))
    tok, bos = FakeLlamaTok(), FakeLlamaTok().vocab[FakeLlamaTok.BOS]
    expect = [chat_ids(tok, p) for r in rows for _, _, p in ps.record_requests(r, ["like", "dislike"], 10)]
    fits = [ids for ids in expect if len(ids) <= 499]
    assert len(fits) == 5 * 6 and llm.sent[:len(fits)] == fits       # u3's 6 prompts are over-length
    assert all(ids.count(bos) == 1 and ids[0] == bos and len(ids) <= 499 for ids in llm.sent)

    sc = _read_rows(tmp_path / "out" / "scores.csv.gz")
    assert sc[0] == ps.SCORE_COLS and len(sc) - 1 == 36
    by = {c: [r for r in sc[1:] if r[-1] == c] for c in "0123"}
    assert all(r[1] == "u3" and r[8] == "nan" and float(r[9]) == 0.0 for r in by["3"]) and len(by["3"]) == 6
    assert by["1"] and all(float(r[7]) == -4.0 and math.isfinite(float(r[8])) for r in by["1"])  # floor-imputed No
    assert rep["censored_main"] == {c: len(by[c]) for c in "0123"} and rep["n_overlength"] == 6
    assert rep["dtype"] == "float16" and rep["topk_logprobs"] == 50 and rep["vllm_version"] == "fake-1"
    assert rep["yes_ids"] == sorted(yes_no_ids(tok)[0]) and rep["no_ids"] == sorted(yes_no_ids(tok)[1])

    sw = _read_rows(tmp_path / "out" / "swap_prior.csv.gz")
    assert sw[0] == ps.SWAP_COLS
    holders = ps.item_holders(rows)
    per_item = {}
    for iid, donor, q, logit, c in sw[1:]:
        assert donor not in holders[iid] and q == "like"
        per_item.setdefault(iid, []).append(donor)
    assert len(per_item) == 8 and all(len(v) == len(set(v)) == 2 for v in per_item.values())
    assert rep["swap_items"] == 8 and rep["swap_donors_min"] == 2 and rep["swap_items_fewer_than_k"] == 0


def test_run_resumes_after_crash_and_matches_clean_run(tmp_path):
    data, _ = _panel(tmp_path)
    clean = FakeLLM()
    ps.run(ps.parse_args(_argv(data, tmp_path / "clean")), loader(clean))
    assert clean.calls == 3 + 2                                         # 3 user chunks + 2 item chunks

    out = tmp_path / "crash"
    with pytest.raises(RuntimeError):
        ps.run(ps.parse_args(_argv(data, out)), loader(FakeLLM(fail_after=2)))
    assert sorted(p.name for p in (out / "parts").glob("*.csv.gz")) == ["scores-00000.csv.gz", "scores-00001.csv.gz"]
    assert not (out / "report.json").exists()

    resumed = FakeLLM()
    rep = ps.run(ps.parse_args(_argv(data, out)), loader(resumed))
    assert resumed.calls == clean.calls - 2 and rep["chunks_resumed"] == 2
    for f in ("scores.csv.gz", "swap_prior.csv.gz"):
        assert _read_rows(out / f) == _read_rows(tmp_path / "clean" / f)

    def no_model(args):
        raise AssertionError("finished run must not load the model")
    assert ps.run(ps.parse_args(_argv(data, out)), no_model) is None   # report.json present: already complete

    (out / "report.json").unlink()
    with pytest.raises(SystemExit):                                     # parts from another config are not mixed
        ps.run(ps.parse_args(_argv(data, out, "--hist_len", "5")), no_model)
    fresh = FakeLLM()
    ps.run(ps.parse_args(_argv(data, out, "--hist_len", "5", "--no_resume")), loader(fresh))
    assert fresh.calls == clean.calls and json.loads((out / "report.json").read_text())["hist_len"] == 5


def test_report_from_old_scorer_is_not_taken_as_complete(tmp_path):
    data, _ = _panel(tmp_path)
    out = tmp_path / "old"
    out.mkdir()
    for rep in ({"scorer": "token_pyes_multiq"}, {"scorer": ps.SCORER}):   # 2nd: no recorded config
        (out / "report.json").write_text(json.dumps(rep))
        with pytest.raises(SystemExit, match="no_resume"):
            ps.run(ps.parse_args(_argv(data, out)), loader(FakeLLM()))


def _no_model(args):
    raise AssertionError("must not load the model")


def test_finished_run_is_rechecked_against_panel_and_flags(tmp_path):
    data, rows = _panel(tmp_path)
    out = tmp_path / "out"
    ps.run(ps.parse_args(_argv(data, out)), loader(FakeLLM()))
    before = (out / "scores.csv.gz").read_bytes()
    assert ps.run(ps.parse_args(_argv(data, out)), _no_model) is None
    assert ps.run(ps.parse_args(_argv(data, out, "--chunk_users", "4", "--chunk_items", "3")), _no_model) is None
    for extra in (["--questions", "next"], ["--hist_len", "5"], ["--n_users", "4"], ["--swap_k", "0"],
                  ["--seed", "1"], ["--dtype", "bfloat16"], ["--topk_logprobs", "20"], ["--lora", "a"]):
        with pytest.raises(SystemExit, match="no_resume"):
            ps.run(ps.parse_args(_argv(data, out, *extra)), _no_model)
    rows[2]["candidate_labels"] = [0, 0, 1]                               # panel bytes change after the run
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="data_sha1"):
        ps.run(ps.parse_args(_argv(data, out)), _no_model)
    assert (out / "scores.csv.gz").read_bytes() == before


class OtherTemplateTok(FakeLlamaTok):
    def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True, **kw):
        return self.BOS + "<|start_header_id|>system<|end_header_id|>\n\nToday<|eot_id|>" + \
            super().apply_chat_template(msg, tokenize, add_generation_prompt)[len(self.BOS):]


def test_resume_refuses_changed_prompt_code_or_tokenizer(tmp_path, monkeypatch):
    data, _ = _panel(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        ps.run(ps.parse_args(_argv(data, out)), loader(FakeLLM(fail_after=1)))
    assert [p.name for p in (out / "parts").glob("*.csv.gz")] == ["scores-00000.csv.gz"]

    other = FakeLLM()
    other.tok = OtherTemplateTok()                                         # same --model path, new chat template
    with pytest.raises(SystemExit, match="probe_ids_sha1"):
        ps.run(ps.parse_args(_argv(data, out)), loader(other))
    orig = ps.build_prompt                                                 # prompt code edited between crash and resume
    monkeypatch.setattr(ps, "build_prompt", lambda *a, **k: orig(*a, **k).replace("expert", "skilled"))
    with pytest.raises(SystemExit, match="prompts_sha1"):
        ps.run(ps.parse_args(_argv(data, out)), _no_model)
    monkeypatch.setattr(ps, "build_prompt", orig)
    rep = ps.run(ps.parse_args(_argv(data, out)), loader(FakeLLM()))
    assert rep["chunks_resumed"] == 1 and rep["config"]["prompts_sha1"] and rep["probe_ids_sha1"]


def test_chunk_sizes_below_one_are_refused(tmp_path):
    for size in (0, -1):
        with pytest.raises(ValueError):
            ps.chunk_bounds(5, size)
    data, _ = _panel(tmp_path)
    for flag in (["--chunk_users", "0"], ["--chunk_users", "-1"], ["--chunk_items", "0"]):
        with pytest.raises(SystemExit, match="must be >= 1"):
            ps.run(ps.parse_args(_argv(data, tmp_path / "o", *flag)), _no_model)
    assert not (tmp_path / "o" / "report.json").exists()


def test_json_sidecars_are_written_atomically(tmp_path, monkeypatch):
    def killed(src, dst):
        raise OSError("killed before rename")
    monkeypatch.setattr(ps.os, "replace", killed)
    with pytest.raises(OSError):
        ps.check_resume_config(tmp_path, {"hist_len": 10})
    assert not (tmp_path / "config.json").exists()                       # no truncated config.json to trip resume
    monkeypatch.undo()
    ps.check_resume_config(tmp_path, {"hist_len": 10})
    ps.write_json(tmp_path / "run_meta.json", {"yes_ids": [4]})
    assert json.loads((tmp_path / "config.json").read_text()) == {"hist_len": 10}
    assert json.loads((tmp_path / "run_meta.json").read_text()) == {"yes_ids": [4]}


def test_censored1_slack_reported(tmp_path):
    data, _ = _panel(tmp_path)
    rep = ps.run(ps.parse_args(_argv(data, tmp_path / "out")), loader(FakeLLM()))
    assert rep["censored1_side_slack_nats"] == {"yes": pytest.approx(math.log(3)), "no": pytest.approx(math.log(2))}


# ---------------------------------------------------------------- load_vllm wiring against a fake vllm package
def _install_fake_vllm(monkeypatch, tokens_prompt: bool):
    seen = {}

    class TokensPrompt(dict):
        pass

    class SamplingParams:
        def __init__(self, **kw):
            seen["sp"] = kw

    class LoRARequest:
        def __init__(self, *a):
            self.args = a

    class LLM:
        def __init__(self, **kw):
            seen["llm"], self.inner = kw, FakeLLM()

        def get_tokenizer(self):
            return self.inner.tok

        def generate(self, prompts, sp, use_tqdm=False, **kw):
            seen.setdefault("prompt_types", set()).update(type(p) for p in prompts)
            seen["gen_kw"] = kw
            return self.inner.generate(prompts, sp)

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = LLM, SamplingParams, "0.fake"
    inputs = types.ModuleType("vllm.inputs")
    if tokens_prompt:
        inputs.TokensPrompt = TokensPrompt
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = LoRARequest
    for name, mod in {"vllm": vllm, "vllm.inputs": inputs, "vllm.lora": types.ModuleType("vllm.lora"),
                      "vllm.lora.request": request}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return seen, TokensPrompt


@pytest.mark.parametrize("tokens_prompt,lora", [(True, "adapter_dir"), (False, None)])
def test_load_vllm_wiring(tmp_path, monkeypatch, tokens_prompt, lora):
    seen, TP = _install_fake_vllm(monkeypatch, tokens_prompt)
    data, _ = _panel(tmp_path)
    rep = ps.run(ps.parse_args(_argv(data, tmp_path / "out", *(["--lora", lora] if lora else []))))  # default loader
    kw = seen["llm"]
    assert (kw["model"], kw["dtype"], kw["max_logprobs"], kw["enable_prefix_caching"], kw["max_model_len"]) == \
        ("fake/model", "float16", 50, True, 500)
    assert seen["sp"] == {"max_tokens": 1, "temperature": 0.0, "logprobs": 50}
    assert seen["prompt_types"] == ({TP} if tokens_prompt else {dict})
    if lora:
        assert kw["enable_lora"] is True and seen["gen_kw"]["lora_request"].args == ("adapter", 1, lora)
    else:
        assert "enable_lora" not in kw and seen["gen_kw"] == {}
    assert rep["vllm_version"] == "0.fake" and rep["lora"] == lora and rep["n_main_prompts"] == 36
