"""Token-level P(Yes) scorer for LLM recommendation panels (vLLM), multi-question, chunked and resumable.

Input rows (`*.jsonl`) — either a same-candidate next-item panel (Lumen `ranking_test.jsonl`:
`positive_item_index`) or a rated panel (`candidate_labels`, `candidate_ratings`):
    user_id, source_event_id, history (list[str]), candidate_item_ids, candidate_titles, candidate_texts, ...

For every (user, candidate, question) the confidence is read from the next-token distribution:
    logit = log P(Yes) - log P(No),  p_yes = sigmoid(logit)
where P(Yes)/P(No) sum the top-k (`--topk_logprobs`, default 50) probabilities of the token ids whose decoded text
is "yes"/"no" (`prompting.yes_no_ids`; both id sets are logged). Column `censored`: 0 both sides in the top-k;
1 one side missing, imputed with the smallest returned logprob (keep the row; the floor bounds each missing id, so
the missing side's sum is within log(n ids of that side) of it, see `prompting.read_yes_no` and report key
`censored1_side_slack_nats`); 2 both missing (logit NaN); 3 prompt longer than the context, not scored (logit NaN).
Prompts are chat-templated and sent as token ids (`prompting.chat_ids`, shared with train_lora_yesno, one BOS).
dtype defaults to float16 (amendment 1, C0). All questions for a candidate share the prompt prefix, so vLLM prefix
caching amortises it.

Records are scored in file order in chunks of `--chunk_users`; each finished chunk is written atomically to
DIR/parts/scores-<5-digit chunk>.csv.gz and skipped on rerun (resume is the default; `--no_resume` starts over).
Parts are merged into DIR/scores.csv.gz; DIR/report.json is written last and marks completion. Resume, and the
"already complete" exit, refuse when the panel bytes, the scoring flags, the rendered prompts (sha1 over every
prompt) or, for resume, the tokenizer/chat template (ids of a probe prompt) differ from the earlier run.

`--swap_k K` additionally scores every unique candidate item under up to K other users' histories with the first
question (user-marginalised item prior π(i)). Donors never hold the item in history_item_ids or
candidate_item_ids (amendment 1, P1.6). Output DIR/swap_prior.csv.gz (chunked over items, parts/swap-*.csv.gz).

    python -m src.confrec.pyes_scorer --data panels/ml1m_rated.jsonl --output runs/ml1m_rated --model <Qwen3-8B> \
        --questions like,dislike,like_para --hist_len 10 --swap_k 8
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from src.confrec.prompting import BODY, QUESTIONS, build_prompt, chat_ids, read_yes_no, yes_no_ids  # noqa: F401
from src.confrec.stats import strict_json

SCORER = "token_pyes_multiq_v2"
SCORE_COLS = ["source_event_id", "user_id", "item_id", "cand_idx", "label", "question",
              "lp_yes", "lp_no", "logit", "yes_no_mass", "censored"]
SWAP_COLS = ["item_id", "donor_user_id", "question", "logit", "censored"]


def labels_of(rec: dict) -> list[int]:
    if "candidate_labels" in rec:
        return [int(x) for x in rec["candidate_labels"]]
    pos = rec["positive_item_index"]
    return [int(i == pos) for i in range(len(rec["candidate_item_ids"]))]


def record_requests(rec: dict, questions: list[str], hist_len: int) -> list[tuple[int, str, str]]:
    """[(cand_idx, question, prompt)] for one panel row; questions inner so a candidate's prompts are adjacent."""
    texts = rec.get("candidate_texts") or [""] * len(rec["candidate_titles"])
    assert len(rec["candidate_titles"]) == len(rec["candidate_item_ids"]) == len(texts), rec.get("source_event_id")
    return [(i, q, build_prompt(rec["history"], title, text, q, hist_len))
            for i, (title, text) in enumerate(zip(rec["candidate_titles"], texts)) for q in questions]


def swap_requests(iids, items: dict, donors: dict, records: list[dict], question: str,
                  hist_len: int) -> list[tuple[object, object, str]]:
    """[(item_id, donor_user_id, prompt)]: item i's (title, text) under each donor record's history."""
    return [(iid, records[j]["user_id"], build_prompt(records[j]["history"], *items[iid], question, hist_len))
            for iid in iids for j in donors[iid]]


def prompts_sha1(prompts) -> str:
    """sha1 over prompt strings in order (fingerprint of the prompt code as applied to this panel)."""
    h = hashlib.sha1()
    for p in prompts:
        h.update(p.encode("utf-8") + b"\x1e")
    return h.hexdigest()


def score_ids(llm, sp, id_lists: list[list[int]], max_model_len: int, yes_ids, no_ids, to_prompt,
              gen_kw: dict | None = None) -> list[tuple[float, float, float, int]]:
    """read_yes_no tuple per prompt. Prompts with len(ids) > max_model_len - 1 are not sent (censored 3)."""
    out = [(math.nan, math.nan, 0.0, 3)] * len(id_lists)
    send = [j for j, ids in enumerate(id_lists) if len(ids) <= max_model_len - 1]
    if send:
        res = llm.generate([to_prompt(id_lists[j]) for j in send], sp, use_tqdm=False, **(gen_kw or {}))
        assert len(res) == len(send), (len(res), len(send))
        for j, o in zip(send, res):
            out[j] = read_yes_no(o.outputs[0].logprobs[0], yes_ids, no_ids)
    return out


def _f(x: float) -> str:
    return f"{x:.6f}" if math.isfinite(x) else "nan"


# ---------------------------------------------------------------- swap-prior donors
def item_holders(records: list[dict]) -> dict[str, set[str]]:
    """item id -> users holding it in history_item_ids (when present) or candidate_item_ids."""
    h: dict[str, set[str]] = {}
    for r in records:
        u = str(r["user_id"])
        for i in list(r.get("history_item_ids") or []) + list(r["candidate_item_ids"]):
            h.setdefault(str(i), set()).add(u)
    return h


def pick_donors(iid, records: list[dict], holders: dict, k: int, seed: int = 0) -> list[int]:
    """Indices of up to k records of distinct users, none of whom holds `iid` (history or candidates).
    Records are drawn uniformly without replacement from random.Random(f"{seed}:{iid}") (deterministic)."""
    bad = holders.get(str(iid), set())
    rng = random.Random(f"{seed}:{iid}")
    n, out, users, tried = len(records), [], set(), set()
    while len(out) < k and len(tried) < n:
        j = rng.randrange(n)
        if j in tried:
            continue
        tried.add(j)
        u = str(records[j]["user_id"])
        if u not in bad and u not in users:
            users.add(u)
            out.append(j)
    return out


def donor_stats(donors: dict, holders: dict, k: int) -> dict:
    n = [len(v) for v in donors.values()] or [0]
    ex = [len(holders.get(str(i), ())) for i in donors] or [0]
    return {"donors_mean": sum(n) / len(n), "donors_min": min(n), "items_fewer_than_k": sum(x < k for x in n),
            "excluded_users_mean": sum(ex) / len(ex), "excluded_users_max": max(ex)}


# ---------------------------------------------------------------- chunk / resume bookkeeping (vLLM-free)
def chunk_bounds(n: int, size: int) -> list[tuple[int, int]]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [(s, min(n, s + size)) for s in range(0, n, size)]


def part_path(parts_dir, kind: str, k: int) -> Path:
    return Path(parts_dir) / f"{kind}-{k:05d}.csv.gz"


def pending_chunks(parts_dir, kind: str, n_chunks: int) -> list[int]:
    return [k for k in range(n_chunks) if not part_path(parts_dir, kind, k).exists()]


def write_csv_gz(path, header: list[str], rows) -> None:
    """Atomic write: <path>.tmp then os.replace, so a killed run never leaves a truncated part."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def merge_parts(paths, out_path) -> dict:
    """Concatenate parts in the given order under one header (atomic). Returns n_rows, counts per censored
    code and the yes_no_mass sum (0 when the column is absent)."""
    if not paths:
        raise ValueError("no parts to merge")
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    header, n, cens, mass = None, 0, {str(c): 0 for c in range(4)}, 0.0
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        for p in paths:
            with gzip.open(p, "rt", newline="", encoding="utf-8") as fi:
                r = csv.reader(fi)
                h = next(r)
                if header is None:
                    header = h
                    w.writerow(h)
                    ci, mi = h.index("censored"), (h.index("yes_no_mass") if "yes_no_mass" in h else None)
                elif h != header:
                    raise ValueError(f"{p}: header {h} != {header}")
                for row in r:
                    w.writerow(row)
                    n += 1
                    cens[row[ci]] = cens.get(row[ci], 0) + 1
                    if mi is not None:
                        mass += float(row[mi])
    os.replace(tmp, out_path)
    return {"n_rows": n, "censored": cens, "mass_sum": mass}


def write_json(path, obj) -> None:
    """Atomic write (<path>.tmp then os.replace): a kill mid-write never leaves a truncated sidecar."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def clear_outputs(out_dir) -> None:
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir / "parts", ignore_errors=True)
    for f in ("report.json", "scores.csv.gz", "swap_prior.csv.gz"):
        (out_dir / f).unlink(missing_ok=True)


def config_diff(old: dict, cfg: dict, ignore=()) -> list[str]:
    return sorted(k for k in (set(old) | set(cfg)) - set(ignore) if old.get(k) != cfg.get(k))


def check_resume_config(parts_dir, cfg: dict) -> None:
    """Write parts/config.json, or refuse to resume parts produced under a different configuration."""
    p = Path(parts_dir) / "config.json"
    cfg = json.loads(json.dumps(cfg))
    if p.exists():
        diff = config_diff(json.loads(p.read_text(encoding="utf-8")), cfg)
        if diff:
            raise SystemExit(f"{p}: existing parts were scored with different {diff}; rerun with --no_resume")
    elif any(Path(parts_dir).glob("*.csv.gz")):
        raise SystemExit(f"{parts_dir} holds parts but no config.json; rerun with --no_resume")
    else:
        write_json(p, cfg)


# chunk sizes change the part layout only, so a finished run stays valid when just these differ
CHUNK_KEYS = ("chunk_users", "chunk_items")


def check_finished(rep_path, cfg: dict) -> None:
    """A finished report.json counts as complete only if it was produced with this exact configuration."""
    rep = json.loads(Path(rep_path).read_text(encoding="utf-8"))
    if rep.get("scorer") != SCORER:
        raise SystemExit(f"{rep_path} was written by scorer {rep.get('scorer')!r}, not {SCORER}; "
                         "rerun with --no_resume")
    old = rep.get("config")
    diff = config_diff(old, json.loads(json.dumps(cfg)), CHUNK_KEYS) if isinstance(old, dict) else ["config"]
    if diff:
        raise SystemExit(f"{rep_path} was produced with different {diff}; rerun with --no_resume")


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- vLLM
def load_vllm(args) -> SimpleNamespace:
    import vllm
    from vllm import LLM, SamplingParams
    try:
        from vllm.inputs import TokensPrompt

        def to_prompt(ids):
            return TokensPrompt(prompt_token_ids=ids)
    except ImportError:
        def to_prompt(ids):
            return {"prompt_token_ids": ids}

    lora_kw = dict(enable_lora=True, max_lora_rank=64) if args.lora else {}
    llm = LLM(model=args.model, tokenizer=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, dtype=args.dtype,
              max_logprobs=args.topk_logprobs, seed=args.seed, **lora_kw)
    gen_kw = {}
    if args.lora:
        from vllm.lora.request import LoRARequest
        gen_kw["lora_request"] = LoRARequest("adapter", 1, args.lora)
    return SimpleNamespace(llm=llm, tok=llm.get_tokenizer(), gen_kw=gen_kw, to_prompt=to_prompt,
                           sp=SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.topk_logprobs),
                           version=getattr(vllm, "__version__", "unknown"))


def run(args, load_model=load_vllm) -> dict | None:
    if args.chunk_users < 1 or args.chunk_items < 1:
        raise SystemExit(f"--chunk_users ({args.chunk_users}) and --chunk_items ({args.chunk_items}) must be >= 1")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / "report.json"
    if args.no_resume:
        clear_outputs(out_dir)

    questions = [q for q in args.questions.split(",") if q]
    bad_q = [q for q in questions if q not in QUESTIONS]
    if not questions or bad_q:
        raise SystemExit(f"unknown questions {bad_q}; choose from {list(QUESTIONS)}")
    backbone = args.backbone or Path(args.model).name
    records = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    if args.n_users:
        records = records[: args.n_users]
    if not records:
        raise SystemExit(f"no records in {args.data}")
    data_sha1 = file_sha1(args.data)

    holders, items, item_list, donors = {}, {}, [], {}
    if args.swap_k > 0:
        holders = item_holders(records)
        for rec in records:
            texts = rec.get("candidate_texts") or [""] * len(rec["candidate_titles"])
            for iid, title, text in zip(rec["candidate_item_ids"], rec["candidate_titles"], texts):
                items.setdefault(iid, (title, text))
        item_list = list(items)
        donors = {iid: pick_donors(iid, records, holders, args.swap_k, args.seed) for iid in item_list}
    cfg = json.loads(json.dumps(dict(
        data_sha1=data_sha1, n_records=len(records), n_users=args.n_users, model=args.model, lora=args.lora,
        questions=questions, hist_len=args.hist_len, swap_k=args.swap_k, seed=args.seed, dtype=args.dtype,
        topk_logprobs=args.topk_logprobs, max_model_len=args.max_model_len, chunk_users=args.chunk_users,
        chunk_items=args.chunk_items, scorer=SCORER,
        prompts_sha1=prompts_sha1(p for rec in records for *_, p in record_requests(rec, questions, args.hist_len)),
        swap_prompts_sha1=(prompts_sha1(f"{iid}\x1f{u}\x1f{p}" for iid, u, p in swap_requests(
            item_list, items, donors, records, questions[0], args.hist_len)) if args.swap_k > 0 else None))))
    if rep_path.exists():
        check_finished(rep_path, cfg)
        print(f"already complete: {rep_path}")
        return None

    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    check_resume_config(parts, cfg)

    main_chunks = chunk_bounds(len(records), args.chunk_users)
    todo_main = pending_chunks(parts, "scores", len(main_chunks))
    todo_swap, swap_chunks = [], []
    if args.swap_k > 0:
        swap_chunks = chunk_bounds(len(item_list), args.chunk_items)
        todo_swap = pending_chunks(parts, "swap", len(swap_chunks))

    meta_path = parts / "run_meta.json"
    t_gen, n_sent = 0.0, 0
    if todo_main or todo_swap or not meta_path.exists():
        m = load_model(args)
        yes_ids, no_ids = yes_no_ids(m.tok)
        if not yes_ids or not no_ids:
            raise SystemExit(f"tokenizer has no yes ids ({sorted(yes_ids)}) or no ids ({sorted(no_ids)})")
        print(f"yes ids {sorted(yes_ids)}  no ids {sorted(no_ids)}  vllm {m.version}", flush=True)
        # ids of one real prompt: catches a tokenizer / chat-template change behind the same --model path
        probe = next((p for rec in records for *_, p in record_requests(rec, questions, args.hist_len)),
                     build_prompt([], "", "", questions[0], 0))
        meta = {"yes_ids": sorted(yes_ids), "no_ids": sorted(no_ids), "vllm_versions": [m.version],
                "probe_ids_sha1": hashlib.sha1(json.dumps(chat_ids(m.tok, probe)).encode()).hexdigest()}
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            diff = [k for k in ("yes_ids", "no_ids", "probe_ids_sha1") if old.get(k) != meta[k]]
            if diff:
                raise SystemExit(f"{meta_path}: {diff} differ from the existing parts (tokenizer or chat template "
                                 "changed); rerun with --no_resume")
            meta["vllm_versions"] = sorted(set(old["vllm_versions"]) | {m.version})
        write_json(meta_path, meta)

        def score(id_lists):
            nonlocal t_gen, n_sent
            t = time.time()
            res = score_ids(m.llm, m.sp, id_lists, args.max_model_len, yes_ids, no_ids, m.to_prompt, m.gen_kw)
            t_gen += time.time() - t
            n_sent += sum(c != 3 for *_, c in res)
            return res

        for k in todo_main:
            s, e = main_chunks[k]
            req, id_lists = [], []
            for rec in records[s:e]:
                labels = labels_of(rec)
                for i, q, p in record_requests(rec, questions, args.hist_len):
                    req.append((rec, i, labels[i], q))
                    id_lists.append(chat_ids(m.tok, p))
            rows = [[rec.get("source_event_id", rec["user_id"]), rec["user_id"], rec["candidate_item_ids"][i], i,
                     lab, q, _f(lpy), _f(lpn), _f(lpy - lpn), _f(mass), c]
                    for (rec, i, lab, q), (lpy, lpn, mass, c) in zip(req, score(id_lists))]
            write_csv_gz(part_path(parts, "scores", k), SCORE_COLS, rows)
            print(f"[scores chunk {k + 1}/{len(main_chunks)}, users {e}/{len(records)}] "
                  f"{n_sent / max(t_gen, 1e-9):.0f} prompts/s", flush=True)

        for k in todo_swap:
            s, e = swap_chunks[k]
            req = swap_requests(item_list[s:e], items, donors, records, questions[0], args.hist_len)
            rows = [[iid, donor, questions[0], _f(lpy - lpn), c]
                    for (iid, donor, _), (lpy, lpn, _, c) in zip(req, score([chat_ids(m.tok, p) for *_, p in req]))]
            write_csv_gz(part_path(parts, "swap", k), SWAP_COLS, rows)
            print(f"[swap chunk {k + 1}/{len(swap_chunks)}, items {e}/{len(item_list)}] "
                  f"{n_sent / max(t_gen, 1e-9):.0f} prompts/s", flush=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    main = merge_parts([part_path(parts, "scores", k) for k in range(len(main_chunks))], out_dir / "scores.csv.gz")
    swap_info = {}
    if args.swap_k > 0:
        sw = merge_parts([part_path(parts, "swap", k) for k in range(len(swap_chunks))],
                         out_dir / "swap_prior.csv.gz")
        swap_info = dict(swap_k=args.swap_k, swap_items=len(item_list), swap_prompts=sw["n_rows"],
                         censored_swap=sw["censored"], swap_n_overlength=sw["censored"]["3"],
                         chunk_items=args.chunk_items, swap_chunks=len(swap_chunks),
                         swap_chunks_resumed=len(swap_chunks) - len(todo_swap),
                         **{f"swap_{k}": v for k, v in donor_stats(donors, holders, args.swap_k).items()})

    report = dict(n_users=len(records), n_prompts=main["n_rows"] + swap_info.get("swap_prompts", 0),
                  n_main_prompts=main["n_rows"], censored_main=main["censored"], n_overlength=main["censored"]["3"],
                  mean_yes_no_mass=main["mass_sum"] / max(1, main["n_rows"]),
                  inference_time_s=t_gen, prompts_sent_this_run=n_sent,
                  prompts_per_s=n_sent / t_gen if t_gen > 0 else float("nan"),
                  chunk_users=args.chunk_users, chunks=len(main_chunks),
                  chunks_resumed=len(main_chunks) - len(todo_main),
                  data_path=args.data, data_sha1=data_sha1, model=args.model, backbone=backbone,
                  questions=questions, hist_len=args.hist_len, seed=args.seed, lora=args.lora, dtype=args.dtype,
                  topk_logprobs=args.topk_logprobs, max_model_len=args.max_model_len,
                  vllm_version=",".join(meta["vllm_versions"]), yes_ids=meta["yes_ids"], no_ids=meta["no_ids"],
                  censored1_side_slack_nats={"yes": math.log(len(meta["yes_ids"])),
                                             "no": math.log(len(meta["no_ids"]))},
                  probe_ids_sha1=meta.get("probe_ids_sha1"), scorer=SCORER, **swap_info, config=cfg)
    report = strict_json(report)
    write_json(rep_path, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv=None) -> argparse.Namespace:
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
    ap.add_argument("--dtype", default="float16", help="float16 (amendment 1 C0); bf16 coarsens logits")
    ap.add_argument("--topk_logprobs", type=int, default=50)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=3072)
    ap.add_argument("--chunk_users", type=int, default=100, help="records per resumable part")
    ap.add_argument("--chunk_items", type=int, default=1000, help="unique items per swap-prior part")
    ap.add_argument("--no_resume", action="store_true", help="discard parts/report in --output and start over")
    ap.add_argument("--lora", default=None, help="PEFT adapter dir from train_lora_yesno (scored via vLLM LoRA)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
