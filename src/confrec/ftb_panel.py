"""FT-B, FT-S and FT-N: the item-balanced fine-tuning arm, its size-matched control and its falsification arm
(idea-stage/PREREG_AMENDMENT_3_ADDENDUM_13.md; EXPLORATORY: no hypothesis, no Holm family, no claim-admission role of its own).
The panels they train on, the recorded recipe, the control-root plumbing and the equality checks behind scripts/sigir/run_ftb.sh. CPU only,
no torch; every bound module is imported read-only (lazily: importing this module needs neither numpy nor torch) and none is changed.

    python -m src.confrec.ftb_panel build --domain ml1m --train outputs/confrec/ftgrid/panels/ml1m/train.jsonl \
        --split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json --out_dir outputs/confrec/ftgrid_ftb/ftb/ml1m
    python -m src.confrec.ftb_panel verify <the build arguments>
    python -m src.confrec.ftb_panel report [--domains ml1m,toys,games,sports] [--panels_root outputs/confrec/ftgrid/panels]
        [--addendum8 idea-stage/PREREG_AMENDMENT_3_ADDENDUM_8.md] [--out FILE.json]
    python -m src.confrec.ftb_panel recipe --adapters outputs/confrec/gateft/adapters --model M --variant V \
        --train_ref outputs/confrec/gateft/train.jsonl --train_file outputs/confrec/ftgrid/panels/ml1m/train.jsonl --split .../ftgrid_split.json
    python -m src.confrec.ftb_panel scoring --pass like|swap --scores outputs/confrec/ftgrid/scores/ml1m \
        --adapters outputs/confrec/gateft/adapters --data .../eval.jsonl|eval_sd_test.jsonl --model M --variant V
    python -m src.confrec.ftb_panel verify_adapter --adapters outputs/confrec/ftgrid_ftb/adapters/ml1m \
        --ref_adapters outputs/confrec/gateft/adapters --names b0,b1,b2 --panels_dir outputs/confrec/ftgrid_ftb/ftb/ml1m [--write]
    python -m src.confrec.ftb_panel verify_scores --arm b|r|n --pass like|swap --scores outputs/confrec/ftgrid_ftb/scores/b/ml1m \
        --ref_scores outputs/confrec/ftgrid/scores/ml1m --adapters outputs/confrec/ftgrid_ftb/adapters/ml1m --seeds 0,1,2
    python -m src.confrec.ftb_panel link --real_scores outputs/confrec/ftgrid/scores/ml1m \
        --q_scores outputs/confrec/ftgrid_ftb/scores/b/ml1m --models zeroshot [--allow_copy]
    python -m src.confrec.ftb_panel record --pilot_log docs/sigir/PILOT_LOG.md --files scripts/sigir/run_ftb.sh ... [--print]
    python -m src.confrec.ftb_panel info --selection .../selection.json --split .../ftgrid_split.json --gate .../gate_ft.json

The three arms (addendum 13 section 2; ML-1M, Qwen3-8B, the registered Gate-FT TRAIN panel panels/ml1m/train.jsonl, one example = one
candidate of a train row, key "<user_id>::<item_id>", item = item_id):
  FT-B  item-balanced. Every item with at least one positive and one negative TRAIN example keeps m_i = min(n_i+, n_i-) positive and m_i
        negative examples, the ones with the SMALLEST key ftprune.tie_key (the fixed seed-0 key, sha1('tie:0:<key>'), ascending: the
        order of ftprune.removal_mask); items with a single class are dropped. Every kept item has label rate exactly 0.5; the panel has
        K = 2 sum(m_i) examples. Rows keep their order, a row keeps the candidates that were kept (every candidate_* list filtered
        together, history untouched, split_panel.filter_candidates), a row left without a candidate is dropped.
  FT-S  size-matched random control, one panel per adapter seed s = 0, 1, 2: K examples of the full TRAIN panel drawn uniformly without
        replacement by the seed s, as the K examples with the smallest sha1('ftbs:<s>:<key>') (an iid random key per example under the
        seed: a uniform K-subset, the same for every row order and every Python version). Item label rates are preserved in expectation.
  FT-N  falsification arm: the FT-B panel (same examples, rows, order) with the labels permuted within each kept item by a fixed seed-0
        key: the item's 2 m_i kept examples ordered by sha1('ftbn:0:<key>') (its own namespace: the FT-B selection and the tie key
        are correlated, so the permutation must not reuse tie_key), the first m_i get label 1 and the others 0. Each item keeps m_i
        positive and m_i negative labels, attached to its examples at random: the item label rate is 0.5 and the label carries no
        user--item evidence. Only candidate_labels changes (as ftq_panel's teacher does); candidate_ratings is left as it is.
Adapters: b0 b1 b2 (FT-B, train_b.jsonl), r0 r1 r2 (FT-S, train_r<s>.jsonl) and n0 n1 (FT-N, train_n.jsonl); seeds 0-2, 0-2 and 0-1.

build           train_b.jsonl, train_r0-2.jsonl, train_n.jsonl and a manifest per panel (train_<x>.manifest.json: K, the number of kept
                items, the share of TRAIN examples dropped, the label rate overall and per item, the sha1 of the source panel and of the
                built panel, the seeds, the code sha1; strict JSON, no clock, host or path: a rerun is byte-identical). Inputs are
                checked (the registered train.jsonl is the file ftgrid_data writes and has the sha1 ftgrid_split.json records), the
                selection is computed twice (a sort and a numpy lexsort), every panel is re-read and compared with the source field by
                field (only the selection, or candidate_labels for FT-N, differs) and the per-item facts are recomputed on what was written.
verify          recomputes all ten files from the source in memory (seconds of CPU) and requires byte equality with the directory: a panel,
                a manifest, a missing or an unexpected file that differs in any byte is refused (exit 1). The runner calls it before any
                training or scoring; neither a step marker nor a manifest is trusted.
report          K and the multiplicity of the TRAIN panels of the given domains (CPU; it reads the registered panels only and writes no
                adapter): examples, items, the share of examples in items with at least 2 examples and in items with both classes (the
                table of addendum 8), K of FT-B, the kept items and the share dropped. --addendum8 compares the multiplicity columns
                with that table. Amazon panels are not run in addendum 13: this is the K the addendum asks to be reported.
recipe          the trainer flags of the real adapters s0-s2 (ML-1M: Gate-FT's): every key of their train_config.json that records an
                argument except --train, --out and --seed (ftq_panel's recipe; one registered recipe, one epoch). Only --train, --out
                and --seed differ between an FT-B / FT-S / FT-N adapter and the registered ones.
scoring         the scorer flags of the real s0's like or swap pass (run.key and the config of its report.json), --data, --output, --model
                and --lora excepted (ftlen_panel's check of the two registered passes).
verify_adapter  after training: the recorded arguments of each adapter equal the real s0's except --train, --out and --seed, and the
                adapter trained on exactly K examples of its panel; --write records its provenance (ftb.json: arm, seed, the sha1 of
                the panel and of its manifest, K and the number of optimizer steps, derived from n_examples, the micro-batch, the
                accumulation and the epochs by the Trainer's schedule; the trainer saves no trainer_state) beside the registered SFT's
                step count; without --write an existing adapter must carry a matching record.
verify_scores   after scoring: the recorded config of each seed's pass equals the real s0's pass of the same kind except `lora`.
link            scores/<tag>/ml1m/zeroshot becomes a relative symlink to the registered zero-shot scores, never a copy (--allow_copy,
                DRY_RUN only, falls back to a copy where the platform has no symlinks).
record          0 iff the pilot log holds the sha1 of every listed file (addendum 13 section 5: recorded before the first adapter is
                trained); --print writes the lines to paste into the log.
info            decision and gate_ft_prompt of selection.json, the variant of the split and the decision of gate_ft.json.
Exit codes: 0 done; 1 error or an inconsistent record; 2 refused input; 4 refused by a registered gate (record).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from src.confrec import ftlen_panel as fl
from src.confrec import ftq_panel as fq
from src.confrec.ftq_panel import FtqError, file_sha1, json_bytes, read_json, sha1_bytes

SPEC = "idea-stage/PREREG_AMENDMENT_3_ADDENDUM_13.md (FT-B, FT-S, FT-N; exploratory)"
DOMAIN = "ml1m"                              # addendum 13 section 2: ML-1M only
REPORT_DOMAINS = ("ml1m", "toys", "games", "sports")
ARM_TAGS = ("b", "r", "n")
ARM_NAME = {"b": "FT-B", "r": "FT-S", "n": "FT-N"}
ARM_SEEDS = {"b": (0, 1, 2), "r": (0, 1, 2), "n": (0, 1)}
ADAPTERS = tuple(f"{t}{k}" for t in ARM_TAGS for k in ARM_SEEDS[t])        # b0 b1 b2 r0 r1 r2 n0 n1
TIE_SEED = 0                                 # ftprune.TIE_SEED: the fixed seed-0 key of the FT-B selection
S_NAMESPACE, N_NAMESPACE, N_SEED = "ftbs", "ftbn", 0
MANIFEST_FORMAT = "ftb_panel_v1"
PROVENANCE_NAME = "ftb.json"
PASSES = ("like", "swap")
LINKED_MODELS = ("zeroshot",)
SWAP_K = fl.SWAP_K
PANELS = (("b", None), ("r", 0), ("r", 1), ("r", 2), ("n", None))           # (arm tag, seed of an FT-S panel)
COMMANDS = ("build", "verify", "report", "recipe", "scoring", "verify_adapter", "verify_scores", "link", "record", "info")
FtbError = FtqError
ADDENDUM8_LABELS = {"ML-1M": "ml1m", "Toys": "toys", "Video_Games": "games", "Sports": "sports"}


# ---------------------------------------------------------------- names
def panel_name(tag: str, seed: int | None = None) -> str:
    return f"train_{tag}{'' if seed is None else seed}.jsonl"


def manifest_name(tag: str, seed: int | None = None) -> str:
    return f"train_{tag}{'' if seed is None else seed}.manifest.json"


def expected_files() -> list:
    """The ten files of a build, in writing order: each panel, then its manifest."""
    return [n for tag, seed in PANELS for n in (panel_name(tag, seed), manifest_name(tag, seed))]


def split_adapter(name: str) -> tuple:
    """('b', 2) from 'b2'; refuses a name that is not one of the eight adapters."""
    if name not in ADAPTERS:
        raise FtbError(f"unknown adapter {name!r}: the adapters of addendum 13 are {', '.join(ADAPTERS)}", 2)
    return name[0], int(name[1:])


def adapter_panel(name: str) -> tuple:
    """(panel file name, manifest file name) of the panel an adapter trains on."""
    tag, seed = split_adapter(name)
    return (panel_name(tag, seed if tag == "r" else None), manifest_name(tag, seed if tag == "r" else None))


def _ftprune():
    from src.confrec import ftprune
    return ftprune


# ---------------------------------------------------------------- the selections (pure functions of the examples' keys and labels)
def item_counts(item, y, mask=None) -> dict:
    """{item: [n_negative, n_positive]} over the examples (those with mask[j] when a mask is given), items in first-seen order."""
    out: dict = {}
    for j, it in enumerate(item):
        if mask is None or mask[j]:
            out.setdefault(it, [0, 0])[int(y[j])] += 1
    return out


def select_balanced(item, y, tie) -> list:
    """FT-B: keep flags. For every item with at least one example of each class, the min(n+, n-) examples of each class with the
    smallest tie key (ascending, ties by position: the order of ftprune.removal_mask); an item with one class keeps nothing. A pure
    function of (item, label, tie key) of each example."""
    if not (len(item) == len(y) == len(tie)):
        raise FtbError(f"{len(item)} items, {len(y)} labels and {len(tie)} tie keys")
    groups: dict = {}
    for j, it in enumerate(item):
        groups.setdefault(it, ([], []))[int(y[j])].append(j)
    keep = [False] * len(item)
    for neg, pos in groups.values():
        m = min(len(neg), len(pos))
        if m == 0:
            continue
        for cls in (neg, pos):
            for j in sorted(cls, key=lambda j: tie[j])[:m]:
                keep[j] = True
    return keep


def select_balanced_lexsort(item, y, tie) -> list:
    """The same selection through numpy.lexsort (a second implementation, compared with select_balanced when the panels are built)."""
    import numpy as np
    n = len(item)
    it, yy, tt = np.asarray(item, dtype=str), np.asarray(y, int), np.asarray(tie, dtype=str)
    order = np.lexsort((np.arange(n), tt, yy, it))              # item, then label, then tie key, then position
    keep = [False] * n
    start = 0
    while start < n:
        stop = start
        while stop < n and it[order[stop]] == it[order[start]]:
            stop += 1
        block = order[start:stop]
        labels = yy[block]
        n_neg = int((labels == 0).sum())
        m = min(n_neg, len(block) - n_neg)
        if m:
            for j in block[:n_neg][:m].tolist() + block[n_neg:][:m].tolist():
                keep[j] = True
        start = stop
    return keep


def s_key(seed: int, key: str) -> str:
    """The random key of an FT-S draw: sha1('ftbs:<seed>:<key>')."""
    return _ftprune().hkey(S_NAMESPACE, int(seed), key)


def draw_uniform(keys, k: int, seed: int) -> list:
    """FT-S: keep flags of a uniform K-subset without replacement under the seed: the K examples with the smallest s_key (ties by
    position). A pure function of the example keys, K and the seed (so independent of the row order of the file)."""
    n = len(keys)
    if not 0 <= k <= n:
        raise FtbError(f"cannot draw {k} of {n} examples")
    hk = [s_key(seed, key) for key in keys]
    keep = [False] * n
    for j in sorted(range(n), key=lambda j: hk[j])[:k]:
        keep[j] = True
    return keep


def draw_uniform_lexsort(keys, k: int, seed: int) -> list:
    import numpy as np
    n = len(keys)
    hk = np.asarray([s_key(seed, key) for key in keys], dtype=str)
    keep = np.zeros(n, bool)
    keep[np.lexsort((np.arange(n), hk))[:k]] = True
    return keep.tolist()


def n_key(key: str) -> str:
    """The permutation key of FT-N: sha1('ftbn:0:<key>'), its own namespace (never the FT-B selection's tie key)."""
    return _ftprune().hkey(N_NAMESPACE, N_SEED, key)


def permute_labels(keys, item, keep) -> dict:
    """FT-N: {example index: new label} for the kept examples. Per kept item the 2 m_i kept examples are ordered by n_key (ties by
    position), the first m_i get label 1 and the others 0: m_i positive and m_i negative labels, attached to the examples by a
    fixed seed-0 key. The label of an example is a function of its key and of its item's kept examples only (no label is read)."""
    groups: dict = {}
    for j, it in enumerate(item):
        if keep[j]:
            groups.setdefault(it, []).append(j)
    nk = {j: n_key(keys[j]) for js in groups.values() for j in js}
    out = {}
    for it, js in groups.items():
        if len(js) % 2:
            raise FtbError(f"item {it!r} keeps {len(js)} examples: an odd number cannot be half positive")
        m = len(js) // 2
        for rank, j in enumerate(sorted(js, key=lambda j: nk[j])):
            out[j] = 1 if rank < m else 0
    return out


# ---------------------------------------------------------------- rows
def select_rows(rows: list, ex: dict, keep, labels: dict | None = None) -> list:
    """The panel rows of a selection: every row with its kept candidates (split_panel.filter_candidates: every candidate_* list
    filtered together, the history untouched), a row left without a candidate dropped; with `labels` ({example index: label}) the
    candidate_labels of the kept candidates are replaced by them."""
    from src.confrec.split_panel import filter_candidates
    out = []
    for r, a, b in zip(rows, ex["starts"], ex["starts"][1:]):
        flags = [bool(keep[a + c]) for c in range(b - a)]
        if not any(flags):
            continue
        sub = filter_candidates(r, flags)
        if labels is not None:
            sub = dict(sub, candidate_labels=[int(labels[a + c]) for c, f in enumerate(flags) if f])
        out.append(sub)
    return out


def check_selected_rows(src_rows: list, got_rows: list, ex: dict, keep, labels: dict | None = None) -> dict:
    """The registered "nothing else changed", recomputed without filter_candidates: the panel's rows are the source rows with exactly
    the kept candidates, in order, every candidate_* list restricted to them, every other field (history, ids, source, ...) equal and
    the candidate_labels the source's (or the permuted ones). Raises FtbError; returns the counts."""
    out_rows = [(r, a, b) for r, a, b in zip(src_rows, ex["starts"], ex["starts"][1:]) if any(keep[a:b])]
    if len(out_rows) != len(got_rows):
        raise FtbError(f"the panel has {len(got_rows)} rows, the selection of the source {len(out_rows)}")
    n = 0
    for k, ((r, a, b), t) in enumerate(zip(out_rows, got_rows)):
        pos = [c for c in range(b - a) if keep[a + c]]
        if list(r) != list(t):
            raise FtbError(f"row {k}: the panel row has other keys than the source row")
        for f in r:
            if f.startswith("candidate_") and isinstance(r[f], list):
                want = [r[f][c] for c in pos]
                if f == "candidate_labels" and labels is not None:
                    want = [int(labels[a + c]) for c in pos]
                if t[f] != want:
                    raise FtbError(f"row {k}: {f} is not the selection's")
            elif t[f] != r[f]:
                raise FtbError(f"row {k}: {f} changed (only the selection, and candidate_labels for FT-N, may)")
        n += len(pos)
    return {"rows": len(got_rows), "examples": n}


def source_facts(item, y) -> dict:
    """The multiplicity of a TRAIN panel (the table of addendum 8) and the FT-B quantities that follow from the item counts alone."""
    counts = item_counts(item, y)
    n = len(item)
    repeated = sum(sum(c) for c in counts.values() if sum(c) >= 2)
    both = [c for c in counts.values() if c[0] and c[1]]
    in_both = sum(sum(c) for c in both)
    k = 2 * sum(min(c) for c in both)
    return {"examples": n, "items": len(counts), "positives": sum(int(v) for v in y),
            "examples_in_items_with_2_or_more_examples": repeated, "share_examples_in_items_with_2_or_more_examples": repeated / n,
            "examples_in_items_with_both_classes": in_both, "share_examples_in_items_with_both_classes": in_both / n,
            "K": k, "kept_items": len(both), "dropped_single_class_items": len(counts) - len(both),
            "dropped_examples_single_class_items": n - in_both, "dropped_examples_balancing": in_both - k,
            "dropped_examples": n - k, "share_examples_dropped": (n - k) / n}


# ---------------------------------------------------------------- the build (all in memory)
def _check_train(train, split, domain: str):
    """Rows, line bytes, the sha1 and the split record of the registered TRAIN panel, with every input check."""
    from src.confrec import ftgrid_data as fd
    train = Path(train)
    if not train.is_file():
        raise FtbError(f"{train}: no such file (run_ftgrid.sh stage 0 writes panels/<d>/train.jsonl)", 1)
    lines, rows = fq.read_rows(train)
    if not rows:
        raise FtbError(f"{train} holds no row")
    train_sha1 = file_sha1(train)
    if sha1_bytes(b"".join(lines)) != train_sha1:
        raise FtbError(f"{train} has blank lines: it is not a file ftgrid_data wrote")
    try:
        fd.check_rows(rows, "train.jsonl")
    except SystemExit as e:
        raise FtbError(str(e.code)) from None
    for k, (line, row) in enumerate(zip(lines, rows)):
        if fd.row_bytes(row) != line:
            raise FtbError(f"{train}: row {k} is not in the format ftgrid_data writes (json.dumps(row, ensure_ascii=False) + "
                           "newline): FT-B needs the registered train.jsonl byte for byte")
    split_info = None
    if split is not None:
        sp = read_json(split)
        recorded = (sp.get("files") or {}).get("train.jsonl")
        if recorded != train_sha1:
            raise FtbError(f"{train}: sha1 {train_sha1} is not the one {split} records for train.jsonl ({recorded}): FT-B selects "
                           "from the registered TRAIN file only")
        if sp.get("domain") not in (None, domain):
            raise FtbError(f"{split} is the split of {sp.get('domain')!r}, not {domain!r}")
        split_info = {"ftgrid_split_sha1": file_sha1(split), "domain": sp.get("domain"), "T": sp.get("T"),
                      "variant": sp.get("variant"), "train_sha1_recorded": recorded}
    return train, rows, train_sha1, split_info


def _train_examples(rows: list) -> dict:
    try:
        return _ftprune().train_examples(rows)
    except SystemExit as e:
        raise FtbError(str(e.code)) from None


def panel_facts(parsed_rows: list) -> dict:
    """Per-item facts recomputed on a written panel: rows, examples, items, positives, and the label rate overall and per item."""
    ex = _train_examples(parsed_rows)
    counts = item_counts(ex["item"], ex["y"])
    n = ex["n"]
    pos = sum(int(v) for v in ex["y"])
    rates = [c[1] / (c[0] + c[1]) for c in counts.values()]
    lines = "".join(f"{it}\t{c[0] + c[1]}\t{c[1]}\n" for it, c in sorted(counts.items()))
    return {"rows": len(parsed_rows), "examples": n, "items": len(counts), "positives": pos, "label_rate_overall": pos / n if n else None,
            "per_item_label_rate": {"min": min(rates) if rates else None, "max": max(rates) if rates else None,
                                    "all_exactly_half": bool(counts) and all(2 * c[1] == c[0] + c[1] for c in counts.values())},
            "items_sha1": sha1_bytes(lines.encode("utf-8")), "keys": ex["key"]}


def compute_panels(train, *, domain: str = DOMAIN, split=None) -> tuple:
    """(files, manifests): the ten files of a build as bytes ({name: bytes}, in writing order) and the five manifests as dicts
    ({panel name: manifest}), recomputed from the registered train.jsonl with every input check and every assertion; nothing is
    written (build writes, verify compares)."""
    from src.confrec import ftgrid_data as fd
    from src.confrec import split_panel
    from src.confrec.stats import strict_json
    if domain != DOMAIN:
        raise FtbError(f"--domain {domain!r}: FT-B, FT-S and FT-N are registered for {DOMAIN} only (addendum 13 section 2)", 2)
    ftprune = _ftprune()
    train, rows, train_sha1, split_info = _check_train(train, split, domain)
    ex = _train_examples(rows)
    n, keys, item = ex["n"], ex["key"], ex["item"]
    y = [int(v) for v in ex["y"]]
    tie = [ftprune.tie_key(k) for k in keys]
    src = source_facts(item, y)

    keep_b = select_balanced(item, y, tie)
    if keep_b != select_balanced_lexsort(item, y, tie):
        raise FtbError("the FT-B selection differs between its two implementations (sort and lexsort)")
    K = sum(keep_b)
    if K != src["K"]:
        raise FtbError(f"the FT-B selection kept {K} examples, the item counts say {src['K']}")
    if not K:
        raise FtbError("no TRAIN item has both classes: the FT-B panel would be empty")
    shuffled = list(range(n))[::-1]                          # the selection is a function of (item, label, key), not of the order
    again = select_balanced([item[j] for j in shuffled], [y[j] for j in shuffled], [tie[j] for j in shuffled])
    back = [False] * n
    for i, j in enumerate(shuffled):
        back[j] = again[i]
    if back != keep_b:
        raise FtbError("the FT-B selection depends on the order of the examples, not on their item, label and tie key only")
    keep_s = {}
    for s in ARM_SEEDS["r"]:
        keep_s[s] = draw_uniform(keys, K, s)
        if keep_s[s] != draw_uniform_lexsort(keys, K, s):
            raise FtbError(f"the FT-S draw of seed {s} differs between its two implementations")
        if sum(keep_s[s]) != K:
            raise FtbError(f"the FT-S draw of seed {s} has {sum(keep_s[s])} examples, not K = {K}")
    labels_n = permute_labels(keys, item, keep_b)

    selections = {("b", None): (keep_b, None), ("n", None): (keep_b, labels_n)}
    selections.update({("r", s): (keep_s[s], None) for s in ARM_SEEDS["r"]})
    files, manifests, facts, built_rows = {}, {}, {}, {}
    code = {"ftb_panel.py": file_sha1(__file__), "ftprune.py": file_sha1(ftprune.__file__), "ftgrid_data.py": file_sha1(fd.__file__),
            "split_panel.py": file_sha1(split_panel.__file__)}
    for tag, seed in PANELS:
        keep, labels = selections[(tag, seed)]
        got = select_rows(rows, ex, keep, labels)
        lines = [fd.row_bytes(r) for r in got]
        parsed = [json.loads(b) for b in lines]              # what is written, re-read: every check below runs on it
        counts = check_selected_rows(rows, parsed, ex, keep, labels)
        fa = panel_facts(parsed)
        if counts["examples"] != K or fa["examples"] != K:
            raise FtbError(f"the {ARM_NAME[tag]} panel has {fa['examples']} examples, not K = {K}")
        if tag != "r" and not fa["per_item_label_rate"]["all_exactly_half"]:
            raise FtbError(f"the {ARM_NAME[tag]} panel has an item whose label rate is not exactly 0.5")
        if not set(fa.pop("keys")) <= set(keys):
            raise FtbError(f"the {ARM_NAME[tag]} panel holds an example that is not a TRAIN example")
        facts[(tag, seed)] = fa
        built_rows[(tag, seed)] = (lines, counts)
        files[panel_name(tag, seed)] = b"".join(lines)

    n_changed = sum(1 for j, v in labels_n.items() if v != y[j])
    n_pos_n: dict = {}
    for j, v in labels_n.items():                            # FT-N: exactly m_i positive labels in every kept item
        n_pos_n[item[j]] = n_pos_n.get(item[j], 0) + v
    if n_pos_n != {it: min(c) for it, c in item_counts(item, y).items() if c[0] and c[1]}:
        raise FtbError("FT-N changed the number of positive labels of an item")
    src_rec = {"file": Path(train).name, "sha1": train_sha1, "rows": len(rows), "examples": n, "items": src["items"],
               "positives": src["positives"], "negatives": n - src["positives"]}
    common = {"format": MANIFEST_FORMAT, "spec": SPEC, "domain": domain, "source": src_rec, "K": K,
              "code_sha1": code, "split": split_info,
              "multiplicity": {k: src[k] for k in ("examples_in_items_with_2_or_more_examples",
                                                   "share_examples_in_items_with_2_or_more_examples",
                                                   "examples_in_items_with_both_classes",
                                                   "share_examples_in_items_with_both_classes")}}
    for tag, seed in PANELS:
        fa = facts[(tag, seed)]
        lines, counts = built_rows[(tag, seed)]
        panel_rec = {"file": panel_name(tag, seed), "sha1": sha1_bytes(files[panel_name(tag, seed)]), "rows": counts["rows"],
                     "examples": counts["examples"]}
        m = {**common, "arm": ARM_NAME[tag], "arm_tag": tag}
        if tag == "b":
            m["definition"] = ("every TRAIN item with at least one positive and one negative example keeps m_i = min(n_i+, n_i-) "
                               "positive and m_i negative examples, those with the smallest ftprune.tie_key ascending; items with a "
                               "single class are dropped (addendum 13 section 2)")
            m["selection"] = {"tie_seed": TIE_SEED, "tie_key": f"sha1('tie:{TIE_SEED}:<user_id>::<item_id>')",
                              "order": "ascending, per item and label class (the order of ftprune.removal_mask)"}
            m.update({k: src[k] for k in ("kept_items", "dropped_single_class_items", "dropped_examples_single_class_items",
                                          "dropped_examples_balancing", "dropped_examples", "share_examples_dropped")})
        elif tag == "r":
            m["definition"] = ("K examples of the full TRAIN panel drawn uniformly without replacement by the seed of the adapter: the "
                               "K examples with the smallest sha1('ftbs:<seed>:<user_id>::<item_id>')")
            m["seed"] = seed
            m["selection"] = {"namespace": S_NAMESPACE, "key": f"sha1('{S_NAMESPACE}:<seed>:<user_id>::<item_id>')",
                              "order": "the K smallest keys"}
            m["K_from"] = "FT-B"
            m["share_examples_dropped"] = (n - K) / n
            m["dropped_examples"] = n - K
        else:
            m["definition"] = ("the FT-B panel with its labels permuted within each kept item by a fixed seed-0 key: the item's kept "
                               "examples ordered by sha1('ftbn:0:<user_id>::<item_id>'), the first m_i get label 1, the others 0")
            m["permutation"] = {"namespace": N_NAMESPACE, "seed": N_SEED, "key": f"sha1('{N_NAMESPACE}:{N_SEED}:<user_id>::<item_id>')",
                                "never": "ftprune.tie_key: the FT-B selection is correlated with it",
                                "only_candidate_labels_change": True}
            m["same_examples_as_FT_B"] = True
            m["labels_changed_vs_FT_B"] = n_changed
            m["share_labels_equal_to_FT_B"] = (K - n_changed) / K
            m.update({k: src[k] for k in ("kept_items", "dropped_single_class_items", "dropped_examples_single_class_items",
                                          "dropped_examples_balancing", "dropped_examples", "share_examples_dropped")})
        m["items_in_panel"] = fa["items"]
        m["label_rate"] = {"overall": fa["label_rate_overall"], "per_item": fa["per_item_label_rate"]}
        m["items_sha1"] = fa["items_sha1"]
        m["panel"] = panel_rec
        m["checks"] = {"train_file_is_what_ftgrid_data_writes": True,
                       "train_sha1_is_the_splits": None if split_info is None else True,
                       "two_implementations_agree": True, "selection_is_order_invariant": True,
                       "only_the_selection_differs_from_train" if tag != "n" else "only_candidate_labels_differ_from_ftb": True,
                       "every_kept_item_label_rate_exactly_half": None if tag == "r" else True,
                       "K_equals_ftb": True}
        manifests[panel_name(tag, seed)] = m
        files[manifest_name(tag, seed)] = json_bytes(strict_json(m))
    return {name: files[name] for name in expected_files()}, manifests


def build(train, out_dir, *, domain: str = DOMAIN, split=None) -> dict:
    """The ten files under out_dir; the manifests ({panel name: manifest})."""
    from src.confrec import build_rated_panels as brp
    files, manifests = compute_panels(train, domain=domain, split=split)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged = {name: out / (name + ".tmp") for name in files}
    try:
        for name, data in files.items():
            staged[name].write_bytes(data)
        for name in files:                                               # each panel, then its manifest
            brp.commit(staged.pop(name), out / name)
    finally:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
    for name, data in files.items():
        if file_sha1(out / name) != sha1_bytes(data):
            raise FtbError(f"{out / name} is not the file that was verified")
    return manifests


def verify(train, out_dir, *, domain: str = DOMAIN, split=None) -> dict:
    """The panels on disk are the panels: recompute all ten files from train.jsonl and require byte equality with out_dir (a step
    marker or a manifest alone is no proof: a panel swapped for another file with its manifest re-tagged is refused, and so is a
    file that should not be there). Returns the manifests."""
    files, manifests = compute_panels(train, domain=domain, split=split)
    out = Path(out_dir)
    for name, want in files.items():
        path = out / name
        if not path.is_file():
            raise FtbError(f"{path}: missing (stage 1 writes the FT-B, FT-S and FT-N panels and their manifests)")
        got = path.read_bytes()
        if got != want:
            raise FtbError(f"{path} is not the panel recomputed from {train}: it has {len(got)} bytes (sha1 {sha1_bytes(got)}), the "
                           f"recomputation {len(want)} bytes (sha1 {sha1_bytes(want)}). The file was changed after stage 1 or was "
                           f"built by other code: move {out} aside and rerun stage 1")
    extra = sorted(p.name for p in out.iterdir() if p.name not in files)
    if extra:
        raise FtbError(f"{out} holds files that stage 1 does not write: {extra}")
    return manifests


# ---------------------------------------------------------------- report: K and the multiplicity of the TRAIN panels
def report_domain(train, split=None, domain: str | None = None) -> dict:
    """K and the multiplicity of one registered TRAIN panel (no panel is built and nothing is written)."""
    train = Path(train)
    if not train.is_file():
        raise FtbError(f"{train}: no such file (the registered TRAIN panel of the domain)", 1)
    lines, rows = fq.read_rows(train)
    if not rows:
        raise FtbError(f"{train} holds no row")
    ex = _train_examples(rows)
    facts = source_facts(ex["item"], [int(v) for v in ex["y"]])
    facts = {"domain": domain, "train_sha1": file_sha1(train), "rows": len(rows), **facts}
    if split is not None and Path(split).is_file():
        recorded = (read_json(split).get("files") or {}).get("train.jsonl")
        facts["train_sha1_equals_split"] = (None if recorded is None else recorded == facts["train_sha1"])
    return facts


def parse_addendum8(text: str) -> dict:
    """{domain: {examples, items, share_repeated, share_both}} from the multiplicity table of addendum 8 section 1."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in ADDENDUM8_LABELS:
            try:
                out[ADDENDUM8_LABELS[cells[0]]] = {"examples": int(cells[1].replace(",", "")), "items": int(cells[2].replace(",", "")),
                                                    "share_repeated": float(cells[3].rstrip("%")) / 100,
                                                    "share_both": float(cells[4].rstrip("%")) / 100}
            except ValueError:
                continue
    return out


def compare_addendum8(facts: dict, row: dict) -> dict:
    """Whether a domain's multiplicity equals its row of the addendum-8 table (counts exactly, shares to the table's 0.1%)."""
    return {"examples": facts["examples"] == row["examples"], "items": facts["items"] == row["items"],
            "share_repeated": abs(round(facts["share_examples_in_items_with_2_or_more_examples"] * 1000) / 1000 - row["share_repeated"]) < 1e-9,
            "share_both": abs(round(facts["share_examples_in_items_with_both_classes"] * 1000) / 1000 - row["share_both"]) < 1e-9}


def format_report(results: dict, table8: dict | None = None) -> str:
    head = ("domain", "examples", "items", "in items >= 2", "in both classes", "K (FT-B)", "kept items", "share dropped")
    lines = ["  ".join(f"{h:>15}" for h in head)]
    for d, f in results.items():
        cells = (d, f["examples"], f["items"], f"{f['share_examples_in_items_with_2_or_more_examples']:.1%}",
                 f"{f['share_examples_in_items_with_both_classes']:.1%}", f["K"], f["kept_items"], f"{f['share_examples_dropped']:.1%}")
        lines.append("  ".join(f"{c:>15}" for c in cells))
    if table8 is not None:
        for d, f in results.items():
            cmp = compare_addendum8(f, table8[d]) if d in table8 else None
            lines.append(f"addendum 8 table, {d}: " + ("no row" if cmp is None else
                         ("matches" if all(cmp.values()) else "DIFFERS in " + ", ".join(k for k, v in cmp.items() if not v))))
    lines.append("K = 2 sum_i min(n_i+, n_i-) over the items with both classes: the FT-B panel (and the FT-S draw and the FT-N "
                 "panel) of the domain; no adapter is trained here (addendum 13 section 2)")
    return "\n".join(lines)


# ---------------------------------------------------------------- the recorded training recipe and the adapters' checks
def optimizer_steps(n_examples: int, bsz: int, grad_accum: int, epochs: float) -> int:
    """The number of optimizer steps of the Trainer's schedule on one process: ceil(epochs x ceil(ceil(n / bsz) / grad_accum))."""
    batches = math.ceil(int(n_examples) / int(bsz))
    per_epoch = max(batches // int(grad_accum) + int(batches % int(grad_accum) > 0), 1)
    return int(math.ceil(float(epochs) * per_epoch))


def check_ftb_training(ref: dict, run: dict, *, out: str, seed: int, train: str, k: int, where: str = "train_config.json") -> None:
    """A trained FT-B / FT-S / FT-N adapter: its recorded arguments equal the real s0's except --train, --out and --seed (the given
    ones), the prompt-determined facts that a subset panel keeps agree, and it trained on exactly the K examples of its panel
    (n_examples plus the skipped over-long ones)."""
    a, b = fq.train_args(ref, "the real adapter s0's train_config.json"), fq.train_args(run, where)
    for key, want in (("train", str(train)), ("out", str(out)), ("seed", int(seed))):
        if b[key] != want:
            raise FtbError(f"{where}: {key} is {b[key]!r}, expected {want!r}")
    diff = [x for x in fq.TRAIN_KEYS if x not in fq.REPLACED and a[x] != b[x]]
    diff += [x for x in ("loss", "hist_len_used", "panel_kind") if ref.get(x) != run.get(x)]
    if diff:
        raise FtbError(f"{where} differs from the recorded recipe of the real s0 in {diff} (only {list(fq.REPLACED)} may differ)")
    n, skipped = run.get("n_examples"), run.get("n_skipped_overlength")
    if n is None or skipped is None or int(n) + int(skipped) != int(k):
        raise FtbError(f"{where}: trained on {n} examples (+ {skipped} skipped), the panel has K = {k}")
    hist, ref_hist = run.get("max_history_len_in_panel"), ref.get("max_history_len_in_panel")
    if hist is not None and ref_hist is not None and int(hist) > int(ref_hist):
        raise FtbError(f"{where}: the panel's longest history ({hist}) exceeds the registered TRAIN panel's ({ref_hist})")


def write_provenance(adapter_dir, *, name: str, panel_sha1: str, manifest_sha1: str, k: int, cfg: dict, ref_cfg: dict) -> dict:
    tag, seed = split_adapter(name)
    args = fq.train_args(cfg)
    steps = optimizer_steps(cfg["n_examples"], args["bsz"], args["grad_accum"], args["epochs"])
    ref_steps = optimizer_steps(ref_cfg["n_examples"], args["bsz"], args["grad_accum"], args["epochs"])
    rec = {"purpose": "FT-B / FT-S / FT-N adapter (exploratory) trained by scripts/sigir/run_ftb.sh", "adapter": name,
           "arm": ARM_NAME[tag], "seed": seed, "panel": adapter_panel(name)[0], "panel_sha1": panel_sha1, "manifest_sha1": manifest_sha1,
           "K": int(k), "n_examples": int(cfg["n_examples"]), "n_skipped_overlength": int(cfg["n_skipped_overlength"]),
           "optimizer_steps": {"this_adapter": steps, "registered_sft": ref_steps, "smaller_than_registered_sft": steps < ref_steps,
                               "derived": "ceil(epochs x ceil(ceil(n_examples / bsz) / grad_accum)) from the recorded train_config.json "
                                          "(the trainer saves no trainer_state)"},
           "args": args, "code_sha1": {"ftb_panel.py": file_sha1(__file__)}}
    (Path(adapter_dir) / PROVENANCE_NAME).write_bytes(json_bytes(rec))
    return rec


def check_provenance(adapter_dir, *, name: str, panel_sha1: str, manifest_sha1: str) -> dict:
    path = Path(adapter_dir) / PROVENANCE_NAME
    if not path.is_file():
        raise FtbError(f"{adapter_dir} has no {PROVENANCE_NAME}: it was not trained by run_ftb.sh on the registered panel (a "
                       "registered adapter, or one trained by hand); move it aside and rerun stage 2")
    rec = read_json(path)
    if rec.get("adapter") != name:
        raise FtbError(f"{adapter_dir} records adapter {rec.get('adapter')!r}, not {name}: move it aside and rerun stage 2")
    if rec.get("panel_sha1") != panel_sha1 or rec.get("manifest_sha1") != manifest_sha1:
        raise FtbError(f"{adapter_dir} was trained on a panel with sha1 {rec.get('panel_sha1')} (manifest {rec.get('manifest_sha1')}), "
                       f"the current {adapter_panel(name)[0]} has {panel_sha1} (manifest {manifest_sha1}): move the adapter aside "
                       "and rerun stage 2")
    return rec


# ---------------------------------------------------------------- command line
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--domain", default=None, help="build, verify: ml1m (the only dataset)")
    ap.add_argument("--domains", default=",".join(REPORT_DOMAINS), help="report: comma list of domains")
    ap.add_argument("--panels_root", default="outputs/confrec/ftgrid/panels", help="report: panels/ (holds <d>/train.jsonl)")
    ap.add_argument("--addendum8", default=None, help="report: addendum 8's file; its multiplicity table is compared")
    ap.add_argument("--out", default=None, help="report: also write the facts as strict JSON here")
    ap.add_argument("--train", default=None, help="build, verify: the registered panels/ml1m/train.jsonl")
    ap.add_argument("--out_dir", default=None, help="build, verify: ftb/ml1m/ (the ten files go here)")
    ap.add_argument("--panels_dir", default=None, help="verify_adapter: the directory of the built panels (out_dir of build)")
    ap.add_argument("--split", default=None, help="ftgrid_split.json of the dataset (info: also read)")
    ap.add_argument("--adapters", default=None, help="recipe / scoring: the real adapters' directory (s0-s2); verify_*: the FT-B root's "
                                                     "adapters/ml1m/ (b0 ... n1)")
    ap.add_argument("--ref_adapters", default=None, help="verify_adapter: the real adapters' directory (s0's config)")
    ap.add_argument("--names", default=None, help="verify_adapter: comma list of adapters (b0,b1,b2,r0,...)")
    ap.add_argument("--scores", default=None, help="scoring: the real scores/ml1m/ ; verify_scores: the arm's scores/<tag>/ml1m/")
    ap.add_argument("--ref_scores", default=None, help="verify_scores: the real scores/ml1m/ (the passes of s0)")
    ap.add_argument("--arm", default=None, choices=list(ARM_TAGS), help="verify_scores: b (FT-B), r (FT-S) or n (FT-N)")
    ap.add_argument("--pass", dest="pass_kind", default=None, choices=list(PASSES), help="scoring, verify_scores: like or swap")
    ap.add_argument("--seeds", default=None, help="verify_scores: comma list of seeds of the arm")
    ap.add_argument("--real_scores", default=None, help="link: the real scores/ml1m/")
    ap.add_argument("--q_scores", default=None, help="link: the arm's scores/<tag>/ml1m/")
    ap.add_argument("--models", default=",".join(LINKED_MODELS), help="link: the models to link")
    ap.add_argument("--allow_copy", action="store_true", help="link: DRY_RUN only, copy where symlinks are unavailable")
    ap.add_argument("--model", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--train_ref", default=None, help="recipe: the TRAIN file path the real adapters recorded")
    ap.add_argument("--train_file", default=None, help="recipe: the registered panels/ml1m/train.jsonl (bytes compared)")
    ap.add_argument("--data", default=None, help="scoring: panels/ml1m/eval.jsonl (like) or eval_sd_test.jsonl (swap)")
    ap.add_argument("--write", action="store_true", help="verify_adapter: write the adapters' provenance records")
    ap.add_argument("--selection", default=None, help="info: selection.json")
    ap.add_argument("--gate", default=None, help="info: gate_ft.json")
    ap.add_argument("--pilot_log", default=None, help="record: docs/sigir/PILOT_LOG.md")
    ap.add_argument("--files", nargs="+", default=None, help="record: files (repo-relative, or absolute) whose sha1 the log must hold")
    ap.add_argument("--root", default=None, help="record: the repo root (default: this checkout)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="record: print the lines for the pilot log")
    ap.add_argument("--append", action="store_true", help="record: DRY_RUN only, append the missing lines to the (temporary) pilot "
                                                          "log, the human step of the rehearsal")
    return ap.parse_args(argv)


def need(a: argparse.Namespace, *names: str) -> None:
    missing = [f"--{n}" for n in names if getattr(a, n) is None]
    if missing:
        raise FtbError(f"{a.command} needs {' '.join(missing)}", 2)


def int_list(text, what: str, allowed) -> list:
    try:
        vals = [int(x) for x in str(text).split(",") if x != ""]
    except ValueError:
        raise FtbError(f"{what} {text!r} is not a comma list of integers", 2) from None
    bad = [v for v in vals if v not in allowed]
    if not vals or bad:
        raise FtbError(f"{what} {text!r}: seeds of this arm are {list(allowed)}", 2)
    return vals


def cmd_recipe(a) -> list:
    need(a, "adapters", "model", "variant", "train_ref")
    adapters = a.adapters.rstrip("/")
    split = read_json(a.split) if a.split else None
    base = fq.check_recipe(fq.sft_configs(adapters), adapters=adapters, model=a.model, variant=a.variant, train_ref=a.train_ref,
                           split=split, split_recipe=False)
    if a.train_file:
        fq.check_train_file(a.train_ref, a.train_file)
    return fq.recipe_flags(base)


def cmd_scoring(a) -> list:
    need(a, "pass_kind", "scores", "adapters", "data", "model", "variant")
    return fl.cmd_scoring(argparse.Namespace(command="scoring", arm=a.pass_kind, scores=a.scores, adapters=a.adapters, data=a.data,
                                             model=a.model, variant=a.variant))


def cmd_verify_adapter(a) -> list:
    need(a, "adapters", "ref_adapters", "names", "panels_dir")
    adapters, panels = a.adapters.rstrip("/"), a.panels_dir.rstrip("/")
    ref = read_json(Path(a.ref_adapters.rstrip("/")) / "s0" / "train_config.json")
    names = [x for x in a.names.split(",") if x]
    if not names:
        raise FtbError("verify_adapter: --names is empty", 2)
    out_recs = []
    for name in names:
        split_adapter(name)
        pname, mname = adapter_panel(name)
        pfile, mfile = Path(panels) / pname, Path(panels) / mname
        for p in (pfile, mfile):
            if not p.is_file():
                raise FtbError(f"{p}: missing (stage 1 builds the panels)")
        man = read_json(mfile)
        if file_sha1(pfile) != man["panel"]["sha1"]:
            raise FtbError(f"{pfile} is not the file {mfile} records (sha1 {man['panel']['sha1']})")
        out = f"{adapters}/{name}"
        cfg = read_json(Path(out) / "train_config.json")
        check_ftb_training(ref, cfg, out=out, seed=split_adapter(name)[1], train=f"{panels}/{pname}", k=man["K"],
                           where=f"{out}/train_config.json")
        if a.write:
            write_provenance(out, name=name, panel_sha1=man["panel"]["sha1"], manifest_sha1=file_sha1(mfile), k=man["K"], cfg=cfg,
                             ref_cfg=ref)
        out_recs.append(check_provenance(out, name=name, panel_sha1=man["panel"]["sha1"], manifest_sha1=file_sha1(mfile)))
    return out_recs


def cmd_verify_scores(a) -> None:
    need(a, "arm", "pass_kind", "scores", "ref_scores", "adapters", "seeds")
    scores, adapters = a.scores.rstrip("/"), a.adapters.rstrip("/")
    ref = fq.report_config(read_json(Path(a.ref_scores.rstrip("/")) / "s0" / a.pass_kind / "report.json"),
                           f"the {a.pass_kind} pass of the real s0")
    for k in int_list(a.seeds, "--seeds", ARM_SEEDS[a.arm]):
        run = Path(scores) / f"s{k}" / a.pass_kind / "report.json"
        fl.check_same_arm_scoring(a.pass_kind, ref, fq.report_config(read_json(run), str(run)), lora=f"{adapters}/{a.arm}{k}",
                                  where=str(run))


def cmd_link(a) -> list:
    need(a, "real_scores", "q_scores")
    try:
        return fq.link_models(a.real_scores, a.q_scores, [m for m in a.models.split(",") if m], allow_copy=a.allow_copy)
    except FtbError as e:                          # the shared helper speaks of "the FT-Q root": here it is the FT-B root
        raise FtbError(str(e).replace("FT-Q", "FT-B"), e.code) from None


def cmd_record(a) -> int:
    need(a, "pilot_log", "files")
    if a.do_print:
        print("\n".join(fq.record_lines(a.files, a.root)))
        return 0
    missing = fq.record_missing(a.pilot_log, a.files, a.root)
    if missing and a.append:
        log = Path(a.pilot_log)
        lead = b"" if not log.stat().st_size or log.read_bytes().endswith(b"\n") else b"\n"
        with open(log, "ab") as f:
            f.write(lead + ("\n".join(fq.record_lines(missing, a.root)) + "\n").encode("utf-8"))
        print(f"[dry] recorded {', '.join(missing)} in {log} (the human step, on the temporary pilot log)")
        missing = fq.record_missing(a.pilot_log, a.files, a.root)
    if missing:
        raise FtbError(f"FT-B record (Amendment 3 addendum 13 section 5): the sha1 of these files is not in {a.pilot_log}: {missing}; "
                       f"run `python -m src.confrec.ftb_panel record --pilot_log {a.pilot_log} --files {' '.join(a.files)} --print`, "
                       "record the lines in the pilot log and push it", 4)
    print(f"FT-B record OK: the sha1 of {', '.join(a.files)} is in {a.pilot_log}")
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.command in ("build", "verify"):
            need(a, "domain", "train", "out_dir")
            if a.domain != DOMAIN:
                raise FtbError(f"--domain {a.domain!r}: FT-B, FT-S and FT-N run on {DOMAIN} only (addendum 13 section 2)", 2)
            if a.command == "build":
                man = build(a.train, a.out_dir, domain=a.domain, split=a.split)
                b = man[panel_name("b")]
                print(json.dumps({name: m["panel"] for name, m in man.items()}, allow_nan=False))
                print(f"ftb_panel build: {a.domain}: K = {b['K']} examples of {b['source']['examples']} ({b['kept_items']} kept items, "
                      f"{b['share_examples_dropped']:.1%} of the TRAIN examples dropped); FT-S draws K examples per seed; FT-N "
                      "permutes the labels within the kept items", file=sys.stderr)
            else:
                man = verify(a.train, a.out_dir, domain=a.domain, split=a.split)
                print(f"ftb_panel verify: {a.domain}: the {len(man)} panels and their manifests in {a.out_dir} are byte for byte "
                      "the panels recomputed from the registered train.jsonl")
        elif a.command == "report":
            domains = [d for d in a.domains.split(",") if d]
            bad = [d for d in domains if d not in REPORT_DOMAINS]
            if bad or not domains:
                raise FtbError(f"--domains {a.domains!r}: choose from {', '.join(REPORT_DOMAINS)}", 2)
            root = Path(a.panels_root)
            results = {d: report_domain(root / d / "train.jsonl", root / d / "ftgrid_split.json", d) for d in domains}
            table8 = parse_addendum8(Path(a.addendum8).read_text(encoding="utf-8")) if a.addendum8 else None
            if a.addendum8 and not table8:
                raise FtbError(f"{a.addendum8}: no multiplicity table found", 1)
            print(format_report(results, table8))
            if a.out:
                from src.confrec.stats import strict_json
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_bytes(json_bytes(strict_json({"spec": SPEC, "status": "exploratory", "domains": results})))
        elif a.command == "recipe":
            print("\n".join(cmd_recipe(a)))
        elif a.command == "scoring":
            print("\n".join(cmd_scoring(a)))
        elif a.command == "verify_adapter":
            cmd_verify_adapter(a)
            print(f"adapters {a.names}: the recorded arguments equal the real s0's except {', '.join(fq.REPLACED)}; trained on exactly "
                  "the K examples of their panel; provenance OK")
        elif a.command == "verify_scores":
            cmd_verify_scores(a)
            print(f"{a.pass_kind} pass of {ARM_NAME[a.arm]} seeds {a.seeds}: the recorded scoring arguments equal the real s0's "
                  f"{a.pass_kind} pass except lora")
        elif a.command == "link":
            print("\n".join(cmd_link(a)))
        elif a.command == "info":
            print("\n".join(fq.cmd_info(a)))
        else:
            return cmd_record(a)
    except FtbError as e:
        print(f"ftb_panel {a.command}: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
