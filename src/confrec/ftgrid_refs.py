"""Amendment 3 addendum 15 (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_15.md): reference extensions for the rated panels, for one
domain and one backbone root. EXPLORATORY and descriptive: no hypothesis, no family, no Holm correction, no p-value; every
interval is a 95% user-cluster percentile interval (2,000 resamples, seed 0). It changes no registered number.

    python -m src.confrec.ftgrid_refs build --domain toys --split outputs/confrec/ftgrid/panels/toys/ftgrid_split.json \
        --panels outputs/confrec/ftgrid/panels/toys --scores_root outputs/confrec/ftgrid/scores \
        --models zeroshot,s0,s1,s2 --raw data/raw --root_label main --out outputs/confrec/ftgrid_refs/main/toys.json \
        [--registered_report outputs/confrec/ftgrid/report/toys.json] [--registered_extra outputs/confrec/ftgrid/extra/toys.json] \
        [--select_rows cal_eval_users|cal_users_outside_sd] [--max_gib 20] [--n_boot 2000] [--seed 0] [--rehearsal]
    python -m src.confrec.ftgrid_refs check_resume <the build arguments>     # exit 0: --out is current, 1: rebuild
    python -m src.confrec.ftgrid_refs record --print | --pilot_log docs/sigir/PILOT_LOG.md [--files ...] [--append]

Pure function of stored files (CPU only, deterministic, strict JSON, no clock, host or path in the output). `build` writes
<out stem>_tables.csv (long format) and then --out (the completion marker, which records the CSV's sha1 and the fingerprint of
every input and of the code: `check_resume` compares it with the current inputs). Every block that cannot be computed is
{"available": false, "reason": ...}. Writes only below outputs/confrec/ftgrid_refs/ (a --rehearsal build writes outside
outputs/confrec instead); a path under outputs/confrec/ftgrid*, ftmethod*, ftprune*, gatefix*, gateft*, nextitem_audit* (other
than ftgrid_refs) is refused in every spelling, symbolic links included.

Reuse, not re-derivation. Rows, users S_d, splits, scores, contexts (ftgrid_extra.load_inputs, ftgrid_report.make_ctx), the
cross-fitted stackers (ftgrid_report.crossfit; ftgrid_extra.wu_fit, auc_table, Cols, wu_regime, ed_rows), the user-cluster
bootstrap (ftgrid_report.mean_draws, rec), the tie-aware UAUC (ftgrid_report.user_aucs), the raw-event scan and the item means
(forensics.scan_events, prior_means), the registered biased MF (forensics.fit_biased_mf, fold_in_user, mf_pair_scores) are
imported read-only and unchanged. The temporal MF is refitted by calling fit_biased_mf with another regularisation weight (it
has the parameter); a test reproduces the registered temporal MF (arrays, train RMSE, held-out RMSE, UAUC) at the registered
weight. Item parameters of every fit are kept, user parameters are dropped (memory); panel users are folded in as the registered
code does it.

Blocks. item_means (section 2.1): m_TR from the LoRA's own TRAIN labels, the registered m and the time-matched m_T beside it, the
share of TEST pairs with a TRAIN example, dUAUC(l - m), dUAUC(l - m_T), dUAUC(l - m_TR) for the zero-shot model and the seed
mean of the LoRA. tuned_mf (section 2.2): the weight chosen by held-out RMSE on CAL rows, variants (a) whole-history fold-in,
(b) last-10-event fold-in, (c) item-bias-only, the personal residuals of (a) and (b), warm / cold / all-pair UAUC with counts of
cold items and users, then G_CF (E-D and E-W versions), G_LLM|CF and G_CF|LLM with the residual of (b) for every regime.
content (section 2.3): TF-IDF fitted on TRAIN rows only, the score c, UAUC of c and of the user-centred c, G_content (E-D and
E-W), G_LLM|content for every regime. labels (section 3): SIGNAL_PRESENT / SIGNAL_ABSENT / INCONCLUSIVE per panel and reference.

Readings. Where the addendum leaves a choice, the most literal reading is taken; the items below are copied into every output as
`readings` (parsed from this docstring) and listed in the implementer's report.
R1  Status. Everything is exploratory (addendum 15: written after every rated-panel result of the fine-tuned programme had been
    read, before any quantity of this addendum was computed). No p-value, ci_excludes_0, sigma_seed rule or reading rule of the
    registered functions is kept: intervals are descriptions. The registered MF numbers stay as registered; this file only adds.
R2  Rows. Level UAUCs and the contrasts of section 2.1 are on E-A's rows (the EVAL users' TEST rows, finite in every score and,
    for a regime, in the like logit of every model of the regime), identical for the scores of one block. Stackers (G blocks)
    are on S_d's TEST rows: the per-regime blocks on E-D's rows of the regime (ftgrid_extra.ed_rows) with a finite m and
    reference; the model-free G of a reference (the one the labels read) on S_d's TEST rows with a finite m and reference.
R3  m_TR. TRAIN examples are the candidates of train.jsonl as written (after the registered cap, before T_d); the key of an
    example is str(candidate_item_ids); mu_TRAIN = the like rate of those examples; k = 5. An item without a TRAIN example
    scores mu_TRAIN. m is the registered shrunk item mean q-hat (forensics.prior_means, k = 5, candidate times) and m_T the
    same at T_d for every pair (as ftgrid_extra E-J); both are computed here from the raw scan, so q-hat is checked against
    the registered report when --registered_report is given (R16).
R4  Seen share. The share of TEST pairs whose item occurs among the candidates of train.jsonl (ftgrid_report.make_ctx's `seen`),
    over all EVAL users' TEST pairs, over S_d's TEST pairs and over the rows of each regime's contrasts.
R5  Regularisation weight. The registered routine has two weights: lam on the latent factors (registered 0.05, ALS-WR weighting)
    and lam_bias on the biases (registered 5.0). The addendum's grid {0.05, 0.2, 1, 5, 20} contains the registered 0.05 and
    its 'regularisation weight' is read as lam; lam_bias stays 5.0. The fold-in uses the fit's lam, as the registered code does.
R6  Selection. The held-out RMSE of the withheld candidate ratings is taken on the CAL rows (candidate timestamp < T_d) of the
    EVAL users, on pairs that are warm under the whole-history fold-in (as the registered held-out RMSE: cold pairs excluded,
    which are identical for every weight). The parenthesis 'users disjoint from S_d's TEST rows' is read as rows (a CAL row
    is disjoint from S_d's TEST rows); the same table on CAL rows of the EVAL users outside S_d is recorded as a sensitivity
    with its own choice, and --select_rows cal_users_outside_sd selects on it instead. Ties (RMSE within 1e-12 of the minimum):
    the larger weight. A TEST row is never used.
R7  Fold-in. (a) is the registered fold-in: the user's own non-candidate first events strictly before the user's first candidate
    timestamp, all of them. (b) restricts (a)'s event list to its last 10 events (chronological order of the scan, ties by item
    id as the panel builder): the subset reading of 'the fold-in restricted to the last 10 events'. The share of users whose
    10 events equal the last 10 history items of the panel row is recorded; the registered window of the split's prompt variant
    is recorded beside the addendum's 10 (never replaced).
R8  Item-bias-only (c). The registered routine with dim = 0 (the factors fully regularised away: P = Q = 0, no user factor), the
    registered lam_bias = 5.0 and 15 iterations; the user bias is part of that fit (as in the registered MF) and is not used
    in scoring: the score is b_i, 0 for an item with no training rating. 'Warm' for (c) is an item with a training rating.
R9  Cold, warm. Warm = the registered mf_warm: the user's fold-in has an event on an item with a training rating (before T_d,
    candidate events excluded) and the item has such a rating. Cold items and cold users are scored by the registered fallback
    (zero bias and zero factors, i.e. the global mean plus the other bias; residual 0) and counted. UAUC over warm / cold / all
    TEST rows of the EVAL users, per-user AUC with ties 1/2, users with both classes among the subset.
R10 G of the collaborative reference. G_CF = dUAUC([m, MF residual (b)] - [m]) with the registered cross-fitted stacker of E-D
    (ftgrid_report.stacker_block / crossfit, fold A = user_halves(S_d, 0)) and the within-user estimator of E-W (ftgrid_extra
    wu_regime: user-centred features, K = 20 splits); G_LLM|CF = dUAUC(M4 - M3) and G_CF|LLM = dUAUC(M4 - M2) of E-F with the
    tuned residual, per regime, by the registered code.
R11 TF-IDF. Implemented without scikit-learn with scikit-learn's defaults spelled out: lower case, token pattern (?u)\\b\\w\\w+\\b,
    unigrams and bigrams, no stop words, sublinear tf (1 + ln count), smooth idf ln((1 + n)/(1 + df)) + 1, L2 norm,
    minimum document frequency 2; a test compares it with TfidfVectorizer when scikit-learn is importable. Documents = the
    distinct items (by item id, first occurrence) of the TRAIN rows, as history item and as candidate; rows of eval.jsonl
    never enter the fit.
R12 Item text. title + category-or-genre string + brand, joined by spaces as one string (bigrams run across the fields). History
    item: history_titles, history_meta, history_brands. Candidate: candidate_titles, candidate_brands and the category string
    parsed from candidate_texts ('Genres: ...' as stored; 'Categories: <cats>. <description>' cut at the first '. ' and at
    80 characters, the build_rated_panels.amazon_meta rule; no categories: ''). The description is never used.
R13 Content score. c(u, i) = mean over the user's history items with rating >= 4 of cos(x_i, x_h) minus the mean over those with
    rating <= 2 (rating 3 ignored), the history being the last 10 events of the row (what the prompt shows; the whole row
    history is a sensitivity). The mean over an empty group is 0 (no evidence); cos is 0 when a vector is zero (an item with
    no vocabulary term). The sensitivity 'both groups non-empty' scores NaN for a user lacking a group instead.
R14 G_LLM|content. The literal text, dUAUC([m, c, e-hat] - [m, c]); the E-F analogue with the item prior, dUAUC([m, c, pi, e-hat]
    - [m, c]), is reported beside it; both with E-W's estimator (primary) and E-D's.
R15 Labels. On the model-free G_r (E-W) interval [lo, hi]: SIGNAL_PRESENT if lo > 0; else SIGNAL_ABSENT if -0.01 <= lo and
    hi <= 0.01; else INCONCLUSIVE (the order of the text: an interval above 0 that also lies within the band is
    SIGNAL_PRESENT, flagged also_within_band). The label is computed whatever the user count; descriptive_min_n is recorded.
R16 Registered checks. With --registered_report / --registered_extra the UAUC of m (and of m_T) on E-A's rows of the zero-shot
    regime must equal the registered value to 1e-9 and the same number of users, else a problem (a summary refuses it).
R17 Memory and time. Sequential per panel; the raw events are dropped after the scan; per fit only the item parameters are kept;
    the estimate (rows x 8 bytes x arrays, factors x 8 bytes) is recorded and a build refuses (exit 3) above --max_gib
    (default 20). Peak RSS is printed, never written to the JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.confrec import forensics as fx
from src.confrec import ftgrid_extra as fe
from src.confrec import ftgrid_report as fr
from src.confrec.prompting import resolve_hist_len
from src.confrec.stats import strict_json

NAN = float("nan")
STATUS = "exploratory"
SPEC = ("idea-stage/PREREG_AMENDMENT_3_ADDENDUM_15.md sections 1-4 (with idea-stage/PREREG_AMENDMENT_3.md sections 1, 3 and "
        "addendum 6 for the registered rows, stackers and bootstrap)")
STATEMENT = ("Exploratory (addendum 15): descriptive references computed after every rated-panel result had been read and before "
             "any quantity of this addendum existed. Intervals are 95% user-cluster percentile intervals (2,000 resamples, seed "
             "0). No hypothesis, no family, no Holm correction, no p-value. The labels SIGNAL_PRESENT / SIGNAL_ABSENT / "
             "INCONCLUSIVE describe the panel and the reference, not the LLM. The registered MF numbers stay as registered.")

# ---- registered constants (addendum 15 and the registered MF / report code)
WEIGHT_GRID = (0.05, 0.2, 1.0, 5.0, 20.0)            # section 2.2: the grid of the regularisation weight
REGISTERED_WEIGHT = 0.05                             # forensics.cf_references' mf_lambda
MF_DIM = fr.MF_KW["mf_dim"]                          # 32
MF_ITERS = fr.MF_KW["mf_iters"]                      # 15
MF_LAMBDA_BIAS = fr.MF_KW["mf_lambda_bias"]          # 5.0
MF_SEED = 0
SHRINK_K = fr.SHRINK_K                               # q-hat / q-hat_T shrinkage, k = 5
TRAIN_SHRINK_K = 5.0                                 # section 2.1: m_TR = (sum labels + 5 mu_TRAIN) / (n_i + 5)
HIST_WINDOW = 10                                     # section 2.2 (b) and the prompt: the last 10 events
SIGNAL_BAND = 0.01                                   # section 3: SIGNAL_ABSENT iff the interval lies within [-0.01, +0.01]
TIE_TOL = 1e-12                                      # RMSE values within this of the minimum are tied (the larger weight)
TFIDF_NGRAM_MAX = 2                                  # unigram and bigram
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR = True
TFIDF_TOKEN = r"(?u)\b\w\w+\b"
META_CHARS = 80                                      # build_rated_panels.META_CHARS (the history_meta cap)
N_BOOT, SEED = fr.N_BOOT, fr.SEED                    # 2,000 user resamples, seed 0
MAX_GIB = 20.0
REGS = ("ZS", "FT")
ROOT_LABELS = ("main", "llama")
SELECT_ROWS = ("cal_eval_users", "cal_users_outside_sd")
LABELS = ("SIGNAL_PRESENT", "SIGNAL_ABSENT", "INCONCLUSIVE")
REGISTERED_PREFIXES = ("ftgrid", "ftmethod", "ftprune", "gatefix", "gateft", "nextitem_audit")
ALLOWED_OUT_NAME = "ftgrid_refs"
RECORD_FILES = ("src/confrec/ftgrid_refs.py", "scripts/sigir/run_ftrefs.sh", "tests/test_confrec_ftrefs.py")
CODE_FILES = ("src/confrec/ftgrid_refs.py", "src/confrec/ftgrid_extra.py", "src/confrec/ftgrid_report.py",
              "src/confrec/forensics.py", "src/confrec/stats.py", "src/confrec/metrics.py",
              "src/confrec/build_rated_panels.py", "src/confrec/prompting.py", "src/confrec/categories.py")
DROP_KEYS = frozenset({"p", "ci_excludes_0", "sigma_seed_rule", "all_seeds_same_sign_as_mean", "abs_mean_gt_2_sigma_seed",
                       "reading_G", "reading_G_CF"})        # registered decision machinery: not kept (R1)
CSV_COLS = ("block", "panel", "root_label", "statistic", "estimate", "lo", "hi", "n_users", "n_pairs", "descriptive_min_n")


def _readings(doc: str | None) -> tuple:
    """The R<n> items of the module docstring's Readings section, one string each (label kept)."""
    return fe._readings(doc) if doc else ()


READINGS = _readings(__doc__)


class RefsError(Exception):
    """A refused input or output (exit code `code`)."""

    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- small helpers
def _na(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _num(x) -> float:
    return fr._num(x)


def _fin(x) -> bool:
    return fr._fin(x)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canon(p) -> Path:
    """Canonical path: symlinks and junctions resolved, '..' removed, case folded where the filesystem folds it."""
    return Path(os.path.normcase(os.path.realpath(str(p))))


def _sha1_file(path) -> str | None:
    return fx.file_sha1(path) if Path(path).is_file() else None


def descriptive(x):
    """The block without the registered decision machinery (p-values, sigma_seed rules, reading rules): intervals only."""
    if isinstance(x, dict):
        return {k: descriptive(v) for k, v in x.items() if k not in DROP_KEYS}
    if isinstance(x, (list, tuple)):
        return [descriptive(v) for v in x]
    return x


def code_sha1() -> dict:
    root = repo_root()
    return {Path(rel).name: _sha1_file(root / rel) for rel in CODE_FILES}


# ---------------------------------------------------------------- the output guard
def registered_root_of(path) -> str | None:
    """The registered output root (first component below outputs/confrec) that `path` resolves into, else None.
    outputs/confrec/ftgrid_refs is the one name that starts with a registered prefix and is allowed."""
    base = _canon(repo_root() / "outputs" / "confrec")
    p = _canon(path)
    try:
        rel = p.relative_to(base)
    except ValueError:
        return None
    first = rel.parts[0] if rel.parts else ""
    if not first or first == ALLOWED_OUT_NAME:
        return None
    return first if any(first.startswith(pre) for pre in REGISTERED_PREFIXES) else None


def check_out_path(out, rehearsal: bool = False) -> None:
    """Refuse (RefsError, exit 2) an output path under a registered root and, outside a rehearsal, anywhere but below
    outputs/confrec/ftgrid_refs/. A rehearsal never writes inside outputs/confrec at all."""
    reg = registered_root_of(out)
    if reg is not None:
        raise RefsError(f"refused: {out} resolves into the registered root outputs/confrec/{reg}; ftgrid_refs writes only "
                        f"below outputs/confrec/{ALLOWED_OUT_NAME}/")
    base = _canon(repo_root() / "outputs" / "confrec")
    allowed = _canon(repo_root() / "outputs" / "confrec" / ALLOWED_OUT_NAME)
    p = _canon(out)
    inside = p == allowed or allowed in p.parents
    if rehearsal:
        if p == base or base in p.parents:
            raise RefsError(f"refused: a --rehearsal build never writes inside outputs/confrec ({out})")
    elif not inside:
        raise RefsError(f"refused: {out} is not below outputs/confrec/{ALLOWED_OUT_NAME}/ (the only place this module "
                        "writes; use --rehearsal for a rehearsal outside outputs/confrec)")


def check_root_label(a) -> None:
    """A build reading a registered root's scores must carry that root's label (the registered check of ftgrid_extra)."""
    try:
        fe.check_root_label(a)
    except SystemExit as e:
        raise RefsError(f"refused: --root_label {a.root_label} does not match the registered root of --scores_root",
                        int(e.code or 2)) from None


# ---------------------------------------------------------------- m_TR (section 2.1)
def train_item_stats(train_rows: list) -> SimpleNamespace:
    """Per item the number of TRAIN examples and of their likes, and the TRAIN like rate mu_TRAIN, from the candidates of
    train.jsonl (candidate_item_ids, candidate_labels)."""
    n, s = Counter(), Counter()
    total = pos = 0
    for r in train_rows:
        for it, y in zip(r["candidate_item_ids"], r["candidate_labels"]):
            n[str(it)] += 1
            s[str(it)] += int(y)
            total += 1
            pos += int(y)
    return SimpleNamespace(n=n, s=s, n_examples=total, n_pos=pos, mu=pos / total if total else NAN)


def m_tr_scores(items, st, k: float = TRAIN_SHRINK_K) -> np.ndarray:
    """m_TR(i) = (sum of the labels of the TRAIN examples of i + k mu_TRAIN) / (n_i + k); an item without a TRAIN example
    scores mu_TRAIN."""
    out = np.full(len(items), NAN)
    if not math.isfinite(st.mu):
        return out
    for j, it in enumerate(items):
        out[j] = (st.s.get(it, 0) + k * st.mu) / (st.n.get(it, 0) + k)
    return out


# ---------------------------------------------------------------- item text and TF-IDF (section 2.3)
_TOKEN_RE = re.compile(TFIDF_TOKEN)


def item_text(title, meta, brand) -> str:
    """title + category-or-genre string + brand as one string."""
    return " ".join(p for p in (str(title or "").strip(), str(meta or "").strip(), str(brand or "").strip()) if p)


def candidate_meta(text) -> str:
    """The category-or-genre string of a candidate from candidate_texts: 'Genres: ...' as stored (ML-1M); 'Categories:
    <cats>. <description>' cut at the first '. ' and at META_CHARS characters (build_rated_panels.amazon_meta); '' otherwise."""
    t = str(text or "")
    if t.startswith("Genres: "):
        return t if t[len("Genres: "):].strip() else ""
    if t.startswith("Categories: "):
        body = t[len("Categories: "):]
        k = body.find(". ")
        cats = (body if k < 0 else body[:k])[:META_CHARS]
        return f"Categories: {cats}" if cats.strip() else ""
    return ""


def _terms(text) -> list:
    toks = _TOKEN_RE.findall(str(text).lower())
    out = list(toks)
    if TFIDF_NGRAM_MAX >= 2:
        out += [a + " " + b for a, b in zip(toks, toks[1:])]
    return out


class Tfidf:
    """TF-IDF as scikit-learn's TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2) with its other defaults
    (lower case, no stop words, smooth idf, L2 norm), in pure Python: a vector is {term index: weight}."""

    def __init__(self, min_df: int = TFIDF_MIN_DF, sublinear_tf: bool = TFIDF_SUBLINEAR):
        self.min_df, self.sublinear_tf = min_df, sublinear_tf
        self.vocab: dict = {}
        self.idf = np.zeros(0)
        self.n_docs = 0

    def fit(self, docs) -> "Tfidf":
        df, n = Counter(), 0
        for d in docs:
            n += 1
            df.update(set(_terms(d)))
        keep = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: j for j, t in enumerate(keep)}
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in keep], float)
        self.n_docs = n
        return self

    def transform(self, text) -> dict:
        cnt = Counter(t for t in _terms(text) if t in self.vocab)
        if not cnt:
            return {}
        w = {}
        for t, c in cnt.items():
            j = self.vocab[t]
            tf = 1.0 + math.log(c) if self.sublinear_tf else float(c)
            w[j] = tf * float(self.idf[j])
        norm = math.sqrt(math.fsum(v * v for v in w.values()))
        return {j: v / norm for j, v in w.items()}


def cosine(a: dict, b: dict) -> float:
    """Dot product of two L2-normalised sparse vectors (0 when either is zero); exactly-rounded sum, so equal vectors give
    equal values whatever the term order."""
    if len(a) > len(b):
        a, b = b, a
    return math.fsum(v * b[j] for j, v in a.items() if j in b)


def row_history(row: dict, window: int | None):
    """[(item id, item text, rating)] of the last `window` events of the row's history (all when window is None); None when
    the history lists are misaligned."""
    ids = list(row.get("history_item_ids") or [])
    n = len(ids)
    titles = list(row.get("history_titles") or [])
    metas = list(row.get("history_meta") or [""] * n)
    brands = list(row.get("history_brands") or [""] * n)
    ratings = list(row.get("history_ratings") or [])
    if not (len(titles) == len(metas) == len(brands) == len(ratings) == n):
        return None
    lo = 0 if window is None else max(0, n - window)
    return [(str(ids[k]), item_text(titles[k], metas[k], brands[k]), _num(ratings[k])) for k in range(lo, n)]


def row_candidates(row: dict) -> list:
    """[(item id, item text)] of the row's candidates."""
    ids = row["candidate_item_ids"]
    n = len(ids)
    titles = list(row.get("candidate_titles") or [""] * n)
    texts = list(row.get("candidate_texts") or [""] * n)
    brands = list(row.get("candidate_brands") or [""] * n)
    return [(str(ids[j]), item_text(titles[j] if j < len(titles) else "", candidate_meta(texts[j] if j < len(texts) else ""),
                                    brands[j] if j < len(brands) else "")) for j in range(n)]


def tfidf_docs(train_rows: list) -> dict:
    """{item id: text} of the distinct items of the TRAIN rows (history items and candidates, first occurrence)."""
    docs = {}
    for r in train_rows:
        for iid, text, _ in row_history(r, None) or []:
            docs.setdefault(iid, text)
        for iid, text in row_candidates(r):
            docs.setdefault(iid, text)
    return docs


def text_consistency(rows: list) -> dict:
    """How many item ids carry more than one text across all rows (history and candidate renderings of one item)."""
    seen: dict = {}
    for r in rows:
        for iid, text, _ in row_history(r, None) or []:
            seen.setdefault(iid, set()).add(text)
        for iid, text in row_candidates(r):
            seen.setdefault(iid, set()).add(text)
    return {"n_items": len(seen), "n_items_with_conflicting_text": sum(len(v) > 1 for v in seen.values())}


def content_scores(eval_rows: list, tf: Tfidf, window: int | None = HIST_WINDOW, empty_group: str = "zero"):
    """c(u, i) for every pair of the panel rows (panel_pairs order). liked = rating >= 4, disliked = rating <= 2, rating 3
    ignored; the mean over an empty group is 0 (empty_group = 'zero') or the score is NaN (empty_group = 'nan'). Returns
    (c array, info counts)."""
    cache: dict = {}

    def vec(text):
        v = cache.get(text)
        if v is None:
            v = cache[text] = tf.transform(text)
        return v
    out, info = [], Counter()
    for r in eval_rows:
        cands = row_candidates(r)
        hist = row_history(r, window)
        if hist is None:
            out += [NAN] * len(cands)
            info["rows_with_misaligned_history"] += 1
            continue
        liked = [vec(t) for _, t, rt in hist if rt >= 4]
        disliked = [vec(t) for _, t, rt in hist if rt <= 2]
        info["rows"] += 1
        info["rows_without_liked_item"] += int(not liked)
        info["rows_without_disliked_item"] += int(not disliked)
        info["rows_without_either"] += int(not liked and not disliked)
        for _, text in cands:
            x = vec(text)
            info["pairs"] += 1
            info["pairs_candidate_without_vocabulary_term"] += int(not x)
            if empty_group == "nan" and (not liked or not disliked):
                out.append(NAN)
                continue
            pos = math.fsum(cosine(x, h) for h in liked) / len(liked) if liked else 0.0
            neg = math.fsum(cosine(x, h) for h in disliked) / len(disliked) if disliked else 0.0
            out.append(pos - neg)
    return np.array(out, float), dict(info)


# ---------------------------------------------------------------- the temporal biased MF (section 2.2)
def memory_estimate(n_train: int, n_users: int, n_items: int, dim: int = MF_DIM, n_events: int | None = None) -> dict:
    """Bytes of the arrays one fit_biased_mf call holds (8 bytes per float64 / int64): the masked copies of u, i, r and the
    arrays u, i, r, e of the fit, the latent factors (dim x users and dim x items), the bias and count vectors, and the
    temporaries of one latent dimension (about six arrays of the rating length); n_events adds the raw scan (about 100 bytes per
    event for the Python tuples while the events are loaded, then the scan arrays)."""
    f8 = 8
    ratings = n_train * f8 * (3 + 4)
    factors = dim * (n_users + n_items) * f8
    vectors = (4 * n_users + 4 * n_items) * f8
    temporaries = n_train * f8 * 6
    out = {"ratings_arrays": ratings, "latent_factors": factors, "bias_and_count_vectors": vectors,
           "loop_temporaries": temporaries, "fit_total": ratings + factors + vectors + temporaries}
    if n_events is not None:
        out["raw_events_python_peak"] = int(n_events) * 100
        out["scan_arrays"] = int(n_events) * f8 * 6
        out["peak_total"] = max(out["raw_events_python_peak"] + out["scan_arrays"], out["scan_arrays"] + out["fit_total"])
    return out


def mf_context(cx, scan: dict, T: float) -> SimpleNamespace:
    """What every fit and fold-in of one panel shares: the training ratings (first ratings with ts < T_d, the panel users'
    candidate events excluded: forensics.scan_events), per-pair item indices and warm indices, each panel user's first
    candidate time, the withheld candidate ratings (the registered expression)."""
    users, items = cx.E["user"].tolist(), cx.E["item"].tolist()
    n_u, n_i = len(scan["uid"]), len(scan["iid"])
    tr = scan["mf_t"] < T
    tr_u, tr_i, tr_r = scan["mf_u"][tr], scan["mf_i"][tr], scan["mf_r"][tr]
    nu_t = np.bincount(tr_u, minlength=n_u)
    ni_t = np.bincount(tr_i, minlength=n_i)
    ii = np.array([scan["iid"].get(i, -1) for i in items], np.int64)
    ii_t = fx._warm_index(ii, ni_t)
    first_t: dict = {}
    for u, t in zip(users, cx.ts):
        if math.isfinite(t):
            first_t[u] = min(t, first_t.get(u, math.inf))
    held = np.where(np.isfinite(cx.E["rating"]), cx.E["rating"],
                    [scan["own"].get((u, i), (NAN, NAN))[1] for u, i in zip(users, items)])
    for k in ("mf_u", "mf_i", "mf_r", "mf_t"):       # the full arrays are not needed any more
        scan.pop(k, None)
    return SimpleNamespace(scan=scan, T=float(T), n_u=n_u, n_i=n_i, tr_u=tr_u, tr_i=tr_i, tr_r=tr_r, nu_t=nu_t, ni_t=ni_t,
                           users=users, items=items, ii=ii, ii_t=ii_t, first_t=first_t, held=np.asarray(held, float),
                           panel_users=sorted(set(users)), n_train=int(tr.sum()))


def fit_item_model(mc, lam: float, dim: int | None = None, iters: int | None = None) -> dict:
    """forensics.fit_biased_mf at the regularisation weight lam on the training ratings; only the item side (mu, b_i, Q) is
    kept, with an empty user side of dim rows (fold_in_user reads only the item side)."""
    dim = MF_DIM if dim is None else dim
    iters = MF_ITERS if iters is None else iters
    mf = fx.fit_biased_mf(mc.tr_u, mc.tr_i, mc.tr_r, mc.n_u, mc.n_i, dim, iters, lam, MF_LAMBDA_BIAS, MF_SEED)
    return {"mu": mf["mu"], "bi": mf["bi"], "Q": mf["Q"], "P": np.zeros((dim, 0)), "train_rmse": mf["train_rmse"],
            "lam": float(lam), "dim": int(dim)}


def fold_in_panel(model: dict, mc, lam: float, window: int | None = None) -> SimpleNamespace:
    """Each panel user's (b_u, p_u) re-solved from the user's own non-candidate first events strictly before the user's first
    candidate (the registered fold-in; item parameters fixed), restricted to the last `window` of those events when given."""
    users = mc.panel_users
    dim = model["P"].shape[0]
    bu, P = np.zeros(len(users)), np.zeros((dim, len(users)))
    warm = np.zeros(len(users), bool)
    n_all, n_used = np.zeros(len(users), int), np.zeros(len(users), int)
    last_ids = []
    for g, u in enumerate(users):
        ft = mc.first_t.get(u, -math.inf)
        ev = [(i, r) for t, i, r in mc.scan["user_hist"].get(u, ()) if t < ft]
        n_all[g] = len(ev)
        if window is not None:
            ev = ev[-window:] if window > 0 else []
        n_used[g] = len(ev)
        last_ids.append([i for i, _ in ev])
        k = mc.scan["uid"].get(u)
        if k is None:
            continue
        idx = [mc.scan["iid"][i] for i, _ in ev]
        bu[g], P[:, g] = fx.fold_in_user(model, idx, [r for _, r in ev], lam, MF_LAMBDA_BIAS)
        warm[g] = any(mc.ni_t[j] > 0 for j in idx)
    return SimpleNamespace(bu=bu, P=P, warm=warm, n_all=n_all, n_used=n_used, last_ids=last_ids, window=window)


def pair_arrays(model: dict, mc, fold) -> dict:
    """Per EVAL pair: the MF score, personal residual p_u.q_i, item bias b_i, user bias and the warm flags of a model with the
    users' fold-in (forensics.mf_pair_scores on a model that holds the panel users only)."""
    gidx = {u: g for g, u in enumerate(mc.panel_users)}
    ug = np.array([gidx[u] for u in mc.users], np.int64)
    ui_f = np.where(fold.warm[ug], ug, -1)
    mf = {"mu": model["mu"], "bu": fold.bu, "bi": model["bi"], "P": fold.P, "Q": model["Q"]}
    s = fx.mf_pair_scores(mf, ui_f, mc.ii_t)
    return {"score": s["score"], "residual": s["residual"], "item_bias": s["item_bias"],
            "user_bias": np.where(ui_f >= 0, fold.bu[np.maximum(ui_f, 0)], 0.0), "user_warm": ui_f >= 0,
            "item_warm": mc.ii_t >= 0, "warm": (ui_f >= 0) & (mc.ii_t >= 0)}


def heldout_rmse(score, held, mask) -> tuple:
    """(RMSE, n) of the MF score against the withheld candidate ratings over the finite pairs of `mask`."""
    ok = np.asarray(mask, bool) & np.isfinite(held) & np.isfinite(score)
    return (float(np.sqrt(np.mean((score[ok] - held[ok]) ** 2))) if ok.any() else NAN), int(ok.sum())


def choose_weight(rmse_by_weight: dict) -> tuple:
    """(the weight, the tied weights): the minimiser of the held-out RMSE over the weights with a finite RMSE; weights whose
    RMSE is within TIE_TOL of the minimum are tied and the larger wins. (None, []) when no RMSE is finite."""
    fin = {w: r for w, r in rmse_by_weight.items() if math.isfinite(r)}
    if not fin:
        return None, []
    m = min(fin.values())
    tied = sorted(w for w, r in fin.items() if r <= m + TIE_TOL)
    return max(tied), tied


def _wkey(w: float) -> str:
    return format(float(w), "g")


def select_regularisation(cx, mc, weights=None, select_rows: str = "cal_eval_users", dim: int | None = None,
                          iters: int | None = None) -> tuple:
    """Fit the MF at every weight (item parameters kept), score the EVAL users' CAL pairs under the whole-history fold-in and
    choose the weight by held-out RMSE (warm CAL pairs; R6). Returns (selection record, {weight: item model}). The
    sensitivity table uses the CAL pairs of the EVAL users outside S_d."""
    weights = tuple(WEIGHT_GRID if weights is None else weights)
    models, rmse, rmse_out, n_all, n_out, train_rmse = {}, {}, {}, 0, 0, {}
    for w in weights:
        model = fit_item_model(mc, w, dim, iters)
        pa = pair_arrays(model, mc, fold_in_panel(model, mc, w, None))
        mask = pa["warm"] & cx.cal
        rmse[w], n_all = heldout_rmse(pa["score"], mc.held, mask)
        rmse_out[w], n_out = heldout_rmse(pa["score"], mc.held, mask & ~cx.sd)
        models[w] = model
        train_rmse[w] = model["train_rmse"]
        print(f"[ftgrid_refs]   lambda {w:g}: train RMSE {model['train_rmse']:.4f}, held-out CAL RMSE {rmse[w]:.4f} "
              f"({n_all} pairs)", flush=True)
    tables = {"cal_eval_users": (rmse, n_all), "cal_users_outside_sd": (rmse_out, n_out)}
    picks = {k: choose_weight(v[0]) for k, v in tables.items()}
    chosen, tied = picks[select_rows]
    sel = {"rule": "held-out RMSE of the withheld candidate ratings on the warm CAL pairs (candidate timestamp < T_d) under "
                   "the whole-history fold-in; the larger weight on ties; no TEST row is read",
           "select_rows": select_rows, "grid": [float(w) for w in weights],
           "chosen": chosen, "tied_weights": tied, "tie_tolerance": TIE_TOL,
           "train_rmse_by_weight": {_wkey(w): train_rmse[w] for w in weights},
           "tables": {k: {"rmse_by_weight": {_wkey(w): v[0][w] for w in weights}, "n_pairs": v[1],
                          "chosen": picks[k][0], "tied_weights": picks[k][1]} for k, v in tables.items()},
           "choice_agrees_across_selection_sets": picks["cal_eval_users"][0] == picks["cal_users_outside_sd"][0]}
    return sel, models


def cold_counts(cx, rows, pa: dict, item_only: bool = False) -> dict:
    """Counts of the TEST rows of EVAL users: warm / cold pairs, cold items and cold users (pairs and distinct)."""
    rows = np.asarray(rows, bool)
    cold_item = rows & ~pa["item_warm"]
    items, users = cx.item, cx.users
    out = {"n_pairs": int(rows.sum()), "n_pairs_warm": int((rows & pa["warm"]).sum()),
           "n_pairs_cold": int((rows & ~pa["warm"]).sum()), "n_pairs_cold_item": int(cold_item.sum()),
           "n_items": int(len(set(items[rows].tolist()))), "n_items_cold": int(len(set(items[cold_item].tolist()))),
           "n_users": int(len(set(users[rows].tolist())))}
    if not item_only:
        cold_user = rows & ~pa["user_warm"]
        out.update(n_pairs_cold_user=int(cold_user.sum()), n_pairs_cold_item_and_user=int((cold_item & cold_user).sum()),
                   n_users_cold=int(len(set(users[cold_user].tolist()))))
    out["share_pairs_cold"] = out["n_pairs_cold"] / out["n_pairs"] if out["n_pairs"] else NAN
    return out


# ---------------------------------------------------------------- UAUC levels and gains (the registered estimators)
def levels(cx, scores: dict, rows, n_boot: int, seed: int) -> dict:
    """UAUC records of the scores on identical rows and users (ftgrid_report.uauc_models: tie-aware per-user AUC, user-cluster
    percentile interval); `rows` is restricted to the rows finite in every score."""
    out = fr.uauc_models({}, cx.y, cx.uc, rows, n_boot, seed, extra=scores)
    return {"rows": out["rows"], "UAUC": out["references"]}


def subset_levels(cx, scores: dict, rows, warm, n_boot: int, seed: int) -> dict:
    """warm / cold / all-pair UAUC of the scores on `rows` (the EVAL users' TEST rows)."""
    rows, warm = np.asarray(rows, bool), np.asarray(warm, bool)
    return {"warm": levels(cx, scores, rows & warm, n_boot, seed), "cold": levels(cx, scores, rows & ~warm, n_boot, seed),
            "all": levels(cx, scores, rows, n_boot, seed)}


def _ed_user_gains(cx, base: list, aug: list, rows) -> tuple:
    """({user: AUC(aug stacker) - AUC(base stacker)}, rows): E-D's cross-fitted stacker (ftgrid_report.crossfit, fold A =
    user_halves(S_d, 0), one split) on `rows` restricted to the rows finite in every feature."""
    rows = np.asarray(rows, bool)
    for v in base + aug:
        rows = rows & np.isfinite(np.asarray(v, float))
    eb = fr.crossfit(np.column_stack(base), cx.y, cx.fold_a, rows)
    ea = fr.crossfit(np.column_stack(aug), cx.y, cx.fold_a, rows)
    ub, ua = fr.user_aucs(eb, cx.y, cx.uc, rows), fr.user_aucs(ea, cx.y, cx.uc, rows)
    return {u: ua[u] - ub[u] for u in set(ub) & set(ua)}, rows


def gain_ed(cx, base: list, aug: list, rows, n_boot: int, seed: int) -> dict:
    """dUAUC(aug - base) with E-D's cross-fitted stacker, per-user differences, user-cluster interval."""
    g, rows = _ed_user_gains(cx, base, aug, rows)
    keys = sorted(g)
    V = np.array([g[u] for u in keys], float)
    D = fr.mean_draws(V, n_boot, seed)[:, 0]
    return fr.rec(V.mean() if len(V) else NAN, D, len(V), fr._rows_of_users(cx.uc, rows, keys))


def gain_ed_models(cx, base: list, augs: dict, rows, n_reg: int, n_boot: int, seed: int) -> dict:
    """gain_ed for several models on identical rows and users, with the seed mean and its interval."""
    per = {m: _ed_user_gains(cx, base, aug, rows) for m, aug in augs.items()}
    keys = sorted(set.intersection(*(set(g) for g, _ in per.values()))) if per else []
    common = np.asarray(rows, bool).copy()
    for aug in augs.values():
        for v in base + aug:
            common &= np.isfinite(np.asarray(v, float))
    cols = fe.Cols(len(keys))
    for m in augs:
        cols.add(m, [per[m][0][u] for u in keys])
    cols.draws(n_boot, seed)
    return cols.models({m: m for m in augs}, fr._rows_of_users(cx.uc, common, keys), n_reg, False)


def gain_ew(cx, feats: dict, base_cols: tuple, aug_cols: tuple, rows, folds: list, n_boot: int, seed: int) -> dict:
    """dUAUC(aug - base) with E-W's estimator (ftgrid_extra.wu_fit: features centred over the user's rows, K = 20 splits,
    per-user differences averaged over the splits, user-cluster interval with the coefficients held fixed) on `rows`."""
    rows = np.asarray(rows, bool)
    for c in set(base_cols) | set(aug_cols):
        rows = rows & np.isfinite(np.asarray(feats[c], float))
    et = fe.wu_fit(cx, feats, {"base": base_cols, "aug": aug_cols}, rows, folds)
    keys, A = fe.auc_table({"base": et["base"], "aug": et["aug"]}, cx.y, cx.uc, rows)
    nu, npairs = len(keys), fr._rows_of_users(cx.uc, rows, keys)
    c = fe.Cols(nu)
    c.add("base", A["base"].mean(1))
    c.add("aug", A["aug"].mean(1))
    c.add("G", (A["aug"] - A["base"]).mean(1))
    c.draws(n_boot, seed)
    return {"rows": {"n_pairs": npairs, "n_users": nu, "K": len(folds)},
            "UAUC_base": c.one("base", npairs), "UAUC_aug": c.one("aug", npairs), "G": c.one("G", npairs)}


def gain_pair(cx, q, ref, rows, folds: list, n_boot: int, seed: int) -> dict:
    """The model-free G of a reference: dUAUC([m, r] - [m]) on S_d's TEST rows with a finite m and reference, E-D's and E-W's
    estimators (the E-W interval is what the labels read)."""
    q, ref = np.asarray(q, float), np.asarray(ref, float)
    rows = np.asarray(rows, bool) & np.isfinite(q) & np.isfinite(ref)
    ew = gain_ew(cx, {"q_hat": q, "r": ref}, ("q_hat",), ("q_hat", "r"), rows, folds, n_boot, seed)
    return {"rows": ew["rows"], "E_D": gain_ed(cx, [q], [q, ref], rows, n_boot, seed), "E_W": ew["G"],
            "UAUC_E_W": {"m": ew["UAUC_base"], "m_and_reference": ew["UAUC_aug"]}}


def regime_gains_mf(X, q, res_b, regimes: dict, folds: list) -> dict:
    """G_CF (E-D and E-W), G_LLM|CF and G_CF|LLM with the tuned residual (b), per regime, through the registered code:
    ftgrid_report.stacker_block (E-D) and ftgrid_extra.wu_regime (E-W, E-F) with the residual replaced."""
    cx, boot = X.cx, (X.n_boot, X.seed)
    Xb = SimpleNamespace(cx=cx, L=X.L, PI=X.PI, refs={"arrays": {"q_hat": q, "mf_residual": res_b}}, sd_ids=X.sd_ids,
                         n_boot=X.n_boot, seed=X.seed)
    out = {}
    for reg in REGS:
        info = regimes[reg]
        ew, ef, _ = fe.wu_regime(Xb, reg, info, folds, {}, {})
        if ew.get("available") is False:
            out[reg] = ew
            continue
        ms, nreg = info["models"], info["n_registered"]
        msw = [m for m in ms if m in X.PI]
        D = fe.ed_rows(cx, ms, msw, X.L, X.PI)
        st = fr.stacker_block(cx, msw, X.L, X.PI, q, res_b, D, *boot, nreg)
        m0 = st["per_model"][msw[0]]["UAUC"]
        out[reg] = descriptive({
            "models": ew["models"], "missing_or_excluded": ew["missing_or_excluded"], "complete": ew["complete"],
            "rows_E_D": st["rows"], "rows_E_W": ew["rows"],
            "G_CF_E_D": st["G_CF"], "UAUC_E_D": {"M0": m0["M0"], "M3": m0["M3"]},
            "G_CF_E_W": ew["G_CF_wu"], "UAUC_E_W": {"M0": ew["UAUC"]["M0"], "M3": ew["UAUC"]["M3"]},
            "G_LLM_given_CF": ef["G_LLM_given_CF"], "G_CF_given_LLM": ef["G_CF_given_LLM"], "UAUC_M4": ef["UAUC_M4"]})
    return out


def regime_gains_content(X, q, c, regimes: dict, folds: list) -> dict:
    """G_LLM|content = dUAUC([m, c, e-hat] - [m, c]) (the literal text) and, beside it, dUAUC([m, c, pi, e-hat] - [m, c]) (the
    E-F analogue with the item prior), per regime and model, E-W's estimator (primary) and E-D's."""
    cx, boot = X.cx, (X.n_boot, X.seed)
    out = {}
    for reg in REGS:
        info = regimes[reg]
        ms, nreg = info["models"], info["n_registered"]
        msw = [m for m in ms if m in X.PI]
        if not ms:
            out[reg] = _na(f"no usable like arm for the {reg} models {list(fr.REGIMES[reg])}")
            continue
        if not X.sd_ids or not msw:
            out[reg] = _na("no S_d" if not X.sd_ids else "no usable swap arm (pi) for the regime's models")
            continue
        D = fe.ed_rows(cx, ms, msw, X.L, X.PI)
        R = D & np.isfinite(q) & np.isfinite(c)
        feats = {m: {"q_hat": q, "c": c, "pi": X.PI[m], "e_hat": X.L[m] - X.PI[m]} for m in msw}
        shared = fe.wu_fit(cx, {"q_hat": q, "c": c}, {"Mc": ("q_hat", "c")}, R, folds)["Mc"]
        per = {m: fe.wu_fit(cx, feats[m], {"Mce": ("q_hat", "c", "e_hat"), "Mcpe": ("q_hat", "c", "pi", "e_hat")}, R, folds)
               for m in msw}
        sc = {("s", "Mc"): shared}
        sc.update({(m, s): per[m][s] for m in msw for s in ("Mce", "Mcpe")})
        keys, A = fe.auc_table(sc, cx.y, cx.uc, R)
        nu, npairs = len(keys), fr._rows_of_users(cx.uc, R, keys)
        cols = fe.Cols(nu)
        cols.add("UAUC_Mc", A[("s", "Mc")].mean(1))
        for m in msw:
            cols.add(("UAUC_Mce", m), A[(m, "Mce")].mean(1))
            cols.add(("UAUC_Mcpe", m), A[(m, "Mcpe")].mean(1))
            cols.add(("G_lit", m), (A[(m, "Mce")] - A[("s", "Mc")]).mean(1))
            cols.add(("G_pi", m), (A[(m, "Mcpe")] - A[("s", "Mc")]).mean(1))
        cols.draws(*boot)
        # E-D's estimator: one split (fold A), pooled cross-fitted stackers, identical rows for every model
        ed = {tag: gain_ed_models(cx, [q, c], {m: [feats[m][f] for f in aug] for m in msw}, R, nreg, *boot)
              for tag, aug in (("lit", ["q_hat", "c", "e_hat"]), ("pi", ["q_hat", "c", "pi", "e_hat"]))}
        blk = {"models": msw, "missing_or_excluded": list(info["missing_or_excluded"]) + [m for m in ms if m not in msw],
               "complete": len(msw) == nreg, "rows_E_W": {"n_pairs": npairs, "n_users": nu, "K": len(folds)},
               "UAUC_m_and_c_E_W": cols.one("UAUC_Mc", npairs),
               "UAUC_m_c_e_hat_E_W": cols.models({m: ("UAUC_Mce", m) for m in msw}, npairs, nreg, False),
               "UAUC_m_c_pi_e_hat_E_W": cols.models({m: ("UAUC_Mcpe", m) for m in msw}, npairs, nreg, False),
               "G_LLM_given_content_E_W": cols.models({m: ("G_lit", m) for m in msw}, npairs, nreg, False),
               "G_LLM_given_content_with_pi_E_W": cols.models({m: ("G_pi", m) for m in msw}, npairs, nreg, False),
               "G_LLM_given_content_E_D": ed["lit"], "G_LLM_given_content_with_pi_E_D": ed["pi"]}
        out[reg] = descriptive(blk)
    return out


# ---------------------------------------------------------------- labels (section 3)
def signal_label(rec: dict | None) -> dict:
    """SIGNAL_PRESENT if the interval of G (E-W) lies above 0 (lo > 0); else SIGNAL_ABSENT if it lies within [-0.01, +0.01];
    else INCONCLUSIVE (the order of the registered text; R15)."""
    rule = {"rule": "SIGNAL_PRESENT if the interval of G_r (E-W) lies above 0 (lo > 0); SIGNAL_ABSENT if it lies within "
                    "[-0.01, +0.01] (-0.01 <= lo and hi <= 0.01); otherwise INCONCLUSIVE; tested in that order",
            "band": SIGNAL_BAND}
    if not isinstance(rec, dict) or "est" not in rec:
        return {**rule, "available": False, "label": None, "reason": "G_r (E-W) was not computed"}
    lo, hi = _num(rec.get("lo")), _num(rec.get("hi"))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return {**rule, "available": False, "label": None, "reason": "the interval of G_r (E-W) is not finite"}
    present = lo > 0
    within = lo >= -SIGNAL_BAND and hi <= SIGNAL_BAND
    label = "SIGNAL_PRESENT" if present else "SIGNAL_ABSENT" if within else "INCONCLUSIVE"
    return {**rule, "available": True, "label": label, "est": rec["est"], "lo": lo, "hi": hi, "n_users": rec.get("n_users"),
            "descriptive_min_n": rec.get("descriptive_min_n"), "within_band": bool(within),
            "also_within_band": bool(present and within),
            "note": "a property of the panel and the reference, not of the LLM"}


# ---------------------------------------------------------------- item means (section 2.1)
def item_means_block(X, st, q, q_T, regimes: dict) -> dict:
    """m_TR, the registered m (q-hat) and the time-matched m_T beside it; the share of TEST pairs with a TRAIN example;
    dUAUC(l - m), dUAUC(l - m_T), dUAUC(l - m_TR) per regime on identical rows (contrast_models, descriptive)."""
    cx, boot = X.cx, (X.n_boot, X.seed)
    m_tr = m_tr_scores(cx.item.tolist(), st)
    refs = {"m_TR": m_tr}
    if q is not None:
        refs = {"m": q, **({"m_T": q_T} if q_T is not None else {}), "m_TR": m_tr}
    out = {"definition": "m_TR(i) = (sum of the labels of the TRAIN examples of i + 5 mu_TRAIN) / (n_i + 5), mu_TRAIN the TRAIN "
                         "like rate, an item without a TRAIN example scores mu_TRAIN; m = the registered q-hat, m_T = q-hat at "
                         "T_d (ftgrid_extra E-J); dUAUC(l - m*) paired per user on identical TEST rows",
           "k": TRAIN_SHRINK_K,
           "train": {"n_examples": st.n_examples, "n_items": len(st.n), "n_likes": st.n_pos, "mu_train": st.mu}}
    if cx.seen is not None:
        t, s = cx.test, cx.test & cx.sd
        out["share_test_pairs_with_train_example"] = {
            "eval_users": float(cx.seen[t].mean()) if t.any() else NAN, "n_pairs_eval_users": int(t.sum()),
            "S_d": float(cx.seen[s].mean()) if s.any() else NAN, "n_pairs_S_d": int(s.sum())}
    else:
        out["share_test_pairs_with_train_example"] = _na("no train.jsonl in --panels")
    R0 = cx.test.copy()
    for v in refs.values():
        R0 &= np.isfinite(v)
    out["UAUC_levels"] = levels(cx, refs, R0, *boot)
    for reg in REGS:
        info = regimes[reg]
        ms, nreg = info["models"], info["n_registered"]
        if not ms:
            out[reg] = _na(f"no usable like arm for the {reg} models {list(fr.REGIMES[reg])}")
            continue
        R = R0.copy()
        for m in ms:
            R &= np.isfinite(X.L[m])
        blk = {"models": ms, "missing_or_excluded": info["missing_or_excluded"], "complete": len(ms) == nreg,
               "share_test_pairs_with_train_example_on_rows": (float(cx.seen[R].mean()) if cx.seen is not None and R.any()
                                                                else NAN)}
        for name, arr in refs.items():
            blk[f"dUAUC_l_minus_{name}"] = fr.contrast_models({m: (X.L[m], arr) for m in ms}, cx.y, cx.uc, R, *boot,
                                                              n_registered=nreg)
        out[reg] = descriptive(blk)
    return descriptive(out)


# <<END-OF-PART-3>>
