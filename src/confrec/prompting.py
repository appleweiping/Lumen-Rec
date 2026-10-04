"""Prompt bank (amendment 2 G1), rendering, chat tokenization and answer-token reading shared by `pyes_scorer` and
`train_lora_yesno`.

Both paths render a (system, user) pair with `render` and tokenize it with `chat_ids`, so the token ids vLLM scores
are exactly the ids a LoRA adapter was trained on (amendment 1, C0):

    from src.confrec.prompting import chat_ids, read_yes_no, render, yes_no_ids
    system, user = render(rec, i, "like", variant="V3")      # rec: one panel row, i: candidate index
    ids = chat_ids(tok, user, system)                          # system None -> one user message, as before
    yes_ids, no_ids = yes_no_ids(tok)                          # resolved once per tokenizer, logged by the scorer
    lp_yes, lp_no, mass, censored = read_yes_no(vllm_output.outputs[0].logprobs[0], yes_ids, no_ids)

VARIANTS (strings verbatim from idea-stage/deliberation_2026-10-02/gatefix_prompt_variants_v1.json; V6 dropped by
amendment 2 G1). Every variant has a rated and a next-item rendering; next-item stays frozen at V0 by policy (G0).
    V0  registered Pilot-1 prompt, byte-identical to `build_prompt` (rated: last 10 rated lines; next-item: last 5)
    V1  V0 with the threshold question family ("Will this user rate the candidate item 4 stars or higher ...")
    V2  TALLRec layout: the same 10 events as liked (rating > 3) / disliked (rating <= 3) title lists, an empty
        list renders "- (none)"; next-item renders as V0 (declared)
    V3  last 20 history events
    V4  history lines "- {title} [{meta}] (rated r/5)", meta = row field history_meta ("Genres: ..." for ML-1M,
        "Categories: " + categories[:80] for Amazon; an empty meta drops the brackets)
    V5  system message (rated_movie | rated_product by domain_kind; next_product on next-item panels) and the user
        template without the persona line
    V7  V1 + V3 + V4 + V5
    T0_probe  diagnosis-battery item-only probe, no history and no system message:
        "Movie: {title}{desc}\\n\\nIs this movie widely considered good? Answer with only Yes or No." ("Product" /
        "product" unless domain_kind is movie); question keys "like" or "quality".
panel_kind: "rated" when a row carries candidate_labels and history_ratings, else "next_item" (auto-detected per row
when None). domain_kind: row field "movie" | "product", else "movie" iff source == "ml1m".
History length: hist_len=None renders the variant's registered window (`resolve_hist_len`); an explicit value is
accepted for V0 (Pilot-1 behaviour, any value) and, for every variant, 0 (no-history arm) or the registered value.
Optional per-candidate field candidate_extras (list[str], "" = none) adds the line "\\n{extra}" right after the
description line in every variant (battery T1/T2); rows without it (absent, None or []) render byte-identically to
before. Misaligned candidate lists raise ValueError (`candidate_fields`). `chat_ids` refuses a chat template that
drops a message or leaves thinking on.
readout="digits" (battery T3 / T0 E[r] only: variants V0 and T0_probe, `DIGIT_VARIANTS`; any gate variant other than
the V0 control raises, amendment 2 G0) replaces the question and " Answer with only Yes or No." by a 1-5 star
question and " Answer with only one digit from 1 to 5." (question keys "like" or "rating"); `digit_ids` /
`read_digits` read the answer. The digit wording is not in the variants JSON (OPERATIONALIZATIONS).
`prompt_strings()` / PROMPT_STRINGS_SHA1: every fixed string and rule of this bank (questions, templates, system
messages, T0 and digit wording, caps, OPERATIONALIZATIONS) as one canonical JSON / sha1, for the freeze record.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

QUESTIONS = {  # like family = the registered Pilot-1 questions (V0)
    "next": "Will this user purchase the candidate item next?",
    "like": "Would this user like the candidate item?",
    "dislike": "Would this user dislike the candidate item?",
    "like_para": "Is the candidate item a good match for this user's taste?",
    "dislike_para": "Would this user be disappointed by the candidate item?",
}
THRESHOLD_QUESTIONS = {  # label-aligned (V1, V7): registered label >= 4 like, <= 2 dislike
    "next": "Will this user purchase the candidate item next?",
    "like": "Will this user rate the candidate item 4 stars or higher (on a 1-5 scale)?",
    "dislike": "Will this user rate the candidate item 2 stars or lower (on a 1-5 scale)?",
    "like_para": "Will this user's star rating for the candidate item be 4 or 5?",
    "dislike_para": "Will this user's star rating for the candidate item be 1 or 2?",
}
QUESTION_FAMILIES = {"like_family": QUESTIONS, "threshold_family": THRESHOLD_QUESTIONS}
BODY = ("You are an expert recommendation system.\n\n"
        "User history (oldest to newest):\n{hist}\n\n"
        "Candidate item:\nTitle: {title}{desc}\n\n"
        "{question} Answer with only Yes or No.")
YESNO_TAIL = " Answer with only Yes or No."
USER_TEMPLATES = {
    "T_chrono": ("You are an expert recommendation system.\n\nUser history (oldest to newest):\n{hist_lines}\n\n"
                 "Candidate item:\nTitle: {cand_title}{cand_desc}\n\n{question} Answer with only Yes or No."),
    "T_chrono_nopersona": ("User history (oldest to newest):\n{hist_lines}\n\n"
                           "Candidate item:\nTitle: {cand_title}{cand_desc}\n\n{question} Answer with only Yes or No."),
    "T_split": ("You are an expert recommendation system.\n\n"
                "Items this user liked (rated 4-5 stars), oldest to newest:\n{liked_lines}\n\n"
                "Items this user disliked (rated 1-3 stars), oldest to newest:\n{disliked_lines}\n\n"
                "Candidate item:\nTitle: {cand_title}{cand_desc}\n\n{question} Answer with only Yes or No."),
}
SYSTEM_MESSAGES = {
    "rated_movie": ("You are a movie recommender. You predict how a specific user will rate movies from that user's "
                    "past ratings. Ratings range from 1 to 5 stars; 4-5 stars means the user liked the movie and 1-2 "
                    "stars means the user disliked it."),
    "rated_product": ("You are a product recommender. You predict how a specific user will rate products from that "
                      "user's past ratings. Ratings range from 1 to 5 stars; 4-5 stars means the user liked the "
                      "product and 1-2 stars means the user disliked it."),
    "next_product": ("You are a product recommender. You predict a specific user's preferences from that user's "
                     "purchase history."),
}
T0_TEMPLATE = "{Noun}: {cand_title}{cand_desc}\n\n{question} Answer with only Yes or No."
T0_QUESTION = "Is this {noun} widely considered good?"
T0_KEYS = ("like", "quality")
# digit readout (diagnosis battery T3 and the T0 E[r] reading only; amendment 2 G0 bars it from every gate variant)
DIGIT_TAIL = " Answer with only one digit from 1 to 5."
DIGIT_QUESTION = "What star rating from 1 to 5 would this user give the candidate item?"
DIGIT_T0_QUESTION = "On a 1-5 star scale, how good is this {noun} widely considered to be?"
DIGIT_KEYS = ("like", "rating")
DIGIT_VARIANTS = ("V0", "T0_probe")   # battery T3 (V0 control) and the T0 E[r] reading; never a selectable gate variant
READOUTS = ("yesno", "digits")

NO_HISTORY = "- (no history available)"
EMPTY_SPLIT = "- (none)"
MAX_TITLE_CHARS = 200
MAX_DESC_CHARS = 200
MAX_CATEGORY_CHARS = 80
CATEGORIES_PREFIX = "Categories: "
SYSTEM_SEP = "\x1d"   # prompt key hashed by the scorer: system + SYSTEM_SEP + user when a system message exists
PANEL_KINDS = ("rated", "next_item")
DOMAIN_KINDS = ("movie", "product")
SHORT_HISTORY = 10   # Pilot-1 rated panels (build_rated_panels default) hold at most 10 history events per row
LINE_FORMATS = {  # history line formats of the variants JSON; meta is `render_meta` of the row's history_meta entry
    "L_rated": "- {title} (rated {r}/5)",
    "L_rated_meta": "- {title} [{meta}] (rated {r}/5)",
    "L_plain": "- {title}",
    "L_plain_meta": "- {title} [{meta}]",
    "L_split": "- {title}",
}
# Rules this module applies that the variants JSON does not state in so many words (logged with the freeze; part of
# PROMPT_STRINGS_SHA1, so changing one changes the recorded sha1).
OPERATIONALIZATIONS = (
    "meta: an empty history meta (no genres / no categories, or 'Categories: ' with an empty body) drops the "
    "' [...]' brackets, so the line renders as L_rated / L_plain",
    "meta: the body after 'Categories: ' is capped at 80 characters at render time (idempotent on build_rated_panels "
    "output); other meta (ML-1M 'Genres: ...') is printed as stored",
    "titles: capped at 200 characters in every history line format (L_rated, L_plain, L_split, L_rated_meta, "
    "L_plain_meta) and in the candidate title; a rated line keeps its ' (rated r/5)' suffix",
    "L_split: the rating is read from the ' (rated r/5)' suffix of each of the last hist_len events (must agree with "
    "history_ratings); liked = r > 3, disliked = r <= 3; an empty list, incl. both lists at hist_len 0, renders "
    "'- (none)'",
    "hist_len 0 (no-history arm): chronological templates render '- (no history available)' as in Pilot 1",
    "system wording: rated_movie iff the row's domain_kind is 'movie' (absent: source == 'ml1m'), else rated_product; "
    "next-item panels use next_product",
    "T0_probe: '{Noun}: {title}{desc}\\n\\nIs this {noun} widely considered good? Answer with only Yes or No.', "
    "noun movie | product by domain_kind, desc as in V0, no history and no system message; question keys like | "
    "quality render the same prompt",
    "digit readout (diagnosis battery T3 on V0 and the T0 E[r] reading only; implementer wording, not in the "
    "variants JSON): DIGIT_QUESTION / DIGIT_T0_QUESTION replace the question and DIGIT_TAIL replaces ' Answer with "
    "only Yes or No.'",
    "candidate_extras (battery T1 / T2): a non-empty entry adds the line '\\n{extra}' right after the description "
    "line (after the title line when there is no description) in every variant; absent, empty or '' entries render "
    "byte-identically to before",
    "next-item panels: every variant other than V0 is rendered for completeness only; the next-item prompt stays "
    "frozen at V0 (amendment 2 G0, F)",
)
_RATED = re.compile(r"^(.*)( \(rated (\d+)/5\))$", re.S)  # rated-panel history line "<title> (rated r/5)"


@dataclass(frozen=True)
class Spec:
    """One rendering of a variant. system: None | "rated" (rated_movie / rated_product by domain_kind) |
    "next_product"; user: USER_TEMPLATES key or "T0"; line: history line format; hist: registered last-k window;
    q: QUESTION_FAMILIES key (None for T0)."""
    system: str | None
    user: str
    line: str | None
    hist: int
    q: str | None


def _both(rated: Spec, next_item: Spec) -> dict:
    return {"rated": rated, "next_item": next_item}


_V0_NEXT = Spec(None, "T_chrono", "L_plain", 5, "like_family")
_T0 = Spec(None, "T0", None, 0, None)
VARIANTS = {
    "V0": _both(Spec(None, "T_chrono", "L_rated", 10, "like_family"), _V0_NEXT),
    "V1": _both(Spec(None, "T_chrono", "L_rated", 10, "threshold_family"),
                Spec(None, "T_chrono", "L_plain", 5, "threshold_family")),
    "V2": _both(Spec(None, "T_split", "L_split", 10, "like_family"), _V0_NEXT),
    "V3": _both(Spec(None, "T_chrono", "L_rated", 20, "like_family"),
                Spec(None, "T_chrono", "L_plain", 20, "like_family")),
    "V4": _both(Spec(None, "T_chrono", "L_rated_meta", 10, "like_family"),
                Spec(None, "T_chrono", "L_plain_meta", 5, "like_family")),
    "V5": _both(Spec("rated", "T_chrono_nopersona", "L_rated", 10, "like_family"),
                Spec("next_product", "T_chrono_nopersona", "L_plain", 5, "like_family")),
    "V7": _both(Spec("rated", "T_chrono_nopersona", "L_rated_meta", 20, "threshold_family"),
                Spec("next_product", "T_chrono_nopersona", "L_plain_meta", 20, "threshold_family")),
    "T0_probe": _both(_T0, _T0),   # diagnostic (battery T0), not a gate variant
}
GATE_VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "V7")
SIMPLICITY_ORDER = ("V0", "V1", "V5", "V3", "V4", "V2", "V7")   # G5 tie-break


def prompt_strings() -> dict:
    """Every fixed string and rule of the bank (incl. the digit-readout and T0 wording, which the yes/no freeze bank
    never renders), as plain JSON data (a copy: mutating it leaves the bank untouched)."""
    return json.loads(json.dumps({
        "question_families": QUESTION_FAMILIES, "user_templates": USER_TEMPLATES, "system_messages": SYSTEM_MESSAGES,
        "pilot1_body": BODY, "yesno_tail": YESNO_TAIL, "line_formats": LINE_FORMATS,
        "t0": {"template": T0_TEMPLATE, "question": T0_QUESTION, "keys": list(T0_KEYS)},
        "digits": {"question": DIGIT_QUESTION, "t0_question": DIGIT_T0_QUESTION, "tail": DIGIT_TAIL,
                   "keys": list(DIGIT_KEYS), "variants": list(DIGIT_VARIANTS)},
        "no_history": NO_HISTORY, "empty_split": EMPTY_SPLIT, "categories_prefix": CATEGORIES_PREFIX,
        "caps": {"title": MAX_TITLE_CHARS, "description": MAX_DESC_CHARS, "categories": MAX_CATEGORY_CHARS},
        "short_history": SHORT_HISTORY,
        "variants": {v: {k: asdict(s) for k, s in kinds.items()} for v, kinds in VARIANTS.items()},
        "simplicity_order": list(SIMPLICITY_ORDER), "operationalizations": list(OPERATIONALIZATIONS),
    }))


# sha1 of the canonical JSON of `prompt_strings()` (sorted keys, UTF-8): record it with the freeze; pyes_scorer
# writes it into every report.json
PROMPT_STRINGS_SHA1 = hashlib.sha1(json.dumps(prompt_strings(), sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()


def cap_history_line(h, max_title_chars: int = MAX_TITLE_CHARS) -> str:
    """Cap the title part of a history line; a rated line keeps its " (rated r/5)" suffix."""
    h = str(h)
    m = _RATED.match(h)
    if m:
        return m.group(1)[:max_title_chars] + m.group(2)
    return h[:max_title_chars]


def build_prompt(history: list[str], title: str, text: str, question: str, hist_len: int,
                 max_title_chars: int = MAX_TITLE_CHARS) -> str:
    """The registered Pilot-1 prompt (V0 user message). Kept verbatim; `render(..., variant="V0")` equals it."""
    hist = history[-hist_len:] if hist_len > 0 else []
    hist_block = ("\n".join(f"- {cap_history_line(h, max_title_chars)}" for h in hist) if hist
                  else "- (no history available)")
    meta = text[:200] if text else ""
    desc = f"\nDescription: {meta}" if meta else ""
    return BODY.format(hist=hist_block, title=str(title)[:max_title_chars], desc=desc,
                       question=QUESTIONS[question])


# ---------------------------------------------------------------- variant bank
def panel_kind_of(rec: dict) -> str:
    return "rated" if "candidate_labels" in rec and "history_ratings" in rec else "next_item"


def domain_kind_of(rec: dict) -> str:
    k = rec.get("domain_kind")
    if k is None:
        src = rec.get("source")
        if not src:
            raise ValueError(f"{rec.get('source_event_id')}: row has neither domain_kind nor source, so the "
                             "movie/product wording is undefined")
        k = "movie" if src == "ml1m" else "product"
    if k not in DOMAIN_KINDS:
        raise ValueError(f"{rec.get('source_event_id')}: domain_kind {k!r} not in {DOMAIN_KINDS}")
    return k


def spec_of(variant: str, panel_kind: str) -> Spec:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {list(VARIANTS)} (V6 is dropped by "
                         "amendment 2 G1)")
    if panel_kind not in PANEL_KINDS:
        raise ValueError(f"panel_kind {panel_kind!r} not in {PANEL_KINDS}")
    return VARIANTS[variant][panel_kind]


def resolve_hist_len(variant: str, panel_kind: str, hist_len: int | None = None) -> int:
    """History events rendered: the variant's registered window when hist_len is None; an explicit value only for
    V0 (any value, the Pilot-1 behaviour) or when it is 0 (no-history arm) or the registered window. T0 has none."""
    spec = spec_of(variant, panel_kind)
    if spec.user == "T0":
        return 0
    if hist_len is None:
        return spec.hist
    hist_len = int(hist_len)
    if hist_len < 0:
        raise ValueError(f"hist_len must be >= 0, got {hist_len}")
    if variant == "V0" or hist_len in (0, spec.hist):
        return hist_len
    raise ValueError(f"{variant} registers the last {spec.hist} history events on {panel_kind} panels; hist_len "
                     f"{hist_len} would render an unregistered prompt (only V0 takes any hist_len; 0 = no history)")


def short_history_error(variant: str, panel_kind: str, hist_len: int, max_hist: int) -> str | None:
    """Why rendering `variant` with an effective window of hist_len events on a panel whose longest row holds
    max_hist events would be a silent no-op (a non-V0 rated rendering asking for more than SHORT_HISTORY events of a
    panel built with SHORT_HISTORY: it would equal the 10-event rendering), else None. Shared by pyes_scorer and
    train_lora_yesno so a G9 adapter is never trained on fewer events than its CONFIRM evaluation renders."""
    if panel_kind == "rated" and variant != "V0" and hist_len > SHORT_HISTORY and max_hist <= SHORT_HISTORY:
        return (f"variant {variant} renders the last {hist_len} history events, but no row of the panel holds more "
                f"than {max_hist}: it would equal the {SHORT_HISTORY}-event rendering. Build the panel with "
                "build_rated_panels --hist_len 20")
    return None


def question_keys(variant: str, readout: str = "yesno") -> tuple[str, ...]:
    """Question keys `render` accepts for a variant and readout. The digit readout exists only for DIGIT_VARIANTS
    (amendment 2 G0 bars it from every gate variant; the battery reads it on the V0 control and on T0_probe)."""
    spec = spec_of(variant, "rated")
    if readout not in READOUTS:
        raise ValueError(f"readout {readout!r} not in {READOUTS}")
    if readout == "digits":
        if variant not in DIGIT_VARIANTS:
            raise ValueError(f"readout 'digits' is defined only for {DIGIT_VARIANTS} (diagnosis battery T3 / T0); "
                             f"amendment 2 G0 bars a digit readout from {variant}")
        return DIGIT_KEYS
    return T0_KEYS if spec.user == "T0" else tuple(QUESTION_FAMILIES[spec.q])


def render_meta(meta) -> str:
    """History meta as printed inside "[...]": Amazon "Categories: " strings keep the first 80 characters of the
    categories (idempotent on builder output); "" (no meta) drops the brackets."""
    m = "" if meta is None else str(meta)
    if m.startswith(CATEGORIES_PREFIX):
        body = m[len(CATEGORIES_PREFIX):][:MAX_CATEGORY_CHARS]
        return CATEGORIES_PREFIX + body if body.strip() else ""
    return m


def _rated_parts(h, rec: dict) -> re.Match:
    m = _RATED.match(str(h))
    if not m:
        raise ValueError(f"{rec.get('source_event_id')}: rated history line without a ' (rated r/5)' suffix: {h!r}")
    return m


class RowRenderer:
    """One row's prompt context under a variant: the system message and the history block are rendered once, then
    `render(title, text, question, extra)` gives the (system, user) pair for any candidate (the row's own, or a
    foreign item for the swap prior)."""

    def __init__(self, rec: dict, variant: str = "V0", panel_kind: str | None = None, *, hist_len: int | None = None,
                 readout: str = "yesno"):
        self.variant, self.readout = variant, readout
        self.kind = panel_kind or panel_kind_of(rec)
        self.spec = spec = spec_of(variant, self.kind)
        keys = question_keys(variant, readout)
        noun = domain_kind_of(rec) if spec.system == "rated" or spec.user == "T0" else None
        self.system = (None if spec.system is None else
                       SYSTEM_MESSAGES[f"rated_{noun}"] if spec.system == "rated" else SYSTEM_MESSAGES[spec.system])
        self.hist_len = resolve_hist_len(variant, self.kind, hist_len)
        if spec.user == "T0":
            tpl = T0_TEMPLATE
            self.fields = {"Noun": noun.capitalize()}
            qtext = DIGIT_T0_QUESTION.format(noun=noun) if readout == "digits" else T0_QUESTION.format(noun=noun)
            self.questions = {k: qtext for k in keys}
        else:
            tpl = USER_TEMPLATES[spec.user]
            self.fields = self._history_fields(rec)
            self.questions = ({k: DIGIT_QUESTION for k in keys} if readout == "digits"
                              else QUESTION_FAMILIES[spec.q])
        assert tpl.endswith("{question}" + YESNO_TAIL), tpl
        self.tpl = tpl if readout == "yesno" else tpl[:-len(YESNO_TAIL)] + DIGIT_TAIL

    def _history_fields(self, rec: dict) -> dict:
        spec, n = self.spec, self.hist_len
        history = rec["history"]
        hist = list(history[-n:]) if n > 0 else []
        if spec.line in ("L_rated_meta", "L_plain_meta"):
            hm = rec.get("history_meta")
            if hm is None:
                raise ValueError(f"{rec.get('source_event_id')}: variant {self.variant} needs the row field "
                                 "history_meta (rebuild the panel with build_rated_panels)")
            if len(hm) != len(history):
                raise ValueError(f"{rec.get('source_event_id')}: history_meta has {len(hm)} entries, history "
                                 f"{len(history)}")
            metas = [render_meta(x) for x in (hm[-n:] if n > 0 else [])]
            mp = [f" [{m}]" if m else "" for m in metas]
        if spec.line == "L_split":
            hr = rec.get("history_ratings")
            if hr is not None and len(hr) != len(history):
                raise ValueError(f"{rec.get('source_event_id')}: history_ratings has {len(hr)} entries, history "
                                 f"{len(history)}")
            hr = list(hr[-n:]) if hr is not None and n > 0 else None
            liked, disliked = [], []
            for k, h in enumerate(hist):
                m = _rated_parts(h, rec)
                r = int(m.group(3))
                if hr is not None and float(hr[k]) != r:
                    raise ValueError(f"{rec.get('source_event_id')}: history line {h!r} disagrees with "
                                     f"history_ratings {hr[k]!r}")
                (liked if r > 3 else disliked).append(f"- {m.group(1)[:MAX_TITLE_CHARS]}")
            return {"liked_lines": "\n".join(liked) if liked else EMPTY_SPLIT,
                    "disliked_lines": "\n".join(disliked) if disliked else EMPTY_SPLIT}
        if spec.line in ("L_rated", "L_plain"):
            lines = [f"- {cap_history_line(h)}" for h in hist]
        elif spec.line == "L_rated_meta":
            lines = []
            for h, p in zip(hist, mp):
                m = _rated_parts(h, rec)
                lines.append(f"- {m.group(1)[:MAX_TITLE_CHARS]}{p}{m.group(2)}")
        elif spec.line == "L_plain_meta":
            lines = [f"- {cap_history_line(h)}{p}" for h, p in zip(hist, mp)]
        else:
            raise AssertionError(spec.line)
        return {"hist_lines": "\n".join(lines) if lines else NO_HISTORY}

    def render(self, title, text, question: str, extra: str = "") -> tuple[str | None, str]:
        try:
            qtext = self.questions[question]
        except KeyError:
            raise ValueError(f"question {question!r} is not defined for {self.variant} (readout {self.readout}); "
                             f"choose from {list(self.questions)}") from None
        meta = text[:MAX_DESC_CHARS] if text else ""
        desc = f"\nDescription: {meta}" if meta else ""
        if extra:
            desc += f"\n{extra}"
        return self.system, self.tpl.format(cand_title=str(title)[:MAX_TITLE_CHARS], cand_desc=desc,
                                            question=qtext, **self.fields)


def candidate_fields(rec: dict) -> tuple[list, list, list]:
    """(titles, texts, extras) of a row's candidates. An absent or empty candidate_texts / candidate_extras means
    none ("" each, as Pilot 1 rendered rows without texts); otherwise each list, and candidate_item_ids when present,
    must align with candidate_titles (ValueError, also under python -O)."""
    titles = list(rec["candidate_titles"])
    n = len(titles)
    texts = list(rec.get("candidate_texts") or [""] * n)
    extras = list(rec.get("candidate_extras") or [""] * n)
    ids = rec.get("candidate_item_ids")
    lens = {"candidate_titles": n, "candidate_texts": len(texts), "candidate_extras": len(extras)}
    if ids is not None:
        lens["candidate_item_ids"] = len(ids)
    if len(set(lens.values())) != 1:
        raise ValueError(f"{rec.get('source_event_id')}: misaligned candidate lists {lens}")
    return titles, texts, extras


def render(rec: dict, i: int, question: str, variant: str = "V0", panel_kind: str | None = None, *,
           hist_len: int | None = None, readout: str = "yesno") -> tuple[str | None, str]:
    """(system message or None, user message) for candidate i of panel row rec."""
    titles, texts, extras = candidate_fields(rec)
    return RowRenderer(rec, variant, panel_kind, hist_len=hist_len, readout=readout).render(
        titles[i], texts[i], question, extras[i])


def render_record(rec: dict, questions, variant: str = "V0", panel_kind: str | None = None, *,
                  hist_len: int | None = None, readout: str = "yesno") -> list[tuple[int, str, str | None, str]]:
    """[(cand_idx, question, system, user)] for every candidate of a row, questions inner (a candidate's prompts
    are adjacent, so they share the vLLM prefix cache)."""
    if rec.get("candidate_item_ids") is None:
        raise ValueError(f"{rec.get('source_event_id')}: row has no candidate_item_ids")
    titles, texts, extras = candidate_fields(rec)
    row = RowRenderer(rec, variant, panel_kind, hist_len=hist_len, readout=readout)
    return [(i, q, *row.render(title, text, q, extra or ""))
            for i, (title, text, extra) in enumerate(zip(titles, texts, extras)) for q in questions]


class ChatPrompt(str):
    """A rendered user message (the str value, exactly what the V0 prompt always was) that carries its system
    message. `chat_ids(tok, p)` adds the system message; `key` is what the scorer hashes (user, or
    system + "\\x1d" + user)."""

    def __new__(cls, user: str, system: str | None = None):
        obj = super().__new__(cls, user)
        obj.system = system
        return obj

    @property
    def key(self) -> str:
        return str(self) if self.system is None else f"{self.system}{SYSTEM_SEP}{self}"


def prompt_key(p) -> str:
    return p.key if isinstance(p, ChatPrompt) else p


# ---------------------------------------------------------------- tokenization and answer reading
def chat_ids(tok, user: str, system: str | None = None) -> list[int]:
    """Chat-template [system,] user (thinking off) and tokenize without extra special tokens. system=None takes
    `user.system` for a ChatPrompt, else no system message (a single user message, the Pilot-1 path).

    The template already inserts BOS where the model needs it; tokenizing its text with the default
    add_special_tokens=True (what vLLM does for a text prompt) gives Llama-3 a second BOS.
    Every message's content (stripped: Llama-3.1 trims it) must appear verbatim in the templated text, so a template
    that drops or rewrites the system message (V5 / V7) fails at the first prompt instead of silently scoring or
    training the system-less prompt (ValueError). Thinking off (amendment 2 G0): when the tokenizer's chat template
    knows `enable_thinking` (Qwen3), the generation prompt must end with the closed empty think block
    "<think>\\n\\n</think>\\n\\n" the template emits for enable_thinking=False (ValueError otherwise)."""
    if system is None:
        system = getattr(user, "system", None)
    msg = [{"role": "user", "content": str(user)}]
    if system is not None:
        msg.insert(0, {"role": "system", "content": str(system)})
    try:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for m in msg:
        if m["content"].strip() not in text:
            raise ValueError(f"the chat template dropped or rewrote the {m['role']} message (its content is not in "
                             f"the templated text): {m['content'][:80]!r}")
    tmpl = getattr(tok, "chat_template", None)
    if isinstance(tmpl, str) and "enable_thinking" in tmpl and not text.rstrip().endswith("</think>"):
        raise ValueError("the chat template supports thinking but the generation prompt does not end with a closed "
                         f"empty think block (thinking would be on; G0 requires it off): ...{text[-60:]!r}")
    return list(tok(text, add_special_tokens=False)["input_ids"])


def yes_no_ids(tok) -> tuple[frozenset, frozenset]:
    """All vocab ids whose decoded text, stripped and lower-cased, is "yes" / "no" (one scan of the vocab)."""
    yes, no = set(), set()
    for i in range(len(tok)):
        s = tok.decode([i]).strip().lower()
        if s == "yes":
            yes.add(i)
        elif s == "no":
            no.add(i)
    return frozenset(yes), frozenset(no)


def digit_ids(tok) -> dict[int, frozenset]:
    """{r: all vocab ids whose decoded text, stripped, is the digit r} for r = 1..5 (one scan of the vocab)."""
    out: dict[int, set] = {r: set() for r in range(1, 6)}
    for i in range(len(tok)):
        s = tok.decode([i]).strip()
        if len(s) == 1 and s in "12345":
            out[int(s)].add(i)
    return {r: frozenset(v) for r, v in out.items()}


def _logsumexp(v: list[float]) -> float:
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def read_yes_no(top: dict, yes_ids, no_ids) -> tuple[float, float, float, int]:
    """(lp_yes, lp_no, mass, censored) from a top-k dict {token_id: obj with .logprob}.

    lp_yes / lp_no = log of the summed probability of the yes / no ids present (finite logprobs only);
    mass = P(yes) + P(no) over the present ids. censored: 0 both sides present; 1 one side absent, imputed with
    the smallest returned logprob `floor` (amendment 1, C0); 2 both absent -> (nan, nan, 0, 2).
    For censored=1 the floor upper-bounds each absent id, not the absent side's sum: that side's log probability is
    <= floor + log(n ids of that side), and the present side sums only its ids in the top-k. So the true logit is
    >= lp_yes - floor - log|no_ids| when No is absent, and <= floor + log|yes_ids| - lp_no when Yes is absent."""
    ys, ns, seen = [], [], []
    for tid, lp in top.items():
        v = float(lp.logprob)
        if not math.isfinite(v):
            continue
        seen.append(v)
        if tid in yes_ids:
            ys.append(v)
        elif tid in no_ids:
            ns.append(v)
    if not ys and not ns:
        return math.nan, math.nan, 0.0, 2
    mass = sum(math.exp(v) for v in ys + ns)
    floor = min(seen)
    return (_logsumexp(ys) if ys else floor, _logsumexp(ns) if ns else floor, mass, 0 if ys and ns else 1)


def read_digits_full(top: dict, digit_id_sets: dict) -> tuple[float, float, float, int, float]:
    """(lp_45, lp_12, mass, censored, exp_rating) from a top-k dict, digit_id_sets = `digit_ids(tok)`.

    lp_r = log of the summed probability of digit r's ids present (finite logprobs only); a digit with no id in the
    top-k is imputed with the smallest returned logprob `floor`, as a missing Yes/No side (amendment 1 C0), and the
    row is censored=1; no digit at all -> censored=2 and NaNs. lp_45 = log(P4 + P5), lp_12 = log(P1 + P2);
    exp_rating = sum_r r P_r / sum_r P_r over the 5 (imputed) digits; mass = summed probability of the present
    digit ids."""
    per: dict[int, list] = {r: [] for r in range(1, 6)}
    seen = []
    for tid, lp in top.items():
        v = float(lp.logprob)
        if not math.isfinite(v):
            continue
        seen.append(v)
        for r, ids in digit_id_sets.items():
            if tid in ids:
                per[int(r)].append(v)
                break
    if not any(per.values()):
        return math.nan, math.nan, 0.0, 2, math.nan
    mass = sum(math.exp(v) for vs in per.values() for v in vs)
    floor = min(seen)
    lp = {r: _logsumexp(vs) if vs else floor for r, vs in per.items()}
    m = max(lp.values())
    w = {r: math.exp(v - m) for r, v in lp.items()}
    exp_rating = sum(r * x for r, x in w.items()) / sum(w.values())
    censored = 0 if all(per.values()) else 1
    return _logsumexp([lp[4], lp[5]]), _logsumexp([lp[1], lp[2]]), mass, censored, exp_rating


def read_digits(top: dict, digit_id_sets: dict) -> tuple[float, float, float, int]:
    """(exp_rating, logit_45_vs_12 = log P(4|5) - log P(1|2), mass, censored); see `read_digits_full`."""
    lp45, lp12, mass, censored, er = read_digits_full(top, digit_id_sets)
    return er, lp45 - lp12, mass, censored
