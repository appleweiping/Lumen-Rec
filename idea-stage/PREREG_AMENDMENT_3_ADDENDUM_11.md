# Pre-registration amendment 3, addendum 11 (2026-10-06): cheap-signal controls for selective serving (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the zero-shot
next-item audits of Toys and Sports (Qwen3-8B, registered prompt V0) had been read, including their selective-serving block (signals p_max, margin, neg_entropy,
max_logit, random: serving the confident half raised NDCG@10 over serving everything), and **before any control signal of this addendum was computed on any panel**.
It adds one descriptive control analysis. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-10, and it never
changes a number of the registered audit (its outputs go to a separate directory).

## 1. Question

The registered audit shows that the LLM's list confidence selects events on which the LLM's own ranking does well (section D of `docs/sigir/NEXTITEM_AUDIT_SPEC.md`:
the risk-coverage curve and the gain of serving the confident half, against a random signal). A random signal is a weak control. The question of this addendum is
whether the confidence selects those events **better than a cheap signal that needs no LLM score**, i.e. whether the LLM's uncertainty knows something that the user's
history length, the user's popularity profile or a popularity-only ranker's own confidence does not.

## 2. Panels, utility and signals

- **Panels.** The panels and scores of the registered audit: Qwen3-8B zero-shot, prompt V0, question `next`, the registered TEST segment (Sports: events 1,001-10,000;
  Toys, Home, Tools: all 10,000 events); and, as the second-backbone replication of the same analysis, the Z2 audits (events 1,001-3,000 of each domain) of Qwen3-8B
  restricted and of Llama-3.1-8B-Instruct, whenever their score files exist. The VALID events and scores the audit uses for its temperature are used here for the
  direction rule (section 3).
- **Utility.** The audit's per-event utility: NDCG@10 of the LLM ranking of the event (HR@1 as the second utility), with the audit's tie handling.
- **LLM signal.** `p_max` of the audit (the list-normalised top-1 probability under the audit's fitted temperature).
- **Control signals** (computed from the panel file and the VALID events only; never from a label, a TEST score or the LLM's output):
  1. `hist_len`: the number of history items of the event.
  2. `profile_pop`: the user popularity profile of section D of the audit specification (mean group score of the history items, head 2, mid 1, tail 0; events whose
     user has no mapped history item are excluded from this signal's serving set and counted).
  3. `pop_conf`: `p_max` of a popularity-only ranker. Its score of a candidate is the group score (head 2, mid 1, tail 0); its temperature is fitted on the VALID events
     by the audit's own temperature routine (negative log-likelihood of the positive under the list softmax); equal scores receive equal probability.
  4. `random`: the audit's fixed-seed random signal (the registered null, shown for reference).

## 3. Direction rule

Serving the events with the highest value of a signal presupposes a direction. For each control signal the direction is the sign of the Spearman correlation, on the VALID events
with LLM scores, between the signal and the event's NDCG@10 (ties and exact zero: positive). The direction is recorded in the output and never taken from the TEST events.
`p_max` is used as it is (the registered direction). If no VALID score file exists for a panel, the panel is not analysed and reads `NOT_RUN` with that reason.

## 4. Estimands

For each signal s: `gain50(s)` = mean utility of the served 50% (k = max(1, round(0.5 n)), ties by the audit's tie-block rule) minus the mean utility of all events; the curve over the
audit's coverage points and the audit's AURC are reported descriptively. The contrast of interest is `Delta(s) = gain50(p_max) - gain50(s)`, with the audit's event-cluster
bootstrap (2,000 resamples, seed 0, the same resamples for all signals, so the contrasts are paired) and a 95% percentile interval; the same for HR@1. The niche-user served share
of section D is reported for each control signal beside the registered ones.

## 5. Reading rule (fixed now)

Per panel and control signal: `LLM_BETTER` if the lower bound of the interval of `Delta(s)` (NDCG@10) is above 0; `CHEAP_BETTER` if its upper bound is below 0; `MATCHED` if the whole
interval lies within +-0.01; otherwise `INCONCLUSIVE` (checked in this order: LLM_BETTER, CHEAP_BETTER, MATCHED). Per panel: `ADDS` if every control signal reads `LLM_BETTER`;
`CHEAP_SUFFICES` if any control signal reads `MATCHED` or `CHEAP_BETTER`; otherwise `MIXED`. Across panels only counts are reported (domains per label; backbones separately); there is no
family, no Holm correction and no direction word beyond these labels. The analysis is exploratory (written after the registered serving block of two domains was read) and is marked
as such wherever it appears; a statement beyond one panel follows the claim-admission rule of Amendment 3 addendum 6 section 9 (the same label on a fresh Amazon domain and on
the second backbone).

## 6. Code and record

A new module and its test file (`src/confrec/nextitem_serving_control.py`, `tests/test_confrec_serving_control.py`) that import the audit module read-only (`nextitem_audit.py` at
its recorded sha1) and write only to `outputs/confrec/nextitem_audit_ctrl/`. Their sha1 are recorded in the pilot log before the first run on any real panel, and a CPU run needs no GPU.
