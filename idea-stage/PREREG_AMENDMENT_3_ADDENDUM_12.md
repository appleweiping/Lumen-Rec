# Pre-registration amendment 3, addendum 12 (2026-10-06): uncertainty-aware routing between the LLM and a published recommender (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the registered
next-item audits of Toys and Sports (Qwen3-8B zero-shot, prompt V0) had been read, including the selective-serving block (serving the confident half raised NDCG@10) and
the reference NDCG@10 of the eight published recommenders and the verbalised reranker on those panels, and **before any routed ranking was computed on any panel**. It adds
one descriptive, exploratory analysis. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-11, and never changes a number
of the registered audit.

## 1. Question

Selective serving (section D of `docs/sigir/NEXTITEM_AUDIT_SPEC.md`) shows on which events the LLM ranks well; it does not say what to do with the other events. The question
is whether the LLM's list confidence can **route** each event between the LLM's own ranking and the ranking of a published recommender so that the routed system beats
both single systems, and whether it does so better than routing by a random signal or by the cheap signals of addendum 11.

## 2. Systems and utility

- **LLM:** the audit's Qwen3-8B zero-shot ranking under the registered prompt V0, question `next`, the registered TEST segment of the audit (Sports: events 1,001-10,000; Toys, Home,
  Tools: all 10,000 events), with the audit's tie handling. Utility of an event: NDCG@10 (tie-expected, as in the audit).
- **Published systems B:** the eight recommenders of the audit's reference set (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec; `docs/sigir/ref_ranks/<domain>/<method>.csv.gz`:
  the rank of the positive among the same 101 candidates per event). Utility of an event: NDCG@10 from the rank, `1/log2(1+rank)` if the rank is at most 10, else 0 (the files carry no ties).
  The verbalised reranker (`ccrp_v3`) of the same directory is reported beside them as context and is never the primary B.
- **Primary B per domain:** the published recommender (of the eight) with the highest TEST NDCG@10 on that domain's TEST segment, computed from the same reference files; the rule is deterministic
  and is recorded in the output together with all eight values.

## 3. Policies and signals

A policy with signal s and coverage c serves the LLM's ranking on the events whose signal lies in the top share c of the segment (k = max(1, round(c n)); events with equal signal form a tie
block whose utilities are replaced by the block's expected utility, the audit's rule, so ties never help) and system B's ranking on all other events. Signals: the LLM's `p_max` (the audit's
list-normalised top-1 probability under the audit's fitted temperature), and the control signals `hist_len`, `profile_pop`, `pop_conf` and `random` of addendum 11 with the directions fixed there on
the VALID events by the sign of the Spearman correlation with the LLM's own NDCG@10 (the direction answers 'on which events does the LLM do well'; it is never fitted on TEST events or on B's utility).
The oracle router (the system with the higher utility on each event) is reported as an upper bound and is never a policy.

## 4. Estimands (fixed now)

Coverage c = 0.5 is the primary point (the registered serving gain of the audit); the curve over c in {0.1, 0.2, ..., 0.9} and its area are descriptive. For each signal, `U_route(s)` is the mean
utility of the routed system at c = 0.5. Primary contrast: `Delta_best = U_route(p_max) - max(U_LLM, U_B)` where U_LLM and U_B are the mean utilities of the two single systems on the same events.
Secondary contrasts (paired, the same resamples): `U_route(p_max) - U_route(s)` for each control signal s (random routing has the closed form c U_LLM + (1 - c) U_B, shown as the reference), the same for
every one of the eight published recommenders as B (descriptive table) and for the verbalised reranker. HR@1 is the second utility. Intervals: the audit's event-cluster percentile bootstrap, 2,000
resamples, seed 0.

## 5. Reading rule (fixed now)

Per panel: `BEATS_BOTH` if the lower bound of the 95% interval of `Delta_best` is above 0; `BELOW_BEST` if its upper bound is below 0; `MATCHED` if the whole interval lies within +-0.005; otherwise
`INCONCLUSIVE` (checked in this order). Against each control signal the labels of addendum 11 (`LLM_BETTER`, `CHEAP_BETTER`, `MATCHED` within +-0.01, `INCONCLUSIVE`). Across panels only counts are
reported; there is no family and no Holm correction, and the analysis is marked exploratory wherever it appears. A statement beyond one panel follows the claim-admission rule of Amendment 3 addendum 6
section 9 (the same label on a fresh Amazon domain and on the second backbone: the Z2 audit of Llama-3.1-8B-Instruct on events 1,001-3,000 whenever its score files exist, with the same B rule).

## 6. Code and record

Implemented beside the code of addendum 11 (a subcommand of `src/confrec/nextitem_serving_control.py` or a module that imports it), importing the audit module read-only, writing only to
`outputs/confrec/nextitem_audit_ctrl/`. Its sha1 are recorded in the pilot log before the first run on a real panel; it needs the audit's score files and the reference rank files and no GPU.
