# Pre-registration amendment 3, addendum 17 (2026-10-06): artefact checks and effect-size floors for the next-item serving, exposure and calibration endpoints (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the registered next-item audits of Toys and Sports (Qwen3-8B zero-shot, prompt V0) had been read and after a third independent same-family review of the draft had shown, from the stored audit files, that
the sampled-candidate design can confound the serving and exposure endpoints (Toys: under random serving niche users already score NDCG@10 0.335 against 0.201 for mainstream users; the user-popularity 'quintiles' hold 17/23/13/47/0% of the users; the head share of the top-10 exceeds the pool's by 0.059 but the target's by 0.040; the paper's Brier of p_max is, in the code, the
multiclass Brier of the 101-way list; 69% of the events fall in the first of 15 equal-width calibration bins; with 10,000 events trivial differences have intervals excluding 0); and **before any quantity of this addendum was computed on any panel**. It adds descriptive CPU-only checks. It changes no design element, endpoint, threshold or family of
Amendment 2, Amendment 3 or addenda 1-16 and never changes a number of the registered audit.

## 1. Questions

(a) Is the LLM's serving gain, its niche-versus-mainstream served share and its calibration explained by how easy an event is made by the sampling design (the popularity of the positive item, the popularity profile of the user) rather than by anything the LLM knows? (b) How much of the head-share excess of the top-10 lists is warranted by the target (the positives are more often head items than the pool)? (c) Which differences
are larger than a smallest effect size of interest?

## 2. Panels

The panels and score files of the registered audit (Qwen3-8B zero-shot, prompt V0, the registered TEST segment of each domain: Sports events 1,001-10,000, Toys, Home, Tools all 10,000 events) and, whenever their score files exist, the Z2 audits (events 1,001-3,000) of Qwen3-8B restricted and Llama-3.1-8B-Instruct. The audit module and its loaders are imported read-only; outputs go to `outputs/confrec/nextitem_audit_diag/`.

## 3. Analyses (event-cluster bootstrap, 2,000 resamples, seed 0, the audit's)

1. **Strata by the positive's popularity group** (head, mid, tail of the audit's group map; the label is used only to stratify, never as a signal): per stratum the number of events, top-1 accuracy and NDCG@10 of the LLM, the AUROC of `p_max` for top-1 correctness, and the gain of serving the most confident half within the stratum for `p_max`, `random` and the three cheap signals of addendum 11 (history length, user popularity profile, popularity-only ranker confidence), each as an estimate with its interval.
2. **Strata by the user's popularity profile** (the audit's quintiles, lowest to highest, with their event counts, empty quintiles listed as empty): per quintile the utility under random serving and under `p_max`, the share served at 50% coverage and the gain of serving the confident half within the quintile.
3. **Exposure against the target.** The head, mid and tail shares of the top-10 lists of the LLM, the verbalised reranker and the eight published recommenders, the same shares of the positives (the target) and of the candidate pool; `Delta_head_vs_target` = head share of the top-10 minus the head share of the positives, and the paired difference to the pool-based `Delta_head`; the NDCG@10 of every system beside its exposure (so that a system at random-level accuracy is identified).
4. **Calibration sensitivity.** The top-label ECE with 15 equal-width bins (the audit's), with 10 equal-mass bins, the top-1 Brier (`p_max` against top-1 correctness) and the multiclass Brier of the list, and the share of events in the first equal-width bin.
5. **Smallest effect sizes of interest (fixed now).** NDCG@10 and HR@1 gains of serving or routing: 0.01; `Delta_head` and `Delta_head_vs_target`: 0.02; head-minus-tail mean `p`: 0.005; ECE and Brier differences: 0.01; AUROC differences: 0.02. An estimate whose whole 95% interval lies within +-SESOI is labelled `NEGLIGIBLE`, one whose interval lies above SESOI (or below -SESOI) `LARGER_THAN_SESOI`, otherwise `UNRESOLVED`; the labels are added to every serving, routing, exposure and calibration estimate that the paper reports.

## 4. Reading rules (fixed now)

No hypothesis, no family, no direction word beyond the labels of item 5; the analyses are exploratory and marked so; a statement beyond one panel follows the claim-admission rule of addendum 6 section 9. If the serving gain of `p_max` is positive in every stratum of item 1 with intervals above 0 the paper may say that it is not explained by the positive's popularity; if it vanishes in a stratum, the paper says so.

## 5. Code and record

Part of the next-item control code of addenda 11 and 12 (a subcommand or a module that imports it) with tests; the audit module stays byte-identical; their sha1 are recorded in the pilot log before the first run on a real panel. CPU only.
