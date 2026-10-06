# Pre-registration amendment 3, addendum 18 (2026-10-06): swap-prior decomposition of the next-item logits (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after the registered next-item audits of Toys and Sports (Qwen3-8B zero-shot, prompt V0) had been read and after three independent same-family reviews of the draft had asked whether the item prior dominates the next-item logits as it
dominates the rated-panel confidence (the decomposition of Amendment 3 was run on rated panels only, although yes/no LLMs are deployed on candidate lists); and **before any swap-arm prompt of a next-item panel was scored and before any quantity of this addendum was computed**. It adds one exploratory GPU scoring step and descriptive CPU analyses. It changes no design element, endpoint,
threshold or family of Amendment 2, Amendment 3 or addenda 1-17 and never changes a number of the registered audit.

## 1. Question

How much of the ranking skill and of the within-event variance of the next-item confidence is a user-independent item prior (the model's mean answer for the candidate under other users' histories), and what does the event-specific residual add beyond that prior and a popularity reference?

## 2. Scoring step (GPU)

For each domain Sports, Toys, Home and Tools (Home and Tools after their registered audit scores exist), the first 1,000 events of the audit's registered TEST segment (Sports: events 1,001-2,000; the other domains: events 1-1,000) are scored with Qwen3-8B zero-shot under the registered prompt V0, question `next`, history length 5, fp16, exactly as the audit
(the frozen audit checkout, `src/confrec/pyes_scorer.py` with `--questions next --hist_len 5 --dtype float16 --topk_logprobs 50 --chunk_users 100`) and with the swap arm `--swap_k 8` (every unique candidate item of the 1,000 events under up to 8 other users' histories; donors hold the item neither in their history nor as a candidate; donor seed 0). The main prompts of the 1,000 events are
scored again by the same run (their logits must agree with the audit's stored scores to the audit's vLLM batching tolerance, 1e-3 on the logit; the maximum difference is reported). The integrity check E1 of the audit applies (a run that fails it is rerun once; a second failure is reported as missing). About 2.8 GPU-hours per domain at the audit's rate of 80 prompts per second
(1,000 events x 101 candidates x (1 + 8) prompts, plus the main prompts).

## 3. Analyses (CPU; event-cluster bootstrap, 2,000 resamples, seed 0, the audit's)

1. **Prior and share.** pi-hat(i) = mean of the finite donor logits of item i; with the event offset removed (candidates centred within the event), r_c the pooled within-event correlation of the two donor-half means (donors 1-4 and 5-8) over the event's candidates, r_8 = 2 r_c / (1 + r_c) (the rated-panel definition, events in place of users) and rho the pooled within-event
   correlation of the centred own-history logit and pi-hat: the item-prior share rho^2 / r_8 (reported only if r_8 >= 0.7), the share of variance of the centred residual e-hat = l - pi-hat - event mean, and the popularity link (the partial Spearman correlation of l and of pi-hat with the candidate's popularity group score, head 2, mid 1, tail 0, as the audit's group map).
2. **Rankings.** HR@1, HR@10, NDCG@10 and the per-event AUC (UAUC) of the rankings by l (own history), by pi-hat alone, by e-hat alone and by the popularity group score alone (ties: the audit's tie-expected rule), with intervals; the share NDCG@10(pi-hat) / NDCG@10(l); the paired contrasts l - pi-hat and l - popularity.
3. **Information gain on events.** G_next = Delta UAUC([pop, pi-hat, e-hat] - [pop, pi-hat]) and Delta UAUC([pop, pi-hat] - [pop]) with logistic stackers cross-fitted over two halves of the events (all 101 candidates of an event stay together), the registered fixed-coefficient bootstrap and a full-refit bootstrap (500 resamples), and the planted-signal and permutation-null checks of addendum 14 item 1 applied to e-hat (500 permutations within events).
4. **Strata.** The analyses of items 2 and 3 within the strata of addendum 17 item 1 (popularity group of the positive item).

## 4. Reading rules (fixed now)

No hypothesis, no family, no direction word beyond the labels of addendum 17 item 5 for the contrasts (smallest effect size of interest: 0.01 in NDCG@10, HR@1 and UAUC); the item-prior share is described, never 'confirmed'; the analysis is exploratory and marked so; a statement beyond one panel follows the claim-admission rule of addendum 6 section 9. If r_8 < 0.7 on a domain, its shares are reported as uninterpretable.

## 5. Code and record

A runner for the scoring step (`scripts/sigir/run_nextitem_swap.sh`, GPU, queue job; it refuses unless the freeze checks pass and the sha1 of the runner and of the analysis module are recorded) and a new analysis module (`src/confrec/nextitem_decomp.py`) with tests; the audit module and the scorer stay byte-identical. Their sha1 are recorded in the pilot log before the first scored prompt.
