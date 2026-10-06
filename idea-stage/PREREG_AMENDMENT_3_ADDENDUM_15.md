# Pre-registration amendment 3, addendum 15 (2026-10-06): reference extensions for the rated panels: matched item means, a tuned and history-matched collaborative reference, a content-based personal reference (exploratory, outcome-free)

**Status: part of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the amendment). Written after every rated-panel result of the fine-tuned programme had been read and after three independent same-family reviews
of the draft had shown, by recomputation from the stored reports, that (i) the temporal MF reference of the registered reports is badly over-fitted on the three Amazon panels (training RMSE 0.10-0.11, held-out RMSE 1.9-2.0 on the 1-5 scale, 26-54% of TEST pairs cold
for MF and scored as ties, MF's own item bias alone scoring 0.61-0.68 UAUC on the warm pairs against 0.55-0.57 for the full MF), so that its personal residual (UAUC about 0.50) and G_CF cannot be read as 'the panels carry no personal signal'; (ii) the item mean q-hat uses ratings
up to the candidate's timestamp and the whole corpus (1.0M-19.6M events) while the LoRA sees 15-23k labels, and q-hat_T is matched in time only; (iii) the MF fold-in uses the user's whole earlier history while the prompt shows 10 events; and **before any quantity of this addendum was computed
on any panel**. It adds descriptive CPU-only references. It changes no design element, endpoint, threshold or family of Amendment 2, Amendment 3 or addenda 1-14, and never changes a number of the registered reports.

## 1. Questions

(a) Do the Amazon panels contain personal signal that a properly regularised collaborative reference, or a non-collaborative content reference that uses exactly what the prompt shows, can find? (b) How does the fine-tuned and the zero-shot LLM compare with an item mean built from the LoRA's own TRAIN labels and with a
time-matched item mean, with intervals? (c) Do G_CF and G_LLM|CF (E-F) change when the collaborative reference is tuned and history-matched?

## 2. References (rated panels: ML-1M, Toys, Video_Games, Sports; the rows, users S_d, splits, cross-fitted stackers and user-cluster bootstrap (2,000 resamples, seed 0) of the registered reports and of `src/confrec/ftgrid_extra.py`, imported unchanged)

1. **Item means.** m_TR(i) = (sum of the labels of the TRAIN examples of i + 5 mu_TRAIN) / (n_i + 5), mu_TRAIN the TRAIN like rate (an item without a TRAIN example scores mu_TRAIN; ties are handled by the tie-aware UAUC); the registered m and the time-matched m_T are reported beside it. Per panel: UAUC of m_TR with its interval, the
   share of TEST pairs whose item has a TRAIN example, and for the zero-shot model and the LoRA seed mean Delta UAUC(l - m), Delta UAUC(l - m_T) and Delta UAUC(l - m_TR) with intervals.
2. **Tuned, history-matched collaborative reference.** The temporal biased MF of the registered reports is refitted on the same ratings before T_d with the regularisation weight chosen on the CAL rows of the EVAL users (the candidates before T_d of users disjoint from S_d's TEST rows; candidate ratings withheld from the fit) from {0.05, 0.2, 1, 5, 20} by the
   held-out RMSE of the ratings (ties: the larger weight), with the registered dimension and number of iterations; the choice is recorded. Variants, each scored on the TEST rows and reported with warm-pair, cold-pair and all-pair UAUC: (a) the tuned MF with the user's whole earlier history in the fold-in; (b) the tuned MF with the fold-in restricted to the user's last
   10 events (what the prompt shows); (c) the item-bias-only model b_i (a fully regularised fit with no user factor) and the personal residual MF - b_u - b_i of (a) and (b). Cold items and users are scored by the registered fallback (global mean) and counted. Then G_CF(E-D and E-W) = Delta UAUC([m, MF residual] - [m]) and the E-F quantities G_LLM|CF and G_CF|LLM are recomputed with the
   tuned residuals of (b) for every panel and regime.
3. **Content-based personal reference.** For every TEST pair a content score c(u,i) = mean over the user's history items h with rating >= 4 of cos(x_i, x_h) minus mean over history items with rating <= 2 of cos(x_i, x_h), x the TF-IDF vector (fitted on the panel's TRAIN rows only; unigram and bigram, sublinear tf, minimum document frequency 2) of the item's title,
   category or genre string and brand (the panel fields history_titles, history_meta, history_brands, candidate_titles and their candidate counterparts; no collaborative information, no rating of the candidate, no item popularity). Reported per panel: UAUC of c alone and of the user-centred c; G_content = Delta UAUC([m, c] - [m]) (E-D and E-W versions) with its interval; G_LLM|content = Delta UAUC([m, c, e-hat] - [m, c])
   for every regime.

## 3. Reading rules (fixed now)

No hypothesis, no family, no Holm correction, no direction word beyond the labels. Per panel and reference r in {tuned MF residual (b), content}: `SIGNAL_PRESENT` if the interval of G_r (E-W) lies above 0, `SIGNAL_ABSENT` if it lies within [-0.01, +0.01], otherwise `INCONCLUSIVE`; the label is a property of the panel and the reference, not of the LLM. A
statement beyond one panel follows the claim-admission rule of addendum 6 section 9. Everything is marked exploratory. The registered MF numbers stay in the tables as registered, and the paper says that they were over-fitted on the Amazon panels.

## 4. Code and record

New module and runner only (`src/confrec/ftgrid_refs.py`, `scripts/sigir/run_ftrefs.sh`) with tests; the registered report, forensics and extra-analysis code is imported read-only and stays byte-identical (the MF is refitted by calling the registered routine with a different weight where it has the parameter, else by a documented copy of its
algorithm with a test that it reproduces the registered MF numbers at the registered weight). CPU only; the container has a 60 GiB RAM cap: the run must stay below 20 GiB (the MF of the two large Amazon panels is the heavy step: report its memory); writes only below `outputs/confrec/ftgrid_refs/`. Their sha1 are recorded in the pilot log before the first run on a real panel.
