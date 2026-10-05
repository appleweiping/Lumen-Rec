# Two same-family reviews of the draft (2026-10-04/05): what was found, what was decided

Both reviewers were Claude (Opus) instances acting as SIGIR programme-committee members, read-only, working from the paper sources, the
amendment chain, the pilot log and the committed result files. **Same-family and provisional: neither review is independent.** Their
text is not stored in the repository; this note keeps the findings that drive work, with the decision taken and where it is handled.
Numbers quoted by the reviewers were re-read from `docs/sigir/results/grid/qwen/ml1m.json` before use (all matched).

## Review 1 (novelty, significance, positioning): provisional score 4/10 as framed, 5-6/10 with the additions below

| finding | decision | where |
|---|---|---|
| As framed the paper is an audit/registration report; the lemma and the protocol are not research contributions. The real finding is already in the ML-1M report: LoRA raises UAUC 0.596 -> 0.742 but the non-prior share falls 0.61 -> 0.10, G = 0.010 against G_CF = 0.036, and the tuned model sits below the item mean (0.759) and MF (0.791) on the same rows. | adopted: findings-first abstract and introduction; lemma kept short as a remark; same-row references beside every LLM number; the guide is a summary, not a contribution | abstract/introduction rewritten (2026-10-04); editor pass |
| Make the headline comparative: MF's item share, a stacker with the CF residual, the LLM prior on cold/sparse items, FT-C as a registered contrast on ML-1M and Toys. | adopted | addendum 6 (E-G, E-F, E-H, FT-C); `ftgrid_extra.py`, `run_ftc.sh` |
| Prior work missing or mis-characterised (PerRecBench, Kang et al., CFT, Lost in Sequence, Di Palma et al., Hou, Lichtenberg, ReLLa, PRP, pointwise-scoring and denoising work, performance prediction). | adopted after verification of every citation | citation agent; related work |
| Inaccurate claims: "not by calibration" (ReLLa reports log loss), "missing non-LLM reference points", "all evaluate P(Yes) by AUC", the novelty of asking how much of P(Yes) is item prior, "an existing benchmark protocol / verbalised reranker" uncited, "two statements do not depend on the data" (an accounting identity). | adopted | introduction rewritten; related work, conclusion, anonymisation: editor pass |
| Thinner than strong uncertainty-aware LLM4Rec papers: no method in the main line, one recipe, two 8B backbones, no alternative uncertainty estimators, utility rests on selective serving. | partly adopted: the nested method slot stays (kill rule); a 3-epoch arm is optional and exploratory; alternative estimators (sampling, verbalised) are out of scope and named as a limitation | limitations; schedule |
| Process over substance: self-hosted registration; compress the protocol; third-party timestamps. | adopted: protocol compressed, deviations table; OpenTimestamps is the authors' decision (not run by the assistant) | `docs/sigir/DEVIATIONS.md`; editor pass |

## Review 2 (methodology, statistics, reproducibility): provisional score 5/10, about 7 with the three changes below

| finding | decision | where |
|---|---|---|
| The non-prior share is not an upper bound on the personal share (a variance-orthogonal share can be below Var(e)/Var(L); planted-world simulation). | adopted and verified in principle: the description is withdrawn; e-share reported beside it; G is the personal-information measure | addendum 6 item 1.1; preliminaries rewritten |
| The pooled, uncentred stackers of E-D can bias G (simulation: down to -0.018 with no personal signal). | adopted; checked on ML-1M with user-centred features: G(ZS) -0.0018 -> +0.0001, G(FT) 0.0096 -> 0.0092, P1 unchanged in sign | addendum 6 item 2 (E-W); PILOT_LOG |
| LoRA 0.742 is below the item mean 0.759 and MF 0.791; q-hat is advantaged by ratings in [T_d, t). | adopted: references beside every number; q-hat_T; paired contrast E-J | addendum 6 item 6 |
| Forking paths in unfrozen wording rules; direction words on unadjusted CIs (a comment rule); mixed outcomes; admission rule weak. | adopted: wording rules frozen by hash, direction words only from registered tests, counts for mixed outcomes, replication = same sign with own interval excluding 0 | addendum 6 items 1.2 and 9; PILOT_LOG |
| Registration credibility: Amendment 1 has no hash line; 13 post-hoc degrees of freedom. | adopted: retroactive record with commit and server reflog times; the deviations table | PILOT_LOG; `DEVIATIONS.md`; appendix table (editor pass) |
| FT-C does not rule out content-to-quality priors on unseen items (ML-1M TEST items are 97% seen). | adopted: FT-C on Toys (11% seen) with a reading rule | addendum 6 item 8; `run_ftc.sh` |
| `decide()` thresholds are unit-dependent (zero-shot logits are about 11-12x over-sharp), no power. | text only for now: the knockout is a descriptive label, scoped to store-name strings (the pseudonymiser replaces the store string wherever it occurs in titles and texts; sub-brands, characters and product lines remain) | editor pass; limitation |
| S6 is confounded by item composition (P2 prunes per class, not per item). | to be registered before the first S6 run: composition diagnostics and an item-stratified random arm | addendum 7 (pending) |
| Rated top-k anatomy uses an oracle threshold; niche groups unequal; Gini CI is a recentred percentile interval. | adopted: deployable top-k variant (E-C'); the other two are documented (Gini and APLT are printed as estimates only) | addendum 6 item 7; deviations table |
| Reproducibility gaps: per-pair scores, model revision hashes, environment lock, nondeterminism statement, code hash in outputs, data builders, anonymisation. | adopted for the artefact | release checklist (later) |
| Code checks (UAUC, user bootstrap, Holm, sigma_seed): no defect found; sigma_seed's sign clause is implied for n <= 5 (coherent, conservative). | noted | none |
