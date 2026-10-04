# GATE-FIX METHODOLOGIST: a pre-registrable prompt-fix protocol for the failed Pilot-1 token-channel gate, built to avoid forking paths.

**Headline.** Keep the 0.60 bar. Run one prompt fix, registered in advance, that never looks at MIRROR. It tests 8 prompt variants taken from TALLRec, Kang et al. 2023 and ReLLa (V0–V7), picks one on ML-1M and Toys dev users who were not in the pilot, and re-tests that one prompt once, with the gate scored first, on the original pilot panels. If it fails, record GATE_FAIL_AFTER_REMEDY, use the zero-shot rated data only as description, and move the method study to the fine-tuned regime under the same 0.60 bar. The bar was set without a reference point: the only comparable published zero-shot yes/no UAUC on ML-1M is 0.527, against 0.682 after fine-tuning. Lowering it now would still be outcome-dependent.

## Analysis
0. BOTTOM LINE
Pilot 1 failed on the registered rule (ML-1M raw like UAUC 0.587 < 0.60). That failure stands and goes into the paper. The registered remedy, "fix the prompt first", is binding, but it does not say how. Done ad hoc, it is the textbook way to manufacture a pass. Under the protocol below the remedy is a single, finite, pre-specified selection experiment:
- 8 prompt variants, V0 = the registered prompt as control.
- Scored only on users who were not in the pilot.
- Only the gated question is scored, so the selection is MIRROR-blind.
- One selection metric with a deterministic tie rule.
- One confirmatory re-test on the original pilot panels, using the unchanged pilot1_gate.py, with the gate computed before any other arm.
- A contingency tree written now, before any new data.
The 0.60 threshold is not amended for the zero-shot Qwen3-8B gate under any outcome.

1. WHAT THE PILOT SAYS ABOUT THE CHANNEL: WORKING BUT WEAK, NOT BROKEN
Evidence from outputs/confrec_pilot/pilot1/*.json and report.json:
- Channel integrity is clean: censoring 0 on all 1.13M rows, Yes+No mass about 0.99999, and the Yes/No token-id sets are logged.
- The sports next-item component passes: raw(next) NDCG@10 0.209 against a bar of 0.186.
- ML-1M UAUC 0.587 is far from 0.5. My estimated 95% CI is about ±0.009, assuming a per-user AUC SD of 0.15–0.20; this has not been computed from the scores.

So the registered rationale ("the token channel or prompt is broken") does not describe this outcome. What failed is efficacy: zero-shot within-user discrimination among items the user chose to rate.

Two diagnostics are relevant. Neither may be used as a decision input.
- `evidence = like − π` collapses ML-1M UAUC to 0.507. That hints that almost all of the zero-shot discrimination is item-level, i.e. the model knows which movies are good, not which ones this user likes.
- Platt slope 0.062 means the logits are roughly 16× too extreme.

These motivate one risk flag: a prompt fix may pass the gate by sharpening the item-quality prior rather than personalization. That would not rescue MIRROR. The protocol therefore keeps the registered evidence/π arms in the confirmatory run and reports them descriptively.

2. (d) WAS 0.60 WELL FOUNDED? CAN IT BE AMENDED?
Published anchors, all read in the PDFs this session:

| Source | Setting | Labels / metric | Result |
|---|---|---|---|
| TALLRec (RecSys'23, 2305.00447), Fig. 1/3a | Zero-shot in-context yes/no, ML-100K "Movie" | >3 like, ≤3 dislike; global AUC; refusals excluded | Alpaca-LoRA 0.46, Davinci-002 0.49, Davinci-003 0.53, ChatGPT 0.50 |
| TALLRec | 64-shot / 256-shot tuning | same | 0.675 / 0.720 |
| CoLLM (TKDE 2025, 2310.19488), Table 2 | ML-1M, temporal split | >3 positive; AUC / UAUC | ICL (zero-shot Vicuna-7B) 0.532 / **0.527**; TALLRec LoRA (Vicuna-7B) 0.710 / **0.682**; MF 0.648 / 0.636; SASRec 0.708 / 0.688; best CoLLM UAUC 0.699 |
| Kang et al. 2023 (2305.06474), Table 2 | ML-1M zero-shot rating prediction | ≥4 positive; global AUC | Flan-U-PaLM-540B 0.708, ChatGPT 0.679, text-davinci-003 0.695; candidate-item-average 0.740; user-average 0.727 |
| ReLLa (WWW'24, 2308.11131), Table 2 | ML-1M zero-shot CTR | global AUC | Vicuna-7B 0.674, Vicuna-13B 0.699, SUBR 0.701 |

Two points matter for comparing these with our gate:
- **Global AUC is not a UAUC anchor.** Kang's user-average heuristic alone reaches a global AUC of 0.727 with zero within-user discrimination, so the 0.68–0.71 zero-shot global AUCs are mostly between-user leniency. ReLLa's numbers are also global.
- **The one comparable zero-shot yes/no UAUC is 0.527** (CoLLM's ICL baseline). Against that, 0.60 sits about 0.07 above the published zero-shot reference and only 0.036 below a supervised MF (0.636).

So 0.60 was an efficacy bar labelled as a sanity bar. The next-item component of the same gate (0.8 × C-CRP on the same panel) was properly anchored to a reference; the rated component was a round number with no reference. One caveat on comparability: our panel drops 3-star candidates, which makes it easier than CoLLM's task. Our 0.587 is therefore not directly comparable to 0.527, and the difference does not license any upward reinterpretation.

**Is amending it legitimate? No.** A threshold re-chosen after the statistic is seen is a researcher degree of freedom (Simmons, Nelson & Simonsohn 2011; Gelman & Loken). That 0.58 might have been a reasonable ex-ante choice is irrelevant; the registration is valuable exactly because it was fixed. Preregistration norms (Nosek et al., PNAS 2018) allow transparent deviations, but a deviation forfeits confirmatory status, so it buys nothing here.

**Defensible path.**
1. Keep 0.60 for the zero-shot gate and report the failure.
2. Execute the registered remedy under Amendment 2 (below).
3. If the remedy fails, report "zero-shot 8B token P(Yes) does not reach the registered bar on within-user rated discrimination". This is a literature-consistent finding (TALLRec, CoLLM ICL), not a hidden failure.
4. For new regimes (fine-tuned, second backbone), register gates now, before their data, with the same constant 0.60. That leaves no room for threshold shopping, and the fine-tuned anchor (TALLRec UAUC 0.682) makes 0.60 meaningful there.
5. Report CPU references on the same panels: item-mean rating UAUC and MF UAUC, computed from non-candidate ratings. They are context only and never decision inputs.
6. Name the design flaw in the limitations/appendix: the rated gate was unanchored and conflated broken with weak. Future gates separate an integrity test (CI lower bound > 0.5, mass ≥ 0.99, censoring ≈ 0) from an efficacy bar anchored to a pre-measured reference.

The downstream Pilot-1 decision rule must not be touched either, not even in a seemingly conservative direction. For example, a per-domain floor that excluded Toys would interact with Toys' already-seen V0 MIRROR sign (−0.007) and make NEGATIVE harder to reach.

3. (a) THE FINITE CANDIDATE SET
**Principles.**
- Every variant is justified by a published LLM-rec yes/no or rating prompt, or by alignment with the registered label. None is justified by the pilot diagnostics.
- Thinking stays off. Prefill-only P(Yes), Qwen3-8B, fp16, Yes/No read by token id (amendment C0 unchanged).
- max_model_len is raised to 4096 for every variant, V0 included. This changes capacity only.

**Excluded, with reasons:**
- Item average rating, popularity counts or any CF statistic in the prompt. It would inject a label proxy and turn the gate into a CF test.
- User profile or user mean rating. These are constant within a user, so UAUC-inert.
- Few-shot demonstrations. They add an uncontrolled demo-selection degree of freedom, and Kang et al. report ≤ 0.014 AUC gain.
- Thinking on, verbalized confidence, rating-digit expected value. These are different channels, not prompt fixes; thinking-on belongs to A11/B11.
- Answer-order swaps. They shift a global bias, which is rank-inert.

**Question families.**

LIKE family (V0's registered strings):
- like: "Would this user like the candidate item?"
- dislike (mirror): "Would this user dislike the candidate item?"
- like_para (placebo): "Is the candidate item a good match for this user's taste?"
- dislike_para: "Would this user be disappointed by the candidate item?"

THRESHOLD family (label-aligned: registered label ≥4 like, ≤2 dislike):
- like: "Will this user rate the candidate item 4 stars or higher (on a 1-5 scale)?"
- dislike (mirror): "Will this user rate the candidate item 2 stars or lower (on a 1-5 scale)?"
- like_para: "Will this user's star rating for the candidate item be 4 or 5?"
- dislike_para: "Will this user's star rating for the candidate item be 1 or 2?"
- Construct note for later MIRROR use: under this family like and dislike are not logical complements, because of the 3-star gap.

The next question is the same in every variant: "Will this user purchase the candidate item next?". The paraphrase placebo is always (L(like) + L(like_para))/2, as registered.

**User templates.** Every template ends "{question} Answer with only Yes or No." and renders the candidate as "Candidate item:\nTitle: {title}{\nDescription: text[:200]}", as now.
- T_chrono (V0): "You are an expert recommendation system.\n\nUser history (oldest to newest):\n{lines}\n\nCandidate item:…"
- T_split: "You are an expert recommendation system.\n\nItems this user liked (rated 4-5 stars), oldest to newest:\n{liked}\n\nItems this user disliked (rated 1-3 stars), oldest to newest:\n{disliked}\n\nCandidate item:…". An empty list renders "- (none)"; the TALLRec rule is >3 liked, ≤3 disliked.
- T_relevant: T_chrono with the header "User history (most relevant to the candidate, oldest to newest):".
- T_chrono_nopersona: T_chrono without its first line.

**Line formats.**
- Rated lines: "- {title} (rated {r}/5)".
- Meta lines: "- {title} [{meta}] (rated {r}/5)". meta is "Genres: …" for ML-1M and "Categories: {cats[:80]}" for Amazon.
- Next-item lines are the same without the star suffix.

**System messages.**
- Rated, movies: "You are a movie recommender. You predict how a specific user will rate movies from that user's past ratings. Ratings range from 1 to 5 stars; 4-5 stars means the user liked the movie and 1-2 stars means the user disliked it." Amazon panels use "product".
- Next-item: "You are a product recommender. You predict a specific user's preferences from that user's purchase history."

**Variants.** Rated panel / next-item panel; the next-item panel has no stars.

| Variant | Grounding | Rated panel | Next-item panel |
|---|---|---|---|
| V0 | Control | T_chrono, last 10 rated lines, LIKE | T_chrono, last 5, LIKE |
| V1 | Label alignment; Kang rating framing | V0 with the THRESHOLD family | V0 rendering; like/dislike swapped to THRESHOLD, next unchanged |
| V2 | TALLRec | T_split over the same 10 events, LIKE | Identical to V0 (declared) |
| V3 | ReLLa Fig. 6: zero-shot AUC peaks near K=15 on ML-1M | Last 20 | Last 20 (tools: 10) |
| V4 | Kang "Title, Genres, Rating" | Meta lines, last 10 | Meta lines, last 5 |
| V5 | System role + explicit rating scale | System message + T_chrono_nopersona, LIKE | Next-item system message + T_chrono_nopersona |
| V6 | ReLLa SUBR | T_relevant, LIKE (selection rule below) | Falls back to V0: per-candidate histories defeat prefix caching and break the cost limit |
| V7 | Combined Kang-style: V1 + V3 + V4 + V5 | System + 20 meta lines + THRESHOLD | System + 20 meta lines + THRESHOLD like/dislike |

V6 selection rule: the 10 of the last 50 pre-candidate events with the highest cosine similarity to the candidate. Embeddings are the Qwen3-8B mean-pooled last-layer state of "Title: {title}. {text[:200]}", reduced with PCA-512 fitted on the panel items. Selected items are printed chronologically; ties go to the more recent event.

The full spec is a machine-readable JSON (scratchpad/deliberation/gatefix_prompt_variants_v1.json).

4. (b) DEV SPLIT, METRIC AND TIE RULE
**Users.**
- `build_rated_panels` shuffles the eligible rows with `Random(seed=0)`, takes the first n, then shuffles each selected user's candidates in turn. A build with n_users = all eligible therefore has the pilot panel as an exact prefix.
  - Verify by checking that the first 1500 lines reproduce data_sha1 985494c7… for ML-1M (and the corresponding sha1 for Toys).
- ML-1M has 3183 eligible users:
  - positions 0–1499 are the pilot panel (the confirmatory gate population);
  - positions 1500–2299 are **dev (800 users)**;
  - positions 2300–3182 are the **reserve (883 users)**, untouched until replication.
- Toys:
  - positions 1500–2299 are dev (800);
  - positions 2300–3799 are the reserve (1500), or all remaining users if fewer are eligible (check toys_rated.meta.json eligible_users).
- Video_Games and Sports rated panels are never used for selection; they remain fresh generalization domains.
- Next-item dev: the Lumen sports **validation** ranking file (sha256 2538001458e4… in panel_reference.py). Take its first 500 events whose user is not among the users of the first 1000 test events. No test-panel event is touched.
- For V3, V6 and V7 the panels are rebuilt with a 50-event history pool stored in new fields. Existing fields stay byte-identical, which is asserted by comparing candidate ids and labels row by row against the pilot panel.

**Size and power.** From the pilot CIs, paired ΔUAUC standard errors at n = 1500 are about 0.0011 (placebo − raw) to 0.0027 (mirror − raw). At n = 800 that gives SE ≈ 0.002–0.004 for variant − V0. With a one-sided α of 0.05/7 and 80% power, the minimum detectable difference is about 0.007–0.013. That matches the gap to the bar (0.013), so 800 dev users is the smallest defensible size.

**MIRROR-blind dev scoring.** On rated dev panels, only the `like` question of each variant is scored. On sports-valid, only `next`, once per distinct rendering: V0, V1, V2 and V6 share one rendering, so there are 5 renderings. dislike, like_para, swap prior and no-history are never scored on dev.

**Eligibility.**
- E1 Integrity on every dev panel: censored=2 ≤ 0.5%, overlength = 0, mean Yes+No mass ≥ 0.95.
- E2 Toys non-inferiority: Toys dev UAUC ≥ Toys dev UAUC(V0) − 0.010.
- E3 Next-item: sports-valid raw(next) NDCG@10_tie_exact ≥ 0.18632, the registered constant.
- E4 Feasibility: the projected M2 next-item grid (4 domains × {next, like, dislike, like_para} × 2 backbones, at the throughput measured on dev) ≤ 60 GPU-h.

**Selection.**
- V* = argmax of ML-1M dev UAUC(raw like) among eligible variants. This is the gated quantity.
- Ties: every variant within 0.005 of the maximum is tied. Among tied variants, take the higher Toys dev UAUC. If still within 0.005, take the earliest in the simplicity order V0 < V1 < V5 < V3 < V4 < V2 < V6 < V7.
- A fix counts as found only if V* ≠ V0 and the paired user-bootstrap (2000 resamples, seed 0) one-sided 1 − 0.05/7 lower bound of UAUC(V*) − UAUC(V0) on ML-1M dev is > 0.

**One shot.** No second round of variants, no backbone substitution, and no threshold change. All 8 variants' dev numbers are published in the appendix as a multi-prompt sensitivity table (Mizrahi et al., TACL 2024; Sclar et al., ICLR 2024). Testing the winner on fresh users removes the dev winner's curse (Cawley & Talbot, JMLR 2010).

5. CONFIRMATORY RE-TEST (gate first)
Stage 3a scores only V* `like` on the original ML-1M 1500-user pilot panel and `next` on the original sports first-1000 test events, then applies pilot1_gate.py's constants unchanged.
- Only on GATE_PASS does Stage 3b score the remaining registered arms under V*: like/dislike/like_para on all three panels, swap prior K=8 and no-history prior on the rated panels. The registered POSITIVE/NEGATIVE/NULL rule (triage §3.1 plus amendment P1) then applies mechanically.
- On FAIL, no V* MIRROR arm is ever computed on pilot users, so a failed gate creates no new diagnostics to be tempted by.
- The V0 pilot diagnostics are archived, labelled "pre-fix, below bar", and never pooled with V* results.

6. (c) CONTINGENCY, written now
- **F0 (no fix on dev):** no eligible variant meets the improvement criterion, or V* = V0. Stage 3 is skipped, because V0 already failed. Record GATE_FAIL_AFTER_REMEDY(dev).
- **F1:** V* fails either component of the gate in Stage 3a. Record GATE_FAIL_AFTER_REMEDY(confirm).
- **On F0 or F1:**
  1. No further prompt rounds, threshold changes or backbone rescues for the zero-shot gate.
  2. Zero-shot rated results become descriptive observation-study material only: overconfidence (ECE, Platt slope), weak discrimination (UAUC), errors made with high confidence (|logit| AUROC). This answers seed questions 2 and 4 honestly. Every number is re-measured on the reserve users under a pre-declared descriptive analysis and labelled "channel below registered bar". The paper states that zero-shot 8B yes/no rated discrimination failed the registered bar, with the 8-prompt table and the literature anchors as context.
  3. The method study moves to the fine-tuned regime (M3 brought forward):
     - TALLRec-style LoRA yes/no SFT, r=16, α=32, 1 epoch, 3 seeds.
     - Prompt: V* under F1, V0 under F0.
     - Training panels per amendment D: exclude the Lumen valid/test users; global temporal split at T = p80.
     - Gate-FT, registered now: ML-1M UAUC(raw like) averaged over 3 seeds ≥ 0.60 on the pilot users' test-period candidates, all seeds reported. The CPU item-mean and MF UAUC are reported as context.
     - Next-item endpoints stay zero-shot; that channel already passes.
     - If Gate-FT passes, the MIRROR question becomes "mirror tuning vs paraphrase tuning at equal data and steps", with ΔUAUC CIs pooled over seeds, 3/3 seed sign agreement and gain > 2σ_seed.
     - Claim C1 is rescoped to the fine-tuned regime only.
  4. If Gate-FT also fails, the rated-panel method line is closed. The spine pivots to the observation study (S1–S6) plus the verbalized C-CRP next-item channel.
- **Second backbone (Llama-3.1-8B):** a replication, never a rescue. Zero-shot method claims require both backbones to pass.

7. FORKING-PATH LEDGER
- **Frozen before any GPU run:** variant texts and the hashes of the rendered prompts per dev panel; split indices; metric; eligibility rules; tie rule; improvement criterion; contingency.
- **Already seen and therefore unusable as evidence:** every V0 Pilot-1 diagnostic (MIRROR +0.019 / −0.007 / −0.018, the evidence collapse, the popularity correlations).
- **Remaining contamination:** the designer has seen those diagnostics. This is mitigated by literature-only grounding of the variants and by MIRROR-blind selection.
- **Claims:** any paper claim must replicate on the reserve users plus Video_Games and Sports rated.
- **Review:** the review of Amendment 2 is still same-family; an independent, ideally cross-family, review before Stage 1 is advised.

8. IMPLEMENTATION AND COST
- **Code:**
  - a VARIANTS registry in prompting.py with system-message support in chat_ids;
  - split, meta and relevance history renderers;
  - builder fields history_pool_* and history_texts;
  - a test that V0 reproduces prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 on the ML-1M pilot panel;
  - all outputs written under a new root, outputs/confrec/gatefix/.
- **GPU-h:**
  - Stage 1 ≈ 1.5 (about 460k prompts plus SUBR embeddings);
  - Stage 3 ≈ 0.6 for 3a, plus ≈ 2 for 3b only on PASS;
  - replication ≈ 4 (reserve users, Video_Games and Sports rated, Llama);
  - Gate-FT ≈ 10;
  - the full fine-tuned MIRROR study ≈ 30, already inside the M3 budget.
- **Calendar:** about one week. The 2027-01-21 deadline leaves room for either branch.

9. UNCERTAINTIES
- My prior is that prompt variants move UAUC by ±0.01–0.02, so F0 or F1 is somewhat more likely than a pass.
- V1 has the strongest a-priori case because it removes a construct mismatch: the label is a star threshold, while the question asks "like".
- A pass achieved through V4 or V7 metadata may improve the item prior rather than personalization, so the registered evidence arm must be read before any personalization story is told.
- The UAUC CI half-width (±0.009) is an estimate.
- Video_Games and Sports rated eligibility counts need checking.

## Recommendations
### Freeze PREREG_AMENDMENT_2 (gate-fix protocol) before any new GPU run  (≈0 GPU-h)
- Rationale: The registered remedy ('fix the prompt first') does not say how. Without a protocol fixed in advance, any prompt search is a garden of forking paths. The variant set, splits, metric, tie rule and contingency must be frozen while no new data exist, and 0.60 is explicitly kept.
- Protocol: 1) Write idea-stage/PREREG_AMENDMENT_2.md with the 8 variants V0-V7: exact system/user templates, the LIKE and THRESHOLD question families with like, dislike (mirror), like_para (placebo) and dislike_para, and next-item renderings. Source: scratchpad/deliberation/gatefix_prompt_variants_v1.json. 2) Implement a VARIANTS registry in src/confrec/prompting.py: system-message support in chat_ids, split/meta/relevance renderers, max_model_len 4096. Add builder fields history_pool_* (last 50 events) and history_texts, leaving existing fields byte-identical. 3) Tests: (a) V0 renders the ML-1M pilot panel to prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524; (b) the all-eligible ML-1M build's first 1500 lines reproduce data_sha1 985494c7b44d010bec4b11ec62f35ad274dfe91f, and the same check for Toys; (c) candidate ids and labels of the extended panels equal the pilot panel row by row. 4) Render every variant on every dev panel, record per-variant prompt sha1 in the amendment, and get an independent (ideally cross-family) review. 5) State in the amendment: no threshold change, no backbone rescue, no second round, the downstream decision rule untouched, and the V0 diagnostics marked unusable as evidence.
- Risk: The variant designer has seen the Pilot-1 diagnostics. This is mitigated by grounding each variant only in published prompts (TALLRec, Kang et al., ReLLa) or registered-label alignment, and by MIRROR-blind selection. Implementation risk: V0 drifting from the registered prompt; the byte-identity test catches it.
- Paper value: Makes any later positive result defensible against forking-paths and threshold-shopping critiques. The paper gets a short appendix paragraph and can cite a time-stamped amendment.

### Stage 1-2: MIRROR-blind dev scoring of V0-V7 and deterministic selection  (≈1.5 GPU-h)
- Rationale: Select the prompt on users disjoint from the pilot. Score only the gated question, so the choice cannot be steered by MIRROR, placebo or evidence outcomes. Dev n=800 gives a minimum detectable paired UAUC difference of about 0.007-0.013, matching the 0.013 gap to the bar.
- Protocol: Dev data: ML-1M eligible positions 1500-2299 (800 users) and Toys positions 1500-2299 (800), from the builder's own seed-0 order. Next-item: Lumen sports ranking_valid.jsonl (sha256 2538001458e4...), first 500 events whose user is not among the first-1000-test-event users. Score Qwen3-8B fp16, thinking off: rated panels 'like' only, for all 8 variants; sports-valid 'next' only, for the 5 distinct renderings. Eligibility: E1 censored=2 <=0.5%, overlength=0, mean Yes+No mass >=0.95; E2 Toys dev UAUC >= V0 Toys dev UAUC - 0.010; E3 sports-valid NDCG@10_tie_exact(next) >= 0.18632; E4 projected M2 next-item grid <= 60 GPU-h. V* = argmax ML-1M dev UAUC(raw like). Ties within 0.005 go to higher Toys dev UAUC, then to the simplicity order V0<V1<V5<V3<V4<V2<V6<V7. A fix is declared only if V*!=V0 and the paired user-bootstrap (2000 resamples, seed 0) one-sided (1-0.05/7) lower bound of UAUC(V*)-UAUC(V0) > 0. Publish all 8x2 dev UAUCs plus sports-valid NDCG as an appendix prompt-sensitivity table.
- Risk: Most likely outcome: all variants within about ±0.01-0.02 of V0, so no fix is declared (F0). E2 may exclude the ML-1M-best variant. SUBR (V6) needs an embedding and PCA step that is new code.
- Paper value: Either outcome is usable. A clean selection unlocks the confirmatory gate. A null gives a citable multi-prompt result: the zero-shot 8B yes/no UAUC ceiling on rated discrimination across 8 literature prompts, which directly answers seed question 4.

### Stage 3: one confirmatory gate re-test on the original pilot panels, gate scored first  (≈2.5 GPU-h)
- Rationale: The registered gate population is the pilot panel. Testing the dev-selected V* there once is an out-of-sample test that removes the dev winner's curse. Scoring the gate before any MIRROR arm means a failed gate never produces new MIRROR diagnostics on pilot users.
- Protocol: 3a: score only V* 'like' on the ML-1M 1500-user pilot panel and V* 'next' on the sports first-1000 test events, with V*'s next-item rendering. Apply pilot1_gate.py constants unchanged (UAUC >= 0.60; NDCG@10 >= 0.8*0.2329). 3b, only on GATE_PASS: score the remaining registered arms under V*: like/dislike/like_para on all three panels, swap K=8 and no-history on the rated panels. Then run pilot_mirror.py and pilot1_gate.py unchanged and take the mechanical POSITIVE/NEGATIVE/NULL decision. Archive the V0 outputs as 'pre-fix, below bar'; never pool them with V*. Write outputs under outputs/confrec/gatefix/ with run.key hashes.
- Risk: Regression to the mean: a dev gain of about +0.013 may shrink below the bar on the pilot panel, giving F1. A pass driven by item-prior sharpening (e.g. via V4/V7 metadata) would still leave MIRROR null. The registered evidence/pi arms must be read before any personalization story.
- Paper value: The single decisive event of the gate-fix line. It either legitimately opens the zero-shot MIRROR study or closes it cleanly, with a fully auditable trail.

### Mandatory fresh-user replication, plus a second-backbone policy  (≈4 GPU-h)
- Rationale: Pilot-1 diagnostics (MIRROR +0.019, evidence collapse, popularity correlations) were seen on pilot users. The task's honesty rule requires re-validation on fresh users and domains before any claim. A Llama-3.1-8B result must replicate the finding, never rescue it.
- Protocol: Run the full registered arm set under the final prompt (V* or V0) and the identical analysis on: ML-1M reserve (positions 2300-3182, 883 users; MDE for dUAUC(MIRROR-placebo) about 0.0095), Toys reserve (positions 2300-3799), and the untouched Video_Games and Sports rated panels (1500 users each). Pre-declare that a pilot-derived claim enters the paper only if it replicates in sign with a 95% user-bootstrap CI excluding 0 on the ML-1M reserve and on at least one fresh Amazon domain. Run Llama-3.1-8B-Instruct on the pilot panels with the same prompt. Zero-shot method claims require both backbones to pass the 0.60 gate. A Llama-only pass is reported as exploratory and does not override a Qwen3 failure.
- Risk: Diagnostics may fail to replicate; that is the intended filter. Amazon rated eligibility (>=3 likes and >=3 dislikes) may give fewer than 1500 users for Video_Games or Sports, which needs checking.
- Paper value: The only route by which any Pilot-1-derived observation can become a paper claim. It also gives the 2-backbone x 4-dataset breadth reviewers expect.

### Pre-registered contingency: GATE_FAIL_AFTER_REMEDY leads to a fine-tuned regime gated at the same 0.60  (≈10 GPU-h)
- Rationale: If no prompt passes, the defensible move is neither lowering the bar nor shopping for backbones. Document that zero-shot 8B yes/no rated discrimination is below the registered bar (consistent with TALLRec and CoLLM ICL) and move the method study to the regime where published yes/no models clear 0.60 UAUC (TALLRec FT 0.682 on ML-1M per CoLLM Table 2), registering that gate now.
- Protocol: On F0 or F1: (1) Write decision GATE_FAIL_AFTER_REMEDY to PILOT_LOG and decision.json. (2) Zero-shot rated numbers (ECE, Platt slope, UAUC, |logit| AUROC) become descriptive observation-study results only, re-measured on the reserve users and labelled 'channel below registered bar'. (3) Gate-FT, registered now: TALLRec-style LoRA yes/no SFT (r=16, alpha=32, 1 epoch, 3 seeds; 4-bit or bf16 with gradient checkpointing on the 4090). Prompt: V* under F1, V0 under F0. Training panels per amendment D (exclude Lumen valid/test users; temporal split T=p80). Pass if ML-1M UAUC(raw like), averaged over seeds, is >= 0.60 on the pilot users' test-period candidates; report all seeds plus CPU item-mean and MF UAUC as context. (4) If Gate-FT passes, the MIRROR question becomes mirror-tuning vs paraphrase-tuning at equal data and steps, with a pooled-seed user-bootstrap CI, 3/3 seed sign agreement and gain > 2 sigma_seed. Claim C1 is rescoped to fine-tuned only. (5) If Gate-FT fails, close the rated-panel method line and pivot the spine to the observation study plus the C-CRP next-item channel.
- Risk: Fine-tuning may overwrite the zero-shot acquiescence structure, changing what MIRROR means. Seed variance (sigma about 1.5pt in the sibling project) can swamp small effects. Rescoping C1 weakens the 'zero-shot and fine-tuned' claim.
- Paper value: Keeps a method paper alive under failure without any outcome-dependent rule change. The fine-tuned regime is also where TALLRec/CoLLM-familiar reviewers expect the yes/no results.

### Threshold-honesty package: literature anchors plus CPU references, reported but never decisive  (≈0 GPU-h)
- Rationale: The 0.60 UAUC bar was an unanchored efficacy bar presented as a sanity check. Saying so openly, with a table of published UAUC/AUC anchors and same-panel CPU references, turns the failure into context without amending anything.
- Protocol: Add to Amendment 2 and to the paper appendix an anchors table: TALLRec zero-shot ICL AUC 0.46-0.53 (ML-100K); CoLLM ML-1M ICL UAUC 0.527, TALLRec-FT UAUC 0.682, MF 0.636; Kang et al. zero-shot global AUC 0.68-0.71 vs user-average heuristic 0.727, showing that global AUC is inflated by user leniency; ReLLa zero-shot global AUC 0.67-0.70. Note that our panel drops 3-star candidates, so numbers are not exactly comparable. On CPU, compute on the same pilot and reserve rated panels the item-mean-rating UAUC and an MF UAUC trained only on ratings strictly before each user's first candidate and excluding all panel candidate events. Pre-declare them as reported context, not gates. In limitations, state the design flaw (broken vs weak conflated) and the corrected future-gate design: integrity test (CI lower bound > 0.5, mass >= 0.99, censoring ~0) separated from an efficacy bar anchored to a pre-measured reference.
- Risk: Reviewers may read a disclosed gate failure as weakness. This is mitigated by showing it is literature-consistent and that the protocol, not the outcome, governs claims. The CPU MF reference may exceed the LLM channel, which supports the 'item prior dominates' reading and must be reported either way.
- Paper value: Pre-empts the 'why 0.60?' and 'did you move the goalposts?' reviews, and contributes a reusable lesson for LLM4Rec evaluation (UAUC vs global AUC anchors).

## Citations
- Bao, Zhang, Zhang, Wang, Feng, He. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. RecSys 2023. arXiv 2305.00447. Verified this session (PDF): zero-shot ICL AUC Movie 0.46/0.49/0.53/0.50; Book 0.53/0.46/0.46/0.50; liked/disliked split prompt; >3 like rule; 64-shot 0.6748.
- Zhang, Feng, Zhang, Bao, Wang, He. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. IEEE TKDE 37(5), 2025. arXiv 2310.19488. Verified this session (PDF Table 2): ML-1M ICL AUC 0.5320 / UAUC 0.5268; TALLRec 0.7097 / 0.6818; MF 0.6482 / 0.6361; labels >3 positive.
- Kang, Ni, Mehta, Sathiamoorthy, Hong, Chi, Cheng. Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv 2305.06474 (2023; arXiv-only venue). Verified this session (PDF Table 2): ML-1M zero-shot global AUC Flan-U-PaLM 0.7084, ChatGPT 0.6794, text-davinci-003 0.6951; item-average 0.7395; user-average 0.7266; >=4 positive.
- Lin et al. ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation. WWW 2024. arXiv 2308.11131. Verified this session (PDF Table 2, Fig. 6): zero-shot ML-1M AUC Vicuna-7B 0.6739, Vicuna-13B 0.6993, ReLLa-SUBR 0.7013; zero-shot AUC peaks near K=15; SUBR uses LLM mean-pooled states, PCA d=512, cosine.
- Hou et al. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024. arXiv 2305.08845. Abstract verified this session (order perception, popularity/position bias, prompting fixes).
- Sclar, Choi, Tsvetkov, Suhr. Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. ICLR 2024. arXiv 2310.11324. Verified via search this session.
- Mizrahi, Kaplan, Malkin, Dror, Shahaf, Stanovsky. State of What Art? A Call for Multi-Prompt LLM Evaluation. TACL 12:933-949, 2024. arXiv 2401.00595. Verified via search this session.
- Cawley, Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11:2079-2107, 2010. Verified via search this session.
- Nosek, Ebersole, DeHaven, Mellor. The preregistration revolution. PNAS 115:2600-2606, 2018. Verified via search this session.
- Simmons, Nelson, Simonsohn. False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant. Psychological Science 22(11), 2011. NOT re-verified this session (well-known).
- Gelman, Loken. The garden of forking paths (2013 working paper) / The statistical crisis in science, American Scientist 102(6), 2014. NOT re-verified this session (well-known).
- Burns et al. Discovering Latent Knowledge in Language Models Without Supervision (CCS). ICLR 2023. arXiv 2212.03827. Taken from idea-stage/NOVELTY_DOSSIER.md (verified there), not re-checked this session.
- The yes-no bias of LLMs reflects answer order and wording (crossed symmetrization). arXiv 2607.05552. Taken from NOVELTY_DOSSIER.md, not re-checked this session.
- Internal: D:/Research/Lumen/outputs/confrec_pilot/pilot1/{ml1m_rated,toys_rated}.pilot_mirror.json, decision.json, ml1m_rated.report.json; src/confrec/{prompting,build_rated_panels,pilot_mirror,pyes_scorer}.py; scripts/sigir/{run_pilot1_mirror.sh,pilot1_gate.py,panel_reference.py}; idea-stage/{triage_verdict.md,PREREG_AMENDMENT_1.md,NOVELTY_DOSSIER.md}; docs/sigir/PILOT_LOG.md.
- Deliberation artifact written this session: D:/_Organized/Temp-Review/_RootDirs/temp/claude/D--/22c9b1b2-12a5-4956-a490-79bcb9beb0d8/scratchpad/deliberation/gatefix_prompt_variants_v1.json (machine-readable V0-V7 prompt spec).