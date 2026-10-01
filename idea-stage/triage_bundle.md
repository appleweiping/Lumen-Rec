# ARIS Phase-4 Triage Bundle (devil's-advocate jury) — Lumen-Rec → SIGIR 2027

Reviewer backend: fresh Claude Opus 5.5 (review_independence = same-family; acceptance_status = provisional).

## Inputs (read in full; absolute paths)
- Candidate ideas, generator A: D:\Research\Lumen\idea-stage\brainstorm_A.md  (11 ideas, "[gen: opus-A]")
- Candidate ideas, generator B: D:\Research\Lumen\idea-stage\brainstorm_B.md  (11 ideas, "[gen: opus-B]")
- Context: D:\Research\Lumen\RESEARCH_BRIEF.md ; D:\Research\Lumen\idea-stage\brainstorm_bundle.md
- Literature: D:\Research\Lumen\docs\sigir\LIT_HOOI_GROUP.md ; D:\Research\Lumen\docs\sigir\LIT_RECENT_UNC_LLM4REC.md
- Fresh pilot evidence: D:\Research\Lumen\docs\sigir\PILOT_LOG.md (CPU complementarity pilot, real numbers)

## Phase-3 mechanical consolidation (executor; NO quality filtering applied)
Near-duplicate clusters (keep both members visible; judge them jointly):
- C1 Thinking-mode confidence: A11 ≈ B11.
- C2 Exposure/MNAR-confounded calibration: A7 ≈ B1.
- C3 Feedback loop / echo chamber: A6 (KuaiRec ground-truth loop), B2 (self-confirming calibration fixed
  point + propensity fix), B10 (abstention feedback loop).
- C4 LLM↔CF serving decisions: A4 (calibrated log-odds fusion), A5 (confidence = redundancy), A9 (evidence-
  gated serving), B3 (ask the fallback + winner's-curse-corrected oracle).
- C5 Familiarity/name mechanism: A2 (pseudonym knockout + name-dropout LoRA), A3 (pretraining exposure via
  OLMo + infini-gram), B5 (concept-erasure mediation).
- C6 Training-data pruning: A8 (natural-noise oracle), B4 (prune the familiar).
- C7 Cross-user calibration: A10 (BUCE), B7 (user leniency).
- Singletons: A1 MIRROR; B6 selective-serving silences niche users; B8 decision-weighted proper scoring;
  B9 counterfactual personalisation confidence.
Objective feasibility gate: all 22 are within budget (single GPU, weeks); none dropped.

Executor annotations (input, not verdicts):
- prior_work: C4 neighbours RouteRec (negative), S-LLMR, ReDAct, PerK, Zou et al. KDD'26; C3 neighbours
  EchoTrace/RecLoop (no confidence), UGR (no loop); C5 neighbours KnowSA (SIGIR'26), "Incumbent Advantage"
  (arXiv 2606.17443, snippet only), popularity-steering (SPREE/PopSteer); C6 neighbours UNIT, GORACS,
  DEALRec, RosePO; A1 neighbours contextual calibration (Zhao et al. 2021) / PMI-DC (surface-form competition).
- so_what (real data): CPU pilot shows per-user oracle LLM↔LLMEmb = +0.053..+0.077 NDCG@10 (+23..+40%) on
  4/4 domains; graph/CF baselines beat the LLM almost only on HEAD targets → complementarity is popularity-
  structured. Prior cheap verbalized-p gate captured ~none of it (corr 0.07).
- effort_note: frozen Lumen panels (10k users × 4 domains) + all 8 official baselines' rankings exist locally
  (no re-run needed); token P(Yes) scorer written; GPU server type still unknown (assume 1×A100-80G or 4090).

## Your task (devil's advocate; rank, do not rewrite)
For each candidate (or cluster), make the strongest case both ways:
- What is the best case FOR it — what would make this the paper people cite at SIGIR?
- What's the strongest objection a reviewer would raise?
- What's the most likely failure mode?
- Is the prior_work note a real novelty problem, or differentiable?
- Rank by expected information and upside within the pilot budget — which results would matter most,
  whichever way they come out?
- Which 2-3 would you actually work on, and why? Which ONE could carry a full SIGIR paper as
  "observation study → one principled method → consistent gains on 3–4 datasets × ≥2 backbones"?

Rank; do not rewrite. An objection is answered or recorded as a named risk on the idea — never absorbed by
adding a module, a gate, or a qualifier. A bold idea with a named risk outranks a hedged idea with none, and
complexity added since the brainstorm is a red flag, not progress. Do not let your picks be uniformly the
safest — if the top set is all LOW-risk, name the high-upside idea that most deserves a pilot slot and what
result would convince you.

Also: the user's six seed questions (S1 correct-but-unsure; S2 wrong because low confidence; S3 echo chamber;
S4 yes/no confidence vs ground truth; S5 popular-high/niche-low; S6 prune by uncertainty) must all be
ANSWERED somewhere in the final paper (main body or appendix), but the paper must have ONE spine. State which
spine best lets all six be answered without becoming a kitchen sink.

Output format (markdown, to the path given in your instructions):
1. A ranked table of all candidates/clusters (rank, id, one-line verdict, risk, expected info value).
2. Per-candidate both-ways analysis (concise).
3. Top-3 pilot picks with a pre-registered pilot design each (≤2 GPU-h; decision criterion: what positive,
   negative and null would each teach).
4. Recommended paper spine + where S1–S6 each land.
5. Final line exactly: `Triage-Verdict: <top-1 id> | review_independence: same-family`.
