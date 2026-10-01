# ARIS Phase-2 Idea-Generation Bundle — Lumen-Rec → SIGIR 2027

You are a senior IR/ML researcher brainstorming research ideas. (Reviewer backend: fresh Claude Opus 5.5
instance; review_independence = same-family, acceptance_status = provisional.)

## Research direction (user's own words, seed idea #4 of the lab)
Uncertainty-aware LLM4Rec. "The LLM may generate correct output, but is it sure with the result? Is the
wrong answer wrong because of low confidence? Will high confidence cause an echo-chamber effect? Check the
output yes/no confidence and see the correlation with ground truth. Popular items with high confidence while
non-popular with low confidence? Prune training based on uncertainty due to noise?" Target venue: **SIGIR 2027
full paper** (9 pages incl. appendix + refs, ACM sigconf, double-anonymous; deadline Jan 21 2027). Scale/quality
should match Bryan Hooi group calibration papers (ICLR'24 confidence elicitation; ProCal NeurIPS'23; ConfTuner
NeurIPS'25) but the contribution must be OURS and original.

## Read these files for full context (absolute paths)
- D:\Research\Lumen\RESEARCH_BRIEF.md  (problem, constraints, prior positives/negatives — READ FIRST)
- D:\Research\Lumen\docs\sigir\LIT_HOOI_GROUP.md  (Hooi-group survey; §8 = outside competitors)
- D:\Research\Lumen\docs\sigir\LIT_RECENT_UNC_LLM4REC.md  (~70 recent papers, gaps, routes around negatives)
- D:\Research\Lumen\Paper\CALIBRA_draft.md  (prior internal audit draft — note its label-mismatch flaw)
- D:\llm4rec\uncertainty-llm4rec\docs\NEGATIVE_RESULTS.md and ADVISOR_MEMO.md (sibling project's exhaustive
  negative results on uncertainty-as-training-signal for generative DPO recommenders)

## Landscape (condensed)
- **Hooi group**: NO paper on uncertainty/calibration in LLM recommenders (lane open). Transferable templates:
  confidence elicitation (verbalized/sampling/aggregation; verbalized clusters 80–100%), ProCal (low-density
  samples are MORE overconfident at matched confidence; Bias-Index; PIECE metric; plug-in fix), ConfTuner
  (tokenized Brier proper-scoring fine-tuning; downstream cascades), UNIT (uncertainty-based pruning of
  training claims), MKJ (yes/no judgment skepticism, worse on low-frequency entities), NeighborAgg (graph
  neighbourhood trust + conformal mislabel detection).
- **Closest competitors (do-not-claim list)**: UQRec (WWW'25: listwise PL-entropy uncertainty, prompt-vs-rec MI
  decomposition, uncertainty-aware prompting); UGR (KDD'26: confident-error-penalising RL reward, difficulty
  re-weighting, confidence tokens; already does request rejection/tail truncation/rerank on 1 dataset, no
  popularity analysis); EviRank (2026: position-level confidence + NDCG-discount calibration); Ravikumar (CIKM'26
  short: verbalized-confidence calibration of ZERO-SHOT frontier LLM recs stratified by popularity, correctness =
  in-catalog; conformal abstention barely helps); KnowSA (SIGIR'26: estimate LLM item knowledge, augment only
  low-knowledge items; popularity is a poor proxy of LLM knowledge); GORACS (KDD'25)/DEALRec (SIGIR'24) data
  selection; SPRec/Flower (DPO/SFT amplify popularity bias); D3 (EMNLP'24 decoding bias); RosePO (soft label
  smoothing of uncertain prefs); PerK (WWW'24: calibrated per-user list length); EchoTrace/RecLoop (feedback-loop
  sims, no confidence); RouteRec (SIGIR'26 workshop: routing requests to LLM did not help).

## Key gaps (no published work found)
1. Calibration / reliability / popularity analysis of **token-level P(Yes)** in yes/no LLM recommenders
   (TALLRec/CoLLM/BinLLM use P(Yes) only for AUC).
2. **Confidence × popularity × feedback loop (echo chamber)** jointly.
3. **Cross-user decisions** with LLM-rec uncertainty: risk–coverage, abstention with fallback to CF/sequential
   models, conformal list truncation (oracle LLM↔CF router = +21% NDCG@10 in our data; cheap gate failed).
4. **Multi-seed-rigorous** study of uncertainty-based pruning/re-weighting of noisy fine-tuning data (literature:
   soft weighting > dropping; GORACS shows loss/gradient scores below random on TALLRec-style tuning).

## Hard constraints / lessons (do not re-propose dead ideas without a new angle)
- Within-list monotone calibration is rank-preserving → uncertainty as a ranking multiplier is inert.
- Popularity-conditioned post-hoc recalibration gave no gain; KnowSA suggests LLM familiarity/pretraining
  exposure, not dataset popularity, is the right variable.
- Uncertainty pruning ≤ random-25% prune in generative DPO; seed σ≈1.5pt → claims need ≥3–5 seeds.
- Lumen's existing score p_i is VERBALIZED (parsed JSON), not P(Yes); held-out next-item labels (base rate 1/101)
  make per-candidate ECE ill-posed — labels must match the question asked, or probabilities must be list-normalised.
- Compute: ONE rented GPU server (type TBD; assume 1×A100/A800-80G or 1×4090-24G), open 7–8B backbones
  (Qwen3-8B, Llama-3.1-8B, + one more family), LoRA. Data: Amazon-2023 categories with ratings, MovieLens,
  etc. ~4–6 weeks of compute before the deadline.

## Your task
Generate 8–12 concrete research ideas. For each idea:
1. One-sentence summary
2. Core hypothesis (what you expect to find and why)
3. Minimum viable experiment (cheapest way to test it, ideally ≤2 GPU-hours pilot)
4. Expected contribution type: empirical finding / new method / theoretical result / diagnostic
5. Risk level: LOW / MEDIUM / HIGH
6. Estimated effort: days / weeks / months
7. Which of the user's six seed questions it answers

Prioritize ideas that are:
- Testable with moderate compute (single GPU, weeks).
- Likely to produce a clear positive OR negative result (both are publishable).
- Simple at the core: one mechanism, few moving parts — an idea a colleague could restate after hearing it once.
  If the novelty only appears once a second module or an extra gate is added, that is packaging, not novelty.
- Aware of the papers above — awareness, not avoidance.

"Apply X to Y" is legitimate when the application would reveal something non-obvious. A direct, well-executed
attack on a central problem is valid when nobody has executed it well. Be genuinely creative: surprising
connections, inverted assumptions, questions nobody thought to ask. Generate first, filter later. A bold,
creative idea with a named risk beats a hedged, complicated one with none. A great idea is one where the answer
matters regardless of which way it goes. SIGIR also rewards one idea that can carry a full paper: an observation
study that motivates ONE principled method with consistent gains on 3–4 datasets.

Write your ideas as markdown to the output path given in your instructions.
