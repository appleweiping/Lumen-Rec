---
created: 2026-05-25
status: frozen
---

# Research Question

**Primary RQ:** Can task-grounded calibrated uncertainty improve the reliability of LLM-based candidate ranking under a controlled same-candidate evaluation protocol?

**Sub-questions:**

1. (Observation) Is verbalized LLM confidence informative but unreliable for recommendation tasks? Across how many domains does this hold?
2. (Method) Can evidence-grounded probability scoring (asking the LLM to estimate relevance probability with justification) produce better-calibrated and more discriminative signals than raw confidence or embedding-based scores?
3. (Evaluation) Under a fair same-backbone (Qwen3-8B), same-candidate (101 items), same-protocol comparison, does our scoring approach outperform 8 official external baselines?

**Target venue:** RecSys 2026 / SIGIR 2026 / WSDM 2027 (long paper, 9-10 pages + references)

**Core claim:** Task-grounded relevance probability scoring achieves state-of-the-art ranking performance on 2/4 Amazon domains and competitive performance on the remaining 2, while providing significantly better-calibrated uncertainty signals (lower ECE) than all baselines.
