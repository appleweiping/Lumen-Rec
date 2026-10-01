---
created: 2026-05-25
status: completed
---

# Experiment Plan

## Overview

Four-domain evaluation of task-grounded relevance probability scoring (C-CRP v3) against 8 official LLM-based recommendation baselines under a controlled same-candidate protocol.

## Setup

- **Model:** Qwen3-8B (local, vLLM batch inference)
- **Domains:** Amazon Beauty (973 users), Books (10,000), Electronics (10,000), Movies (10,000)
- **Candidates:** 101 per user (1 positive + 100 negatives)
- **Metrics:** HR@5, NDCG@5, HR@10, NDCG@10, HR@20, NDCG@20, MRR
- **Baselines:** ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLM-ESR, ProEx, ProMax, RLMRec

## Blocks

1. **Observation block:** Raw confidence diagnostic (ECE, AUROC, Brier, reliability bins) for C-CRP v3 and 4 baselines × 4 domains
2. **Main comparison block:** C-CRP v3 vs 8 baselines × 4 domains, full metrics
3. **Ablation block:** Formal C-CRP (uncertainty decomposition), v4 (enhanced prompt), temperature scaling — all documented as negative/null results

## Results Summary

- Books: C-CRP v3 SOTA (NDCG@10 +21.5% vs LLMEmb)
- Electronics: C-CRP v3 SOTA (NDCG@10 +52.5% vs LLMEmb)
- Beauty: C-CRP v3 #2 (NDCG@10 0.134 vs ProEx 0.151)
- Movies: C-CRP v3 #4-6 (NDCG@10 0.128 vs LLMEmb 0.169)
- Calibration: All baselines ECE 0.31-0.66; C-CRP v3 ECE significantly lower
