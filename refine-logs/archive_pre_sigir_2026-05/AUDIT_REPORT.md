---
created: 2026-05-25
verdict: PASS
---

# Experiment Audit Report

## Verdict: PASS

All main-table results satisfy the evidence requirements for paper submission.

## Evidence Inventory

### Main Table (C-CRP v3 vs 8 Official Baselines × 4 Domains)

| Item | Status |
|------|--------|
| C-CRP v3 Beauty (973 users, 101 candidates) | ✅ completed_result |
| C-CRP v3 Books (10,000 users, 101 candidates) | ✅ completed_result |
| C-CRP v3 Electronics (10,000 users, 101 candidates) | ✅ completed_result |
| C-CRP v3 Movies (10,000 users, 101 candidates) | ✅ completed_result |
| 8 Official Baselines × 4 Domains (32 rows) | ✅ completed_result |
| Same-candidate protocol enforced | ✅ |
| Same backbone (Qwen3-8B) for all | ✅ |
| Full metrics (@5/@10/@20 + MRR) | ✅ |

### Calibration Observation (Motivation)

| Item | Status |
|------|--------|
| C-CRP v3 diagnostic (ECE/AUROC/Brier) — 3 domains | ✅ completed |
| Beauty shadow_v1 diagnostic | ✅ completed |
| Baseline calibration diagnostic (4 methods × 4 domains) | ✅ completed |
| Reliability bins (calibration curves) — 4 domains | ✅ completed |

### Ablation / Negative Results

| Item | Status |
|------|--------|
| Formal C-CRP (uncertainty decomposition) — worse than v3 | ✅ documented |
| C-CRP v4 (enhanced prompt) — worse than v3 on movies | ✅ documented |
| Temperature scaling — mathematically ineffective for ranking | ✅ documented |

## Protocol Compliance

- [x] All baselines use official code or official-code-level implementation
- [x] Unified backbone: Qwen3-8B for all methods
- [x] Same-candidate protocol: 1 positive + 100 negatives per user
- [x] Validation-only model selection (no test-set tuning)
- [x] Score coverage = 1.0 for all rows
- [x] No toy/subset data in main table (verified: beauty=973, others=10,000)

## Known Limitations (must be discussed in paper)

1. Movies domain: C-CRP v3 ranks #4-6, not SOTA (LLMEmb leads)
2. Beauty domain: C-CRP v3 ranks #2, slightly below ProEx
3. Single backbone (Qwen3-8B) — generalization to other LLMs not tested
4. No statistical significance tests with multiple seeds (single run)
5. No full-catalog evaluation (101-candidate protocol only)
