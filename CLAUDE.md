# CLAUDE.md

Read `AGENTS.md` first. It is the authoritative operating contract for this repository.

This is the Uncertainty project: Actionable Uncertainty for LLM-based Recommendation.

## ACTIVE LINE (2026-10-01): SIGIR 2027 — uncertainty-aware LLM4Rec (branch `sigir2027`)

- **Server:** `ssh lumen-gpu` (SeetaCloud `connect.westc.seetacloud.com:45105`, key auth only; never store
  passwords). Data/models under `/root/autodl-tmp/lumen`. Old `pony-rec-gpu` is gone; regenerate data with
  `scripts/sigir/{bootstrap_server.sh,slim_amazon2023.py,rebuild_panels.sh}` (panels verified against
  `docs/sigir/panel_refs/`). Server checkout: `git clone -b sigir2027 https://github.com/appleweiping/Lumen-Rec.git`.
- **ARIS state (executor + reviewer = Claude Opus; all verdicts same-family / provisional):** `RESEARCH_BRIEF.md`
  → `idea-stage/IDEA_REPORT.md` (selected A1 MIRROR, conditional on Pilot 1; fallback B9/A9) → pilots
  `scripts/sigir/run_pilot1_mirror.sh`, `run_pilot2_3.sh` (pre-registered rules in `idea-stage/triage_verdict.md`
  §3) → `refine-logs/EXPERIMENT_PLAN.md`. Evidence log: `docs/sigir/PILOT_LOG.md`. Paper: `Paper/sigir2027/`.
- **Code:** `src/confrec/`; tests `tests/test_confrec_*.py`.
- **Sync:** the working folder `D:\Research\Lumen` has a damaged `.git`; SIGIR work is mirrored into the clean clone
  `D:\Research\_lumen_fresh` (branch `sigir2027`) with `scripts/sigir/sync_to_clean_clone.ps1` and pushed from there.

## Continuity Rule

Use `agentmemory` for shared recall, but do not depend on it as the only durable
state. If memory is stale, empty, or unavailable after a restart, recover from
GitHub `main`, `AGENTS.md`, canonical docs, local evidence packages/manifests,
and fresh pony-rec-gpu audits. Important checkpoints must be mirrored outside
agentmemory by committing/pushing related local changes, updating canonical
docs when status or next actions change, and copying or manifesting important
server evidence locally before cleanup.

## Quick Orientation

- **Current collaboration routing**: Claude reviewer tooling is unavailable in
  this thread; when multi-agent review is required, use GPT-5.5 xhigh
  sub-agents instead.
- **Stage**: M5/M6 handoff (C-CRP v3 and sports/toys/home/tools official gates passed; Phase 2.5 paper-critical modules closed; local submission-package/release-candidate stack ready, but final submission blocked by ProMax public page-range/DOI-Crossref visibility and private manual submission-system confirmation)
- **Core Claim (REFRAMED, dual framing — authoritative)**: A task-grounded **pointwise LLM relevance posterior** (the model's own per-candidate relevance probability, used directly as the ranking score) is the working reranking signal — it ranks **first in 6 of 8 domains** (+21.6% to +53.2% NDCG@10 over the strongest baseline; competitive but not first in Beauty #2 / Movies #5). The **calibrated-uncertainty / risk decomposition does NOT improve same-candidate ranking** — that is reported as a **characterized negative result**. This is a scoped same-candidate reranking claim, not full-catalog SOTA and not an ECE/calibration-guarantee claim. C-CRP now expands to **Candidate-Conditioned Relevance Posterior** (not "Calibrated"). The old "calibrated uncertainty improves ranking" wording is **stale/reversed** — see `docs/paper_claims_and_status.md` (source of truth).
- **Methods**: C-CRP v3 = pointwise relevance posterior (main/headline); the uncertainty/risk family is an analyzed-and-inert design space (negative result). SRPD = trainable framework (ablation/supplementary).
- **Baselines**: 8 official external in the main block (ELMRec, IRLLRec, LLM2Rec, LLMEmb, LLMESR, ProEx, ProMax, RLMRec). ProEx (KDD 2026) and ProMax (SIGIR 2026) share one official repo (BlueGhostYi/ProRec). SETRec is a 9th baseline run/attempted but EXCLUDED (official_blocked, `setrec_upstream_tokenize_all_cuda_oom` CUDA-OOM on large domains), held supplementary unless a future official gate passes.
- **Domains (all 8 in the main comparison)**: beauty, books, electronics, movies (original 4) + sports, toys, home, tools (new 4). Diagnostic modules (ablation/hyperparameter/observation/raw-p attribution) are scoped to the 4 new domains with full per-event signal rows.
- **Server**: `pony-rec-gpu` (SSH accessible via `ssh pony-rec-gpu`, key-based auth configured)
  - Server project path: `~/projects/pony-rec-rescue-shadow-v6`
  - Local project path: `D:\Research\Uncertainty`
  - SSH config: `125.71.97.70:15302`, user `ajifang`
  - GPU: RTX 4090 (49GB VRAM)

## Experiment Roadmap (2026-06-05)

### Phase 1: C-CRP v3 Scoring (COMPLETE)
Run C-CRP v3 on all 8 domains with Qwen3-8B via vLLM.
- beauty (973 users): DONE — HR@10=0.229
- books (10k users): DONE — HR@10=0.476 **SOTA**
- electronics (10k users): DONE — HR@10=0.299 **SOTA**
- movies (10k users): DONE — HR@10=0.208
- sports (10k users): DONE — HR@5/10/20=0.275/0.382/0.517, NDCG@5/10/20=0.198/0.233/0.267, MRR=0.208
- toys (10k users): DONE — HR@5/10/20=0.317/0.396/0.506, NDCG@5/10/20=0.245/0.271/0.298, MRR=0.250
- home (10k users): DONE — HR@5/10/20=0.156/0.226/0.351, NDCG@5/10/20=0.110/0.132/0.164, MRR=0.126
- tools (10k users): DONE — HR@5/10/20=0.194/0.270/0.393, NDCG@5/10/20=0.142/0.166/0.197, MRR=0.156

For new-domain C-CRP v3 status reports, always include the full metric set:
HR@5/@10/@20, NDCG@5/@10/@20, MRR, `n_users`, `n_prompts`, data path, and
score/rank row counts. Do not summarize only @10.

Original-domain C-CRP v3 reports are not in the new-domain flat directory
layout. Use `outputs/ccrp_v3_formal/<domain>/report.json` for
beauty/books/electronics/movies and `outputs/ccrp_v3_formal/main_comparison_table.csv`
for their metric-complete 8-baseline comparison. As of the 2026-05-31 artifact
audit, those old-domain baseline metrics exist, but some method-specific
evidence packs still need provenance/audit reconciliation under the current
strict gate.

### Phase 2: Official Baselines on New Domains
Run 8 official baselines on sports/toys/home/tools (same protocol as original 4).
Script: `scripts/run_baselines_new_domains.sh` (already on server).
The script is aligned to the canonical 8-method block and excludes SETRec while
SETRec remains blocked/supplementary in `configs/official_external_baselines.yaml`.
It uses `${PYTHON:-/home/ajifang/miniconda3/bin/python}` to avoid non-interactive
SSH sessions failing on a missing bare `python`.
Use the documented single-domain loop under current storage pressure, e.g.
`DOMAINS_OVERRIDE=sports bash scripts/run_baselines_new_domains.sh`. The runner
audits exact score coverage and imports complete `@5/@10/@20 + MRR` metrics
after each completed score file.
Current status (2026-06-05): sports, toys, home, and tools are each 8/8
official baselines complete and have passed their domain/comparison/paired-test
gates against C-CRP. The Tools artifacts are
`outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}` and
`outputs/summary/tools_official_ccrp_20260605_*`; the paired summary records
`claim_gate=tools_domain_pass`, C-CRP observed-best on all seven metrics, and
56/56 C-CRP-vs-official paired tests positive and Holm-significant. The compact
four-new-domain gate certificate is
`outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`; it supports
the narrow same-candidate Sports/Toys/Home/Tools claim only, not paper-ready or
full-catalog SOTA wording. Do not start more official baseline rows unless a
later audit finds a concrete invalid row.

Storage note: before the home launch, completed sports/toys LLMEmb and LLM-ESR
upstream staging directories were removed only after final server audits and
local-light packages passed. The cleanup manifest is
`outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`;
final official evidence directories and local lightweight packages were
preserved. Disk recovered from about `5.9G` to `17G` free.

### Phase 3: Full Comparison Table + Statistical Tests
The four-new-domain official+C-CRP compact gate summary exists at
`outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`. Remaining
Phase 3 work is paper-facing table integration and any all-domain/reconciled
artifact packaging required by the final claim boundary.

### Phase 2.5 / Paper-Critical Method Evidence
Before claiming paper readiness, complete four method-story gates:

- observation/motivation study showing why uncertainty is needed, using
  representative completed baselines/domains under fair same-candidate settings
  and producing a paper-ready figure or table;
- leave-one-component-out C-CRP ablations over the actual nontrivial
  implementation components, including score mode, boundary uncertainty,
  calibration gap, evidence support/insufficiency, counterevidence, risk
  penalty, eta, confidence weight, and C-CRP weight triples where supported;
- hyperparameter sensitivity curves for real controls such as eta, C-CRP
  weights, confidence weight, uncertainty gates/thresholds, anchor penalties,
  and SRPD learning-rate/lambda controls when SRPD rows are used;
- a clean framework overview figure showing the full pipeline and where
  uncertainty/components enter the ranking decision.

Each finished module needs commands, configs, seeds when applicable, git
commit, row counts, provenance notes, plots/tables, key logs, and a
lightweight local evidence package with server/local manifest comparison.

### Phase 4: Paper Writing (ARIS paper-write skill)

### Phase 5: GPT-5.5/Codex Review Cycle (must reach 8/10)

## Project Goal

本项目的最终目标是产出一篇达到顶会投稿水平的论文。具体要求：

1. **诚实的强结果（非"每个域都 SOTA"）** — C-CRP v3 的 pointwise posterior 在 **8 个域中的 6 个**排第一（超越全部 8 个 official baselines，+21.6%~+53.2% NDCG@10）；Beauty(#2)/Movies(#5) 如实报告为落后域。**严禁**写成"first in every domain / 每个域都 SOTA"。uncertainty/risk 分解是 characterized **negative result**，不得作为 headline 贡献。
2. **创新非缝合** — 方法必须是原创的、有理论动机的，不是拼凑已有方法
3. **禁止 toy 化** — 所有实验必须是 full-scale（10k 用户、101 候选），不能用小样本代替
4. **8 个 official baselines** — 每个域都必须有完整的 8 个 baseline 对比
5. **公平比较** — 所有 setting 对齐（用户数、候选数、指标 @5/@10/@20、统一 backbone）
6. **GPT-5.5 review 达到 8/10** — 按 ARIS 审核标准，多维度评价，总分必须 ≥8
7. **实验做完再写** — 不能实验做一半就提交 review，必须结果完整后再写作
8. **每步 commit + memory** — 每个关键产物都要 commit 到 GitHub，更新项目文档
9. **目标续跑监控** — 长期项目使用当前线程 heartbeat 每 2 小时激活一次；每次续跑只做一个有界监控周期，禁止刚结束就连续自触发或无脑重复检查
10. **服务器跑实验，本地不跑** — GPU 实验全在服务器，本地只做版本控制和写作
11. **motivation/ablation/hyperparameter/figure 四件套必做** — 这些是 paper-ready 前置门禁，不是写作润色项

## Artifact Management Rules

### 本地保留（commit 到 GitHub）：
- `report.json` — 每个域/方法的指标结果
- `user_ranks.jsonl` — 每用户排名（统计检验用）
- `main_comparison_table.csv` — 对比表
- `fairness_provenance.json` — baseline 公平性证据
- `*_score_audit.json` — 分数审计
- `*_run_summary.json` — 运行摘要
- Paper 源文件、scripts、configs、docs

### 只留服务器（不下载、不提交）：
- `scores.csv`（87MB/域，可重新生成）
- `predictions/`（600MB+）
- `embeddings/`、`checkpoints/`
- 原始数据、模型权重

### 不提交到 GitHub：
- API keys、credentials
- `__pycache__/`、`.pyc`、editor swap files
- 上述大文件

## Your Role (Claude Code / Opus / Sonnet)

### ⚠️ 接手前必做：检查服务器运行状态

**每次新会话开始，第一件事检查服务器是否有正在运行的实验。不检查就操作 = 违规。**

```bash
ssh pony-rec-gpu "ps aux | grep python | grep -v grep | grep -i 'pony-rec\|ccrp\|baseline\|uncertainty'"
ssh pony-rec-gpu "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
ssh pony-rec-gpu "tail -5 ~/projects/pony-rec-rescue-shadow-v6/ccrp_v3_all_domains.log 2>/dev/null"
```

**严禁：** 不检查就启动新实验、kill 正在运行的进程、重跑已完成的实验。
**正确做法：** 确认 GPU 空闲 → 检查哪些域已完成 → 再决定下一步。

When invoked by Codex or OpenCode as a reviewer:
- You are **read-only**. Do not edit files, stage, commit, or push.
- Challenge rigor, novelty, technical depth, ablation completeness, baseline coverage, leakage risk, table eligibility.
- Reviewer objections remain blockers until addressed.
- Apply top-conference reviewer standards (RecSys/SIGIR/WSDM/KDD/NeurIPS).

When the user opens Claude Code directly with write scope:
- Follow `AGENTS.md` operating contract fully.
- Update canonical docs after meaningful changes.
- Run readiness checks before pushing.

## Key Docs (Read Order)

1. `AGENTS.md` — full operating contract (280 lines)
2. `docs/milestones/README.md` — M0-M6 milestone map
3. `docs/paper_claims_and_status.md` — frozen claims, status labels
4. `docs/top_conference_review_gate.md` — standing reviewer gate
5. `docs/server_runbook.md` — server execution patterns
6. `configs/official_external_baselines.yaml` — baseline configs

## Evidence Rules (Summary)

- Only `completed_result` rows enter main tables
- Score gates: exact key match, no missing/extra/duplicate, finite scores, coverage=1.0
- Paired statistical tests required for significance claims
- `official_completed` requires: pinned repo/commit, default hparams, full provenance, score coverage=1.0
- When in doubt, downgrade the claim, not the evidence standard

## Multi-Agent System

This project participates in the 6-agent collaboration system:

| Agent | Role in this project |
|-------|---------------------|
| OpenCode (像素猫) | Implementation, analysis, comparison builders, doc updates |
| Codex | Coordination, parallel execution, server command generation |
| Opus | Architecture review, complex reasoning, security/leakage audit |
| Sonnet | Code review, table eligibility checks, second opinion |
| Haiku | Quick lint, format checks |
| DeepSeek (鲸鱼) | Translation, bulk text, Chinese content |

Agent Hub daemon on port 9800 provides real-time collaboration when running.
