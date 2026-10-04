# One-month schedule (user directive 2026-10-03: finish within a month, i.e. a submittable draft by 2026-11-03)

SIGIR 2027 full-paper deadline is 2027-01-21, so Nov–Jan is buffer for review rounds and polishing. Everything below is
gated by the pre-registrations (`idea-stage/PREREG_AMENDMENT_{1,2}.md`); no step may be skipped by redefining a gate.

**Measured GPU throughput (RTX 4090 D 24 GB, Qwen3-8B fp16 prefill-only):** rated ML-1M 87 prompts/s, rated Toys
35/s, next-item 175/s with 4 questions per candidate (≈44 candidate-prefixes/s). Cost is dominated by the candidate
prefix, not the number of questions: the 2-question audit runs at ≈76 prompts/s (≈38 candidates/s), so one 10k×101
domain costs ≈7.4 GPU-h and the whole next-item audit (4 test domains + 4 valid-2k) ≈ **36 GPU-h**, ending about
Oct 5 if uninterrupted. One backbone, one GPU: schedule the GPU, not the humans.

## Phase map

| window | CPU/code | GPU | gate |
|---|---|---|---|
| Oct 3–5 | amendment-2 code integrated, synced, FREEZE hashes recorded | **next-item audit** (`run_nextitem_audit.sh`, ≈36 h, carve-out N; started 2026-10-03 16:55 local) | FREEZE_ACK in PILOT_LOG |
| Oct 5–7 | audit analysis code; forensics A2–A8 on Pilot-1 scores; paper protocol/related-work rewrite | gate-fix stage 1 (dev, ≈2 h) → stage 2 (confirm, ≈1.7 h) → diagnosis battery (≈1 h) | G5 FIX_FOUND? → G6 GATE_PASS? |
| Oct 7–12 | `ftgrid_*` code frozen and hashed (A3 section 0); Gate-FT data | **Gate-FT** (3 seeds, ≈ 5 h + scoring) | mean UAUC ≥ 0.65 |
| Oct 12–22 | decomposition + pseudonym knockout analyses; paper results sections | fine-tuned grid (Qwen: ML-1M, Toys, Games, Sports ×3 seeds), swap priors, knockout | claim admission rules (F) |
| Oct 15–22 (parallel CPU) | S6 pruning arms, nested method slot (prior-offset LoRA) | method slot + pruning arms | method slot kill date Nov 30 |
| Oct 22–28 | Llama-3.1-8B replication (2 datasets), fresh-domain replication | replication runs | admission needs sign replication |
| Oct 22–Nov 3 | full paper, appendix, 3× adversarial review rounds, number audit, anonymity/format audit | reruns only | submittable draft |

## GPU operations (server)

- **Queue.** `scripts/sigir/gpu_queue.sh` runs on the server (`/root/autodl-tmp/gpuq`): one job at a time, lowest file name
  first from `pending/`; the next-item audit (`run_nextitem_audit.sh`, frozen checkout) is the resumable **filler** and is
  preempted automatically when a job is pending, then resumes from its last finished 100-user chunk. Enqueue with
  `cp job.sh /root/autodl-tmp/gpuq/pending/20_name.sh`; logs in `gpuq/logs/`; `touch gpuq/PAUSE_FILLER` / `STOP`.
  Registered stages are shell scripts in `scripts/sigir/` (they carry their own skip-if-done markers).
- **Order of registered stages:** `run_gatefix.sh` stage 0 (CPU, run by hand: freeze hashes) → record hashes in PILOT_LOG →
  queue `FREEZE_ACK=1 run_gatefix.sh` (stage 1 and, on FIX_FOUND, stage 2) → queue `run_diag_battery.sh` →
  (`BENCH=1 run_gateft.sh` once, to time LoRA training) → queue `run_gateft.sh`.
- **Gate-FT cost (measured on a synthetic panel, 2026-10-04, carve-out O):** 3.84 examples/s, peak 19,958 MiB, so ≈ 23.3k
  pre-T DEV examples ≈ 1.7 GPU-h per seed (≈ 5 GPU-h for 3 seeds) plus ≈ 25 min of adapter scoring; the last-token loss
  (`src/confrec/lora_trainer.py`) agrees with the stock Trainer, accumulation included (tested). Real prompt lengths decide;
  `BENCH=1 run_gateft.sh` re-measures on the real panel.
- **Amendment 3 gating** (`idea-stage/PREREG_AMENDMENT_3.md` section 0): the amendment's sha1 must be in PILOT_LOG before the
  Gate-FT adapters are trained; scoring (Gate-FT stages C–D and the zero-shot panels of section 4) additionally waits for the
  record of the bound code (`ftgrid_data.py`, `ftgrid_report.py`, `run_ftgrid.sh`, ...) and the four `ftgrid_split.json`.
  Order of the program, budget (≈ 67–94 GPU-h) and cut checkpoints (2026-10-22, 2026-10-29) are in A3 section 10.

## Cut rules (if behind)
- Week of Oct 22: Llama cut to 2 datasets; fine-tuned grid cut to 3 datasets.
- Nov 30 kill date for the nested method slot stays; if killed the paper is the audit + decomposition only.
- Gate-FT FAIL closes the rated-panel method line: paper = zero-shot audit + full-scale next-item exposure audit.

## Standing rules
- Every GPU stage is a resumable script with completion markers; outputs mirrored locally after each stage.
- Sync after every milestone: local (`D:\Research\Lumen`) → clean clone → GitHub `sigir2027` → server `git pull`.
- All verdicts are same-family (Claude) and provisional; label them so.
