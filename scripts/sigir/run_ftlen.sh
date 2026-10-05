#!/usr/bin/env bash
# FT-L, the longer-training robustness arm (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_9.md; EXPLORATORY: no hypothesis, no Holm family,
# no claim-admission role) for ONE dataset (ML-1M or Toys): two adapters s0, s1 (seeds 0 and 1) trained like the registered adapters
# s0-s2 of the dataset (the section-2 recipe, read from their recorded train_config.json: the real-label registered TRAIN panel, LoRA
# r 16, lr 1e-4, effective batch 32, ...) except `--epochs 3`, scored with the `like` question on the whole EVAL panel and with the
# swap-prior arm (--swap_k 8) on S_d's TEST rows, with exactly the arguments of the real s0's passes; the zero-shot scores are the
# registered ones (linked, never rescored). A NEW script (no bound file changes; nothing of run_ftq.sh or ftq_panel.py is edited: what
# is shared is imported read-only). GPU server; one job at a time through scripts/sigir/gpu_queue.sh, one job per dataset:
#     cd /root/autodl-tmp/lumen-rec && bash scripts/sigir/run_ftlen.sh ml1m        (and toys)
#   usage: bash scripts/sigir/run_ftlen.sh D       D in ml1m, toys (the only argument)
#   env:   MODEL     Qwen3-8B dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); any other backbone is refused
#          OUT_ROOT  the FT-L root outputs/confrec/ftgrid_len3, the only root outside DRY_RUN. A relative OUT_ROOT resolves from the
#                    repo root (the script's working directory). It is compared canonically, in lower case on every platform
#                    (realpath -m: //, ., .., absolute paths, symlinks; on Windows also D:/... spellings): any spelling of that root
#                    is accepted, any other directory (the grid root outputs/confrec/ftgrid, ftgrid_len3/sub, ...) is refused, and so
#                    is that root reached through a link (outputs, outputs/confrec or ftgrid_len3 itself a symlink)
#          VARIANT   default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          STAGES    comma list of 1-3 or all (default all)
#          PYTHON    interpreter (skips the conda activation)
#          DRY_RUN   0 or 1 (any other value is refused with exit 2); 1 = a CPU rehearsal, see below
#          FTLEN_ALLOW_RETRY, FTLEN_RETRY_REASON   s0, s1 or s0,s1 and a non-empty reason: a deliberate retry of an adapter whose like or
#                    swap pass failed E1 twice (FAILED_INTEGRITY is sticky per seed, see stages 1 and 2); the retry is logged
#   reads (never written): the grid root outputs/confrec/ftgrid (panels/D: train.jsonl, eval.jsonl, eval_sd_test.jsonl,
#          ftgrid_split.json; adapters/D and scores/D of the real s0-s2: their train_config.json and the run.key and report config of
#          the real s0's like and swap passes; ML-1M: Gate-FT's adapters outputs/confrec/gateft/adapters and train.jsonl)
#   writes (the FT-L root only, outputs/confrec/ftgrid_len3): adapters/D/s0, s1; scores/D/zeroshot = a symlink to the registered
#          zero-shot scores (linked, never copied or rescored), scores/D/s0/{like,swap} and s1/{like,swap}; report/D.json (+ D_tables.csv),
#          report/NOTE_FTLEN.txt and, when a retry is allowed, report/NOTE_FTLEN_overrides.txt; build/D/ (step markers). Nothing below
#          outputs/confrec/ftgrid is written, and nothing is written through a link. What is covered: (1) every path this script
#          names below the FT-L root, with the temporary names ftgrid_report writes beside its products (report/D.json.tmp,
#          D_tables.csv.tmp), must resolve, links included, to the same place below the canonical root, else the run is refused (exit 2)
#          before it writes anything; (2) a sweep for links (find OUT_ROOT -type l) at the start and before each stage allows only the
#          scores/<dataset>/zeroshot that this script makes itself (asserted when made, and resolved to the registered zero-shot score
#          directory at the start); (3) the report's .tmp files, and any .tmp file in a scoring directory, are removed before
#          ftgrid_report or the scorer runs. The scorer's own temporary names (like/*.tmp, like/parts/*.tmp) are not enumerated: only
#          (2) and (3) cover them, and a bound tool that wrote another temporary name would be covered by (2) alone
#   gates    (before stages 1-3, in this order; any failure exits 4 and starts nothing): the recorded Gate-FT decision is GATE_FT_PASS
#            (Amendment 3 section 0); `ftgrid_freeze --check --stage amendment` and `--stage core` (the full record, a CPU check run
#            again at every start); the gate-fix decisions (selection.json, and gate.json when a fix was found); and the FT-L record
#            (addendum 9 section 3): the sha1 of this script, of src/confrec/ftlen_panel.py and of tests/test_confrec_ftlen.py are in
#            docs/sigir/PILOT_LOG.md. Imported or used as they are, and recorded separately (not part of that record):
#            src/confrec/ftq_panel.py (the recipe validation, the run.key and report-config readers, the link plumbing),
#            train_lora_yesno.py, pyes_scorer.py, ftgrid_report.py and src/confrec/ftgrid_extra.py (see below)
#   stage 1  (GPU) ftlen_panel recipe: the trainer flags the real adapters s0-s2 recorded in their train_config.json (ML-1M: Gate-FT's;
#            s0-s2 must agree and be the registered recipe, one epoch, with the split's length rule for the Amazon datasets), --train
#            kept as recorded (its bytes must be the registered train.jsonl) and `--epochs 3` for the recorded 1.0; --out and --seed
#            are the only other flags that change. The records of the real s0's like and swap passes are validated too before the
#            first training (ftlen_panel scoring), so a wrong record stops the job early. Then train s0, s1 (seeds 0, 1)
#            with train_lora_yesno; the trained config is checked against the real s0's (only --out, --seed and --epochs may differ)
#            and the adapter's provenance (ftlen.json: 3 epochs, the sha1 of the TRAIN panel) is written. An adapter with
#            train_config.json and weights is never retrained, but must carry that provenance (a registered 1-epoch adapter placed in
#            the root is refused). FAILED_INTEGRITY is sticky per seed: while scores/D/s<k>/like or /swap, or a .stale.* of either,
#            holds it (that pass failed E1 twice), making s<k> again is refused (exit 1, naming the directory) unless
#            FTLEN_ALLOW_RETRY=s<k> and a non-empty FTLEN_RETRY_REASON are set; the override is appended to
#            report/NOTE_FTLEN_overrides.txt (seed, UTC date, reason). A finished failed state is resumed without refusal
#   stage 2  (GPU) per seed: like on panels/D/eval.jsonl (CAL and TEST rows), then the swap-prior arm (--swap_k 8) on
#            panels/D/eval_sd_test.jsonl (S_d's TEST rows): pyes_scorer with exactly the arguments of the real s0's pass of the arm (its
#            run.key and the config of its report.json, nothing else of that report is read; ftlen_panel scoring); no other arm
#            (no-history, star permutation, knockout) is scored. E1 as run_ftgrid.sh (a failing run is moved to DIR.e1fail.<time>
#            and rerun once, a second failure of the same run.key leaves DIR/FAILED_INTEGRITY: the adapter is reported as missing, never
#            replaced; one deliberate difference: run_ftgrid.sh counts every DIR.e1fail.*, here only those whose run.key is the current
#            one, so a leftover of an earlier panel, adapter or argument vector does not turn a first failure into the final one); the
#            recorded scoring config of each pass equals the real s0's pass of the same arm except `lora` (ftlen_panel verify_scores).
#            The same sticky rule as stage 1: a pass that would be scored again under another run.key (a re-made adapter) while
#            FAILED_INTEGRITY sits in like/, swap/ or a .stale.* of that seed is refused (exit 1) unless the override is set.
#            Then the registered zero-shot scores are linked into the root (ftlen_panel link)
#   stage 3  (CPU) ftgrid_report (unchanged) --models zeroshot,s0,s1 --scores_root outputs/confrec/ftgrid_len3/scores --out
#            outputs/confrec/ftgrid_len3/report/D.json: s0 and s1 are the FT models of the report (two 3-epoch seeds: every statement that
#            needs the three registered seeds, the P1 decision and the E-B family, is INCOMPLETE or descriptive here, and no decision or
#            Holm result of this file is registered); report/NOTE_FTLEN.txt says so and says that the registered report of the same
#            dataset, outputs/confrec/ftgrid/report/D.json, is the one-epoch result. It is never touched (checked)
# ftgrid_extra is NOT run by this script. Its runner run_ftextra.sh refuses this root (it takes the three registered roots only, and a
# DRY_RUN root must lie outside outputs/confrec/ftgrid*), so the module is run by hand, as it is, for the blocks E-W (G_wu), E-J
# (dUAUC(L - q-hat)) and E-G (the e-share); it reads this root's like and swap arms of s0, s1 and the registered zero-shot link (the
# decomposition arms it does not need are absent here and are not loaded):
#     python -m src.confrec.ftgrid_extra build --domain D --split outputs/confrec/ftgrid/panels/D/ftgrid_split.json \
#       --panels outputs/confrec/ftgrid/panels/D --scores_root outputs/confrec/ftgrid_len3/scores --models zeroshot,s0,s1 \
#       --raw data/raw --out outputs/confrec/ftgrid_len3/extra/D.json --root_label llama --n_boot 2000
# --root_label llama is deliberate: the root is outside the registered ones, so any label is accepted by the module, and only `llama`
# keeps the file from being eligible for an A3-6 family (family_eligible_panel false, no FT_C_reading); the module also records that the
# Qwen backbone does not belong to that root as an input problem, which makes `summarize` refuse the file, as it must (an exploratory
# file is never summarized). The values are "two seeds, exploratory": the FT regime is incomplete (s2 is absent) and every block that
# needs three seeds says so.
# Re-runnable: a finished step is skipped (markers newer than their inputs); a scoring dir is skipped when report.json exists and run.key
# (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else it is moved to DIR.stale.<time>.
# GPU time (addendum 9 section 3, checked against the file times of the real adapters and passes): training scales with the epochs, the
# scoring is the registered one. ML-1M 2 x 3 x 46 min + 2 x 9 min like + 2 x 12.5 min swap = 5.3 h (addendum 9: about 5.6); Toys
# 2 x 3 x 48 min + 2 x 16 min + 2 x 42 min = 6.7 h (addendum 9: about 7.7).
# DRY_RUN=1: the same chain, CPU only, on run_ftgrid.sh's own DRY_RUN world of D (synthetic raw data, panels, Gate-FT context, s0-s2 and
# their like and decomposition passes, the temporary pilot log), built first by `DRY_RUN=1 STAGES=0,1,2,3,4 run_ftgrid.sh D`; skipped once
# it exists. Everything lies in a temporary directory outside outputs/confrec: tmp_outputs/ftlen_dryrun/{ftgrid: the world, ftgrid_len3:
# the FT-L root}. The DRY_RUN guard is an allow-list: an OUT_ROOT and a world must, in any spelling and through any link, lie under
# tmp_outputs of the repo, or outside the repo's parent directory (so never in the repo: outputs/ with the registered roots ftgrid*,
# ftmethod, gateft, gatefix and every other result directory, data/, src/, scripts/, docs/, tests/, idea-stage/, Paper/; never beside
# it: a sibling checkout, the data next to it, ../outside_repo; never above it); anything else is refused with exit 2, and so is an
# OUT_ROOT that is, or lies inside, the world it reads, and a world that holds a link leading anywhere else. A relative OUT_ROOT resolves
# from the repo root; every comparison is made on lower-case canonical forms. The trainer and the scorer are run_ftgrid.sh's stand-ins
# (the real argparse, the real scorer code with a fake model); every other step is the real code, the freeze checks included (on the
# temporary pilot log). The rehearsal first shows that an empty pilot log fails the amendment, core and FT-L record checks; the human step
# of recording the FT-L files is done on the temporary log unless DRY_NO_RECORD=1 (the record is then missing and the run is refused).
# Without symlinks (a Windows machine) DRY_RUN copies the real zero-shot scores instead of linking them. DRY_GATE and DRY_E1_FAIL act as
# in run_ftgrid.sh (DRY_E1_FAIL=s0/like: the s0 like pass fails E1). DRY_NO_WORLD=1 neither builds nor repairs that world (a test hook): a
# damaged world is refused as in a real run, not healed first. DRY_NO_RECORD and DRY_NO_WORLD are 0 or 1 too.
# Exit codes (as run_ftgrid.sh): 0 done; 1 error or an inconsistent recorded input; 2 usage or refused input; 4 refused by the freeze /
# gate rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m|toys) ;;
  *) echo "usage: bash scripts/sigir/run_ftlen.sh {ml1m|toys}" >&2; exit 2 ;;
esac
# the switches are 0 or 1 and nothing else: DRY_RUN=yes must not start a real job, nor DRY_RUN= a rehearsal
DRY_RUN="${DRY_RUN-0}"
DRY_NO_RECORD="${DRY_NO_RECORD-0}"
DRY_NO_WORLD="${DRY_NO_WORLD-0}"
for sw in DRY_RUN DRY_NO_RECORD DRY_NO_WORLD; do
  case "${!sw}" in 0|1) ;; *) echo "$sw must be 0 or 1 (it is '${!sw}'): nothing was started" >&2; exit 2 ;; esac
done
# a deliberate retry of an adapter whose like or swap pass failed E1 twice (see retry_guard): FTLEN_ALLOW_RETRY names the seeds, and
# FTLEN_RETRY_REASON, which must then be non-empty, is logged
FTLEN_ALLOW_RETRY="${FTLEN_ALLOW_RETRY-}"
FTLEN_RETRY_REASON="${FTLEN_RETRY_REASON-}"
if [ -n "$FTLEN_ALLOW_RETRY" ]; then
  case "$FTLEN_ALLOW_RETRY" in s0|s1|s0,s1|s1,s0) ;;
    *) echo "FTLEN_ALLOW_RETRY must be s0, s1 or s0,s1 (it is '$FTLEN_ALLOW_RETRY'): nothing was started" >&2; exit 2 ;; esac
  if [ -z "$FTLEN_RETRY_REASON" ]; then
    echo "FTLEN_ALLOW_RETRY=$FTLEN_ALLOW_RETRY needs a non-empty FTLEN_RETRY_REASON (it is logged): nothing was started" >&2; exit 2
  fi
fi
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftgrid        # the registered grid root: read, never written
REGL=outputs/confrec/ftgrid_len3  # the FT-L root: everything this script writes
DRYB=tmp_outputs/ftlen_dryrun     # DRY_RUN's world and FT-L root: a temporary directory outside outputs/confrec
command -v realpath > /dev/null 2>&1 || { echo "realpath (coreutils) is required" >&2; exit 1; }
FOLD=0                            # MSYS / Cygwin (Git Bash): D:/... and D:\... spellings go through cygpath
case "${OSTYPE:-}" in msys*|cygwin*) FOLD=1 ;; esac
# canon_all PATH...: the array CANON gets the canonical absolute path of every argument, in one realpath start (realpath -m: symlinks,
# //, . and .. resolved, the path need not exist; a relative path resolves from the repo root, the working directory). Every canonical
# form is lower case on every platform: a case-insensitive filesystem (macOS, a Windows share) is not always MSYS, so every comparison
# folds case
canon_all() {
  local -a a=("$@")
  if [ "$FOLD" = 1 ] && command -v cygpath > /dev/null 2>&1; then mapfile -t a < <(cygpath -u -- "$@"); fi
  mapfile -t CANON < <(realpath -m -- "${a[@]}")
  [ "${#CANON[@]}" = "$#" ] || return 1
  CANON=("${CANON[@],,}")
}
if [ "$DRY_RUN" = 1 ]; then
  MODEL="${MODEL:-dryrun/Qwen3-8B}"
  GRID=$DRYB/ftgrid               # run_ftgrid.sh's DRY_RUN root: the synthetic world the script reads
  OUT_ROOT="${OUT_ROOT:-$DRYB/ftgrid_len3}"
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  GRID="$REG"
  OUT_ROOT="${OUT_ROOT:-$REGL}"
fi
while [ "${OUT_ROOT%/}" != "$OUT_ROOT" ]; do OUT_ROOT="${OUT_ROOT%/}"; done      # ftgrid_len3// -> ftgrid_len3
if [ -z "$OUT_ROOT" ]; then echo "OUT_ROOT is empty or the filesystem root: FT-L has one registered root, $REGL" >&2; exit 2; fi
# Everything this script writes below the FT-L root, as a path below it, with the temporary names the bound tools write beside their
# products (ftgrid_report: report/D.json.tmp and D_tables.csv.tmp). The link it makes itself, scores/D/zeroshot, is not listed:
# ftlen_panel link asserts it, and sweep_links below allows no other link. The scorer's like/*.tmp names are not enumerated: the sweep
# and the removal of stale .tmp files before each scorer run cover them
TREE=(adapters "adapters/$D" "adapters/$D/s0" "adapters/$D/s1" scores "scores/$D" "scores/$D/s0" "scores/$D/s1" "scores/$D/s0/like"
      "scores/$D/s0/swap" "scores/$D/s1/like" "scores/$D/s1/swap" report "report/$D.json" "report/$D.json.tmp" "report/${D}_tables.csv"
      "report/${D}_tables.csv.tmp" report/NOTE_FTLEN.txt report/NOTE_FTLEN.txt.tmp report/NOTE_FTLEN_overrides.txt build "build/$D"
      "build/$D/done")
if [ "$DRY_RUN" = 1 ]; then SPELL=$OUT_ROOT; else SPELL=$REGL; fi          # the spelling the files are written with
ARGS=(. .. "$OUT_ROOT" "$GRID")
for r in "${TREE[@]}"; do ARGS+=("$SPELL/$r"); done
# the canonical forms decide, never the spelling: a string comparison is defeated by //, ., .., an absolute path, a link or letter case
canon_all "${ARGS[@]}" || { echo "cannot resolve OUT_ROOT=$OUT_ROOT or the paths below it" >&2; exit 2; }
ROOTC=${CANON[0]}; PARENTC=${CANON[1]}; OUTC=${CANON[2]}; GRIDC=${CANON[3]}
# under PATH ROOT: PATH is ROOT or lies below it (canonical forms)
under() { case "$1/" in "${2%/}/"*) return 0 ;; esac; return 1; }
# dry_ok PATH: a rehearsal root (canonical) is under the repo's tmp_outputs, or outside the repo's parent directory: never in the repo
# (outputs/, data/, src/, scripts/, docs/, tests/, idea-stage/, Paper/, ...), never beside it (a sibling checkout, the data next to it)
# and never above it
dry_ok() {
  if under "$1" "$ROOTC/tmp_outputs"; then return 0; fi
  if under "$1" "$PARENTC" || under "$PARENTC" "$1"; then return 1; fi
  return 0
}
if [ "$DRY_RUN" = 1 ]; then
  for p in "$OUTC" "$GRIDC"; do
    if ! dry_ok "$p"; then
      echo "DRY_RUN=1 never writes to a registered output root, nor anywhere in the repo or beside it ($p): a rehearsal root is under" \
        "$ROOTC/tmp_outputs (default $DRYB) or outside $PARENTC; a relative OUT_ROOT resolves from the repo root $ROOTC" >&2
      exit 2
    fi
  done
  if under "$OUTC" "$GRIDC"; then
    echo "DRY_RUN=1: OUT_ROOT=$OUT_ROOT resolves to $OUTC, the synthetic world it reads or a directory inside it" >&2; exit 2
  fi
elif [ "$OUTC" != "$ROOTC/$REGL" ]; then
  # canonical equality with the registered root below the repo root: no link may lie between the repo root and the FT-L root either
  echo "OUT_ROOT=$OUT_ROOT resolves to $OUTC, not to $ROOTC/$REGL: FT-L has one registered root, $REGL, reached through no link (the" \
    "grid root $REG is read, never written); use DRY_RUN=1 for a rehearsal" >&2
  exit 2
else
  OUT_ROOT=$REGL
fi
# nothing written below the FT-L root may leave it through a link: every path of the tree resolves, links included, to the same place
# below the canonical root
for i in "${!TREE[@]}"; do
  expect="$OUTC/${TREE[$i]}"
  expect="${expect,,}"
  if [ "${CANON[$((i + 4))]}" != "$expect" ]; then
    echo "FT-L refused: $SPELL/${TREE[$i]} resolves to ${CANON[$((i + 4))]}, not to $expect: a link between the FT-L root and the" \
      "files this script writes would redirect them (nothing was written)" >&2
    exit 2
  fi
done
# DRY_RUN lets run_ftgrid.sh's own DRY_RUN write the synthetic world: a link inside it must stay inside the rehearsal's own places
if [ "$DRY_RUN" = 1 ] && [ -e "$GRID" ]; then
  mapfile -t WLINKS < <(find "$GRID" -type l 2> /dev/null)
  if [ "${#WLINKS[@]}" -gt 0 ]; then
    canon_all "${WLINKS[@]}" || { echo "cannot resolve the links inside the synthetic world" >&2; exit 2; }
    for p in "${CANON[@]}"; do
      if ! dry_ok "$p"; then
        echo "DRY_RUN=1 never writes to a registered output root: a link inside the synthetic world leads to $p, in the repo or beside" \
          "it, outside $ROOTC/tmp_outputs" >&2
        exit 2
      fi
    done
  fi
fi
# sweep_links [full]: the only link below the FT-L root is the scores/<dataset>/zeroshot that ftlen_panel link makes (and asserts). Any
# other link, in particular a planted temporary name of a bound tool (report/D.json.tmp, like/*.tmp), is refused with exit 2 before a
# tool can write through it. "full" (the start of a run) also resolves each allowed link and compares it with the registered zero-shot
# score directory of its dataset
sweep_links() {
  [ -d "$OUT_ROOT" ] || return 0
  local -a found=() chk=() exp=()
  local l rel n i
  mapfile -t found < <(find "$OUT_ROOT" -type l 2> /dev/null)
  for l in "${found[@]}"; do
    rel=${l#"$OUT_ROOT"/}
    if ! [[ $rel =~ ^scores/(ml1m|toys)/zeroshot$ ]]; then
      echo "FT-L refused: $l is a link: the only link below $OUT_ROOT is scores/<dataset>/zeroshot, and a tool would write through" \
        "it (nothing was written)" >&2
      exit 2
    fi
    chk+=("$l"); exp+=("$GRID/scores/${BASH_REMATCH[1]}/zeroshot")
  done
  if [ "${1:-}" = full ] && [ "${#chk[@]}" -gt 0 ]; then
    n=${#chk[@]}
    canon_all "${chk[@]}" "${exp[@]}" || { echo "cannot resolve the links below $OUT_ROOT" >&2; exit 2; }
    for ((i = 0; i < n; i++)); do
      if [ "${CANON[$i]}" != "${CANON[$((i + n))]}" ]; then
        echo "FT-L refused: ${chk[$i]} resolves to ${CANON[$i]}, not to the registered zero-shot score directory ${CANON[$((i + n))]}" >&2
        exit 2
      fi
    done
  fi
}
sweep_links full
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL is not Qwen3-8B: FT-L is registered for the Gate-FT backbone only (addendum 9 section 1)" >&2; exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|3|all) ;; *) echo "unknown stage '$s' in STAGES (1-3, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 3 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$GRID/panels/$D"                 # the grid's panels of D (read)
SPLIT="$P/ftgrid_split.json"
RS="$GRID/scores/$D"                # the real scores: zeroshot (linked into the FT-L root) and s0-s2's like and swap passes (their records)
QA="$OUT_ROOT/adapters/$D"          # the 3-epoch adapters s0, s1
QS="$OUT_ROOT/scores/$D"            # scores/D/zeroshot: a link to the registered zero-shot scores; scores/D/s<k>/{like,swap}
REP="$OUT_ROOT/report/$D.json"
REGREP="$GRID/report/$D.json"       # the registered (one-epoch) report of D: never touched
B="$OUT_ROOT/build/$D"              # step markers
DONE="$B/done"
LEN_FILES=(scripts/sigir/run_ftlen.sh src/confrec/ftlen_panel.py tests/test_confrec_ftlen.py)   # addendum 9 section 3: recorded before the first training
SWAP_K=8                            # addendum 9 section 1: the swap-prior arm of the registered decomposition stage
if [ "$DRY_RUN" = 1 ]; then
  DRYD="$GRID/_dry"
  G="$DRYD/gatefix"; GT="$DRYD/gateft"; RAW="$DRYD/raw"; PILOT_LOG="$DRYD/PILOT_LOG.md"
  N_BOOT=20                         # the report's cost is its bootstrap: its structure does not depend on the count (the real one is 2,000)
  CORE_SPLITS=(--split "$SPLIT")
  LINK_FLAG="--allow_copy"
else
  G=outputs/confrec/gatefix; GT=outputs/confrec/gateft; RAW=data/raw; PILOT_LOG=docs/sigir/PILOT_LOG.md
  N_BOOT=2000                       # A3 section 3: 2,000 user resamples
  CORE_SPLITS=(--split "$REG/panels/ml1m/ftgrid_split.json" --split "$REG/panels/toys/ftgrid_split.json"
    --split "$REG/panels/games/ftgrid_split.json" --split "$REG/panels/sports/ftgrid_split.json")
  LINK_FLAG=""
fi
if [ "$D" = ml1m ]; then            # the real adapters of ML-1M are Gate-FT's (never retrained, scored by their path)
  RA="$GT/adapters"; TRAIN_REF="$GT/train.jsonl"; SPLIT_RECIPE_FLAG=""
else
  RA="$GRID/adapters/$D"; TRAIN_REF="$P/train.jsonl"; SPLIT_RECIPE_FLAG="--split_recipe"
fi

# ---- helpers ----
# fresh OUT DEP...: OUT exists and is newer than every DEP (a missing DEP makes it stale)
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }
# step PRODUCT DEP... -- CMD...: run CMD unless PRODUCT exists and its marker DONE/<basename>.done is newer than every DEP;
# the marker is touched only after CMD succeeds, so an interrupted step always reruns
step() {
  local prod=$1 deps=() mark; shift
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  mark="$DONE/$(basename "$prod").done"
  if [ -e "$prod" ] && fresh "$mark" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$mark"
  "$@" || return 1
  mkdir -p "$DONE"; touch "$mark"
}
adapter_done() { [ -f "$1/train_config.json" ] && ls "$1"/adapter_model.* >/dev/null 2>&1; }
# e1_ok DIR: section 2 / E1 of a finished scorer run: censored-2 share <= 0.5%, no overlength prompt, mean Yes+No mass
# >= 0.95 on the main prompts; a swap run's donor prompts also need the censored-2 share and no overlength prompt
# (run_ftgrid.sh's function, verbatim: tests/test_confrec_ftlen.py compares the two)
e1_ok() {
  "$PY" - "$1/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
n = int(r["n_main_prompts"])
ok = n > 0 and int(r["censored_main"].get("2", 0)) <= 0.005 * n and int(r["n_overlength"]) == 0 \
    and float(r["mean_yes_no_mass"]) >= 0.95
if "censored_swap" in r:
    ns = int(r["swap_prompts"])
    ok = ok and int(r["censored_swap"].get("2", 0)) <= 0.005 * ns and int(r["censored_swap"].get("3", 0)) == 0 \
        and int(r.get("swap_n_overlength") or 0) == 0
sys.exit(0 if ok else 1)
PY
}
# MPY runs the entry points that load the model (the trainer, the scorer). Under DRY_RUN it is fake_py: run_ftgrid.sh's
# stand-ins; any other call runs as is
fake_py() {
  if [ "$1" = -m ]; then
    case "$2" in
      src.confrec.pyes_scorer) shift 2; "$PY" "$FAKES" scorer "$@"; return ;;
      src.confrec.train_lora_yesno) shift 2; "$PY" "$FAKES" trainer "$@"; return ;;
    esac
  fi
  "$PY" "$@"
}
MPY="$PY"
if [ "$DRY_RUN" = 1 ]; then MPY=fake_py; fi
# e1_failed_before DIR KEY: a run with this run.key already failed E1 and was moved aside (DIR.e1fail.*), so its one rerun is used up.
# A DIR.e1fail.* of another key belongs to an earlier panel, adapter or argument vector and does not count. (run_ftgrid.sh counts
# every DIR.e1fail.*: a leftover of an earlier run then turns a transient first failure into the final one. A deliberate deviation.)
e1_failed_before() {
  local d
  for d in "$1".e1fail.*; do
    if [ -f "$d/run.key" ] && [ "$(cat "$d/run.key")" = "$2" ]; then return 0; fi
  done
  return 1
}
# retry_guard SEED: FAILED_INTEGRITY is sticky per seed. When a pass of s<SEED> failed E1 twice (scores/D/s<SEED>/like or /swap, or a
# .stale.* of either, holds FAILED_INTEGRITY), re-making the adapter or scoring it under another key would start a fresh E1 budget, and
# section 2 says it is reported as missing, never replaced: stages 1 and 2 refuse (exit 1, naming the directory) unless
# FTLEN_ALLOW_RETRY names the seed and FTLEN_RETRY_REASON says why; the override is appended to report/NOTE_FTLEN_overrides.txt
# (once per run)
retry_guard() {
  local seed=$1 d hit="" arm
  for arm in like swap; do
    for d in "$QS/s$seed/$arm" "$QS/s$seed/$arm".stale.*; do
      if [ -f "$d/FAILED_INTEGRITY" ]; then hit=$d; break 2; fi
    done
  done
  if [ -z "$hit" ]; then return 0; fi
  case ",$FTLEN_ALLOW_RETRY," in
    *",s$seed,"*) ;;
    *) echo "FT-L refused: a pass of s$seed failed E1 twice ($hit/FAILED_INTEGRITY): making s$seed again or scoring it under" \
         "another key would reset its E1 budget, and section 2 reports it as missing, never replaced. A deliberate retry needs" \
         "FTLEN_ALLOW_RETRY=s$seed and FTLEN_RETRY_REASON='why' (logged in $OUT_ROOT/report/NOTE_FTLEN_overrides.txt)" >&2
       exit 1 ;;
  esac
  case "${RETRY_LOGGED:-}" in *",s$seed,"*) return 0 ;; esac
  mkdir -p "$OUT_ROOT/report"
  printf 's%s\t%s\t%s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FTLEN_RETRY_REASON" >> "$OUT_ROOT/report/NOTE_FTLEN_overrides.txt"
  RETRY_LOGGED="${RETRY_LOGGED:-},s$seed,"
  echo "[retry] s$seed ($hit/FAILED_INTEGRITY): FTLEN_ALLOW_RETRY, logged in $OUT_ROOT/report/NOTE_FTLEN_overrides.txt" >&2
}
# score DATA DIR --lora A [--swap_k K]: pyes_scorer with SARGS, the scorer flags of the real s0 passes (fp16, top-50 logprobs,
# max_model_len 4096, 100-user chunks, the selected variant, yes/no readout, like: run_ftgrid.sh's score(), as its record says);
# completion marker DIR/report.json, run.key = panel sha1 + model + variant + adapter weights sha1 + args (the format of run_ftgrid.sh)
score() {
  local data=$1 dir=$2 key old="" lora="" prev="" x wsha=-; shift 2
  for x in "$@"; do if [ "$prev" = --lora ]; then lora="$x"; fi; prev="$x"; done
  if [ -n "$lora" ]; then wsha=$(cat "$lora"/adapter_model.* | sha1sum | cut -d' ' -f1); fi
  key="$(sha1sum "$data" | cut -d' ' -f1) $MODEL $VARIANT $wsha $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -n "${RETRY_SEED:-}" ]; then retry_guard "$RETRY_SEED"; fi      # a failed integrity is not escaped by a new key
  if [ -e "$dir" ] && [ "$old" != "$key" ]; then
    mv "$dir" "$dir.stale.$(date +%Y%m%d%H%M%S)"; echo "[moved aside] $dir (panel, model, adapter or args changed)"
  fi
  mkdir -p "$dir"
  find "$dir" -name '*.tmp' -delete 2> /dev/null || true                 # a stale temporary file is not written through
  echo "$key" > "$dir/run.key"
  "$MPY" -m src.confrec.pyes_scorer --data "$data" --output "$dir" --model "$MODEL" "${SARGS[@]}" "$@"
  if ! e1_ok "$dir"; then
    if e1_failed_before "$dir" "$key"; then
      touch "$dir/FAILED_INTEGRITY"
      echo "FAILED_INTEGRITY: $dir failed E1 twice (section 2: reported as missing, never replaced)" >&2
    else
      mv "$dir" "$dir.e1fail.$(date +%Y%m%d%H%M%S)"
      echo "[E1 failed] $dir: moved aside, rerun once (section 2)" >&2
      score "$data" "$dir" "$@"
    fi
  fi
}
# the checks of the gates; the optional argument is another pilot log (the DRY_RUN rehearsal of an empty one)
check_amendment() { "$PY" -m src.confrec.ftgrid_freeze --check --stage amendment --pilot_log "${1:-$PILOT_LOG}"; }
check_core() { "$PY" -m src.confrec.ftgrid_freeze --check --stage core --pilot_log "${1:-$PILOT_LOG}" "${CORE_SPLITS[@]}"; }
check_record() { "$PY" -m src.confrec.ftlen_panel record --pilot_log "${1:-$PILOT_LOG}" --files "${LEN_FILES[@]}"; }
# ensure_record: the FT-L record is in the pilot log (exit 4 otherwise). DRY_RUN does the human step on the temporary log first
# (--append writes the missing lines), unless DRY_NO_RECORD=1
ensure_record() {
  if [ "$DRY_RUN" = 1 ] && [ "$DRY_NO_RECORD" != 1 ]; then
    "$PY" -m src.confrec.ftlen_panel record --pilot_log "$PILOT_LOG" --files "${LEN_FILES[@]}" --append || exit 4
  else
    check_record || exit 4
  fi
}
# rehearse: DRY_RUN only, once per world: an empty pilot log fails the amendment, core and FT-L record checks
rehearse() {
  if [ -f "$B/rehearsed" ]; then return 0; fi
  [ -f "$DRYD/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRYD/PILOT_LOG.empty.md"
  if check_amendment "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 || check_core "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 \
      || check_record "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1; then
    echo "DRY_RUN: a freeze or record check passed on an empty pilot log" >&2; exit 1
  fi
  echo "[dry] gate rehearsal: an empty pilot log fails the amendment, core and FT-L record checks"
  mkdir -p "$B"; touch "$B/rehearsed"
}
# gates: everything the registered text asks before a GPU job (Amendment 3 section 0, addendum 9 section 3)
gates() {
  if [ "$GATE_DECISION" != GATE_FT_PASS ]; then
    echo "FT-L refused: Gate-FT decision $GATE_DECISION (section 0: FT-L runs only after a recorded GATE_FT_PASS, $GT/gate_ft.json)" >&2
    exit 4
  fi
  if [ "$DRY_RUN" = 1 ]; then rehearse; fi
  check_amendment || { echo "FT-L refused: the freeze record of Amendment 3 (stage amendment) is not in $PILOT_LOG" >&2; exit 4; }
  check_core || { echo "FT-L refused: the full record of Amendment 3 (stage core) is not in $PILOT_LOG" >&2; exit 4; }
  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then
    echo "FT-L refused: selection.json found a fix but the gate-fix confirm stage has recorded no gate.json (section 4)" >&2
    exit 4
  fi
  ensure_record
}
# read_info: SEL_DECISION, SEL_PROMPT (selection.json), SPLIT_VARIANT (the split) and GATE_DECISION (gate_ft.json; missing if absent)
read_info() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftlen_panel info --selection "$SEL" --split "$SPLIT" --gate "$GT/gate_ft.json") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  { read -r SEL_DECISION; read -r SEL_PROMPT; read -r SPLIT_VARIANT; read -r GATE_DECISION; } < <(printf '%s\n' "$out" | tr -d '\r')
}
# recipe: RECIPE = the trainer flags the real adapters s0-s2 recorded in their train_config.json with --train kept and `--epochs 3`
# for the recorded epochs, except --out and --seed (ftlen_panel validates them: one registered recipe, this model, variant and TRAIN
# file; ML-1M: Gate-FT's adapters)
recipe() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftlen_panel recipe --adapters "$RA" --model "$MODEL" --variant "$VARIANT" --train_ref "$TRAIN_REF" \
    --train_file "$P/train.jsonl" --split "$SPLIT" $SPLIT_RECIPE_FLAG) || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t RECIPE < <(printf '%s\n' "$out" | tr -d '\r')
  echo "[recipe] trainer flags recorded by the real s0-s2 (--out and --seed replaced, --epochs 3, --train kept): ${RECIPE[*]}"
}
# scoring_recipe: SARGS = the scorer flags of the real s0's like pass (its run.key and the config of its report.json); the swap pass
# of the real s0 must be recorded with the same flags (the arm adds only --swap_k 8 and its panel)
scoring_recipe() {
  local like swap rc=0
  if [ -n "${SARGS_READ:-}" ]; then return 0; fi
  like=$("$PY" -m src.confrec.ftlen_panel scoring --arm like --scores "$RS" --adapters "$RA" --data "$P/eval.jsonl" --model "$MODEL" \
    --variant "$VARIANT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  swap=$("$PY" -m src.confrec.ftlen_panel scoring --arm swap --scores "$RS" --adapters "$RA" --data "$P/eval_sd_test.jsonl" \
    --model "$MODEL" --variant "$VARIANT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  if [ "$(printf '%s' "$like" | tr -d '\r')" != "$(printf '%s' "$swap" | tr -d '\r')" ]; then
    echo "the real s0's like and swap passes were not scored with the same arguments (like: $(echo $like), swap: $(echo $swap))" >&2
    exit 1
  fi
  mapfile -t SARGS < <(printf '%s\n' "$like" | tr -d '\r')
  SARGS_READ=1
  echo "[recipe] scorer flags recorded by the real s0 like and swap passes (--data, --output, --model, --lora replaced; swap adds" \
    "--swap_k $SWAP_K): ${SARGS[*]}"
}
# verify_adapters SEEDS [--write]: each s<seed>'s recorded arguments equal the real s0's except --out, --seed and --epochs (3); its provenance
verify_adapters() {
  local seeds=$1; shift
  "$PY" -m src.confrec.ftlen_panel verify_adapter --adapters "$QA" --ref_adapters "$RA" --seeds "$seeds" --train_file "$P/train.jsonl" \
    "$@" || exit $?
}
# train_adapter OUT SEED: the recorded recipe with 3 epochs on the registered TRAIN panel; skipped when train_config.json and weights
# exist (their provenance was verified before the first training)
train_adapter() {
  local out=$1 seed=$2
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  retry_guard "$seed"                  # before any training: a failed integrity is not escaped by making the adapter again
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_yesno --out "$out" --seed "$seed" "${RECIPE[@]}" 2>&1 | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
  verify_adapters "$seed" --write
}
# verify_arm ARM SEED: the recorded scoring config of s<SEED>'s pass of the arm equals the real s0's pass of the arm except `lora`
verify_arm() {
  "$PY" -m src.confrec.ftlen_panel verify_scores --arm "$1" --scores "$QS" --ref_scores "$RS" --adapters "$QA" --seeds "$2" || exit $?
}
# link_scores: the registered zero-shot scores become a symlink of the FT-L root (never a copy; DRY_RUN without symlinks copies)
link_scores() {
  "$PY" -m src.confrec.ftlen_panel link --real_scores "$RS" --q_scores "$QS" --models zeroshot $LINK_FLAG || exit $?
}
# note_ftlen: report/NOTE_FTLEN.txt, the sidecar that says what the reports of this root are (written when missing or different)
note_ftlen() {
  local f="$OUT_ROOT/report/NOTE_FTLEN.txt" tmp="$OUT_ROOT/report/NOTE_FTLEN.txt.tmp"
  mkdir -p "$OUT_ROOT/report"
  printf '%s\n' \
    "FT-L, the longer-training robustness arm (Amendment 3 addendum 9; EXPLORATORY: no hypothesis, no Holm family, no claim-admission" \
    "role), written by scripts/sigir/run_ftlen.sh." \
    "" \
    "1. The adapters s0 and s1 and their scores in this root are 3-EPOCH adapters: the registered recipe of the dataset's adapters" \
    "   (read from their recorded train_config.json) with --epochs 3 and nothing else changed, on the registered real-label TRAIN panel," \
    "   seeds 0 and 1. They are not the registered adapters. zeroshot here is a link to the registered zero-shot scores. The registered" \
    "   report of the same dataset, outputs/confrec/ftgrid/report/<D>.json, is the ONE-EPOCH result, and the only registered one." \
    "2. report/<D>.json and report/<D>_tables.csv are the unchanged ftgrid_report output on zeroshot, s0 and s1. It reads s0 and s1 as" \
    "   the FT models: every number that needs the three registered seeds (the P1 decision, the E-B family, the sigma_seed rule) is" \
    "   INCOMPLETE or descriptive here, and no decision or Holm result of this file is registered. Every value is two seeds, exploratory." \
    "3. ftgrid_extra is not run by this script (its runner refuses this root); see the header of scripts/sigir/run_ftlen.sh for the" \
    "   command that builds its blocks on this root and for the root label that keeps the file out of every A3-6 family." > "$tmp"
  if [ -f "$f" ] && cmp -s "$tmp" "$f"; then rm -f "$tmp"; else mv "$tmp" "$f"; fi
}
# registered_sha: the sha1 of the registered report of D and its tables ('-' when absent)
registered_sha() {
  local f
  for f in "$REGREP" "${REGREP%.json}_tables.csv"; do
    if [ -f "$f" ]; then sha1sum "$f" | cut -d' ' -f1; else echo -; fi
  done
}

# ---- DRY_RUN: run_ftgrid.sh's synthetic world of D (its stages 0-4) and the stand-ins ----
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$DRYD"
  FAKES="$DRYD/ftgrid_fakes.py"
  if [ "$DRY_NO_WORLD" = 1 ]; then
    echo "[skip] DRY_NO_WORLD=1: the synthetic world of $D is used as it is ($GRID)"
  elif [ -f "$SPLIT" ] && [ -f "$FAKES" ] && [ -f "$RS/zeroshot/like/report.json" ] && [ -f "$RS/zeroshot/swap/report.json" ] \
      && [ -f "$RS/s0/like/report.json" ] && [ -f "$RS/s0/swap/report.json" ] && [ -f "$RS/s1/like/report.json" ] \
      && [ -f "$RS/s2/like/report.json" ] && [ -f "$P/eval_sd_test.jsonl" ]; then
    echo "[skip] run_ftgrid.sh's DRY_RUN world for $D exists ($GRID)"
  else
    echo "== DRY_RUN: run_ftgrid.sh's synthetic world for $D (its stages 0-4: panels, the real adapters, their like and decomposition passes)"
    if ! DRY_RUN=1 OUT_ROOT="$GRID" MODEL="$MODEL" STAGES=0,1,2,3,4 PYTHON="$PY" bash scripts/sigir/run_ftgrid.sh "$D" \
        > "$DRYD/ftgrid.$D.log" 2>&1; then
      tail -n 40 "$DRYD/ftgrid.$D.log" >&2
      echo "DRY_RUN: run_ftgrid.sh $D failed (log: $DRYD/ftgrid.$D.log)" >&2; exit 1
    fi
  fi
fi

# ---- the prompt and the recorded decisions ----
SEL="$G/dev/selection.json"
[ -f "$SEL" ] || { echo "missing $SEL (the gate-fix stage 1 selection; run_gatefix.sh writes it)" >&2; exit 1; }
read_info
VARIANT="${VARIANT:-$SEL_PROMPT}"
if [ "$VARIANT" != "$SEL_PROMPT" ] && [ "$DRY_RUN" != 1 ]; then
  echo "VARIANT=$VARIANT is not selection.json's gate_ft_prompt $SEL_PROMPT: section 2 registers no other variant" >&2
  exit 2
fi
if [ -n "$SPLIT_VARIANT" ] && [ "$SPLIT_VARIANT" != "$VARIANT" ]; then
  echo "$SPLIT was built under variant $SPLIT_VARIANT, not $VARIANT" >&2; exit 2
fi
echo "run_ftlen $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ---- the registered gates come first: a refused run touches nothing ----
GATED=0
for s in 1 2 3; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then gates; fi

# ================= stage 1: the 3-epoch adapters s0, s1 (GPU) =================
if want 1; then
  sweep_links
  echo "== stage 1: adapters s0, s1, 3 epochs ($D)"
  for f in "$P/train.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  recipe
  scoring_recipe       # pre-flight: the real s0's like and swap passes must be the registered ones before any GPU hour is spent
  have=""
  for seed in 0 1; do if adapter_done "$QA/s$seed"; then have="${have:+$have,}$seed"; fi; done
  if [ -n "$have" ]; then verify_adapters "$have"; fi
  for seed in 0 1; do train_adapter "$QA/s$seed" "$seed"; done
fi

# ================= stage 2: like and swap for s0, s1 (GPU) =================
if want 2; then
  sweep_links
  echo "== stage 2: like on eval.jsonl and swap (--swap_k $SWAP_K) on eval_sd_test.jsonl for s0, s1 ($D)"
  for f in "$P/eval.jsonl" "$P/eval_sd_test.jsonl"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  scoring_recipe
  for seed in 0 1; do
    adapter_done "$QA/s$seed" || { echo "adapter s$seed of $D is missing or incomplete: $QA/s$seed (stage 1)" >&2; exit 1; }
  done
  verify_adapters 0,1
  for seed in 0 1; do
    RETRY_SEED=$seed score "$P/eval.jsonl" "$QS/s$seed/like" --lora "$QA/s$seed"
    verify_arm like "$seed"
    RETRY_SEED=$seed score "$P/eval_sd_test.jsonl" "$QS/s$seed/swap" --lora "$QA/s$seed" --swap_k "$SWAP_K"
    verify_arm swap "$seed"
  done
  link_scores
fi

# ================= stage 3: ftgrid_report with zeroshot, s0, s1 (CPU) =================
if want 3; then
  sweep_links
  echo "== stage 3: ftgrid_report with zeroshot, s0, s1 -> $REP ($D)"
  for arm in like swap; do
    [ -f "$RS/zeroshot/$arm/report.json" ] || { echo "missing the registered zero-shot $arm pass $RS/zeroshot/$arm (run_ftgrid.sh $D)" >&2; exit 1; }
  done
  for m in s0 s1; do
    for arm in like swap; do
      { [ -f "$QS/$m/$arm/report.json" ] || [ -f "$QS/$m/$arm/FAILED_INTEGRITY" ]; } \
        || { echo "missing the $arm pass $QS/$m/$arm (stage 2)" >&2; exit 1; }
    done
  done
  link_scores
  note_ftlen
  RDEPS=("$SPLIT" src/confrec/ftgrid_report.py)
  for f in "$QS"/*/*/report.json; do if [ -f "$f" ]; then RDEPS+=("$f"); fi; done
  REG_BEFORE=$(registered_sha)
  mkdir -p "$(dirname "$REP")"
  rm -f "$REP.tmp" "${REP%.json}_tables.csv.tmp"        # a stale temporary file of the report is not written through
  step "$REP" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftgrid_report --domain "$D" --split "$SPLIT" --panels "$P" --scores_root "$OUT_ROOT/scores" \
      --models zeroshot,s0,s1 --raw "$RAW" --out "$REP" --n_boot "$N_BOOT" --seed 0
  if [ "$(registered_sha)" != "$REG_BEFORE" ]; then
    echo "the registered report $REGREP changed: it is never touched (addendum 9)" >&2; exit 1
  fi
fi
for m in s0 s1; do
  for arm in like swap; do
    if [ -f "$QS/$m/$arm/FAILED_INTEGRITY" ]; then
      echo "NOTE: $QS/$m/$arm is FAILED_INTEGRITY: that pass of $m is reported as missing, never replaced (section 2)" >&2
    fi
  done
done
echo "run_ftlen $D: done (stages:$STAGES)"
