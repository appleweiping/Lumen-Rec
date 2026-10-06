#!/usr/bin/env bash
# FT-B, the item-balanced fine-tuning arm, FT-S, its size-matched random control, and FT-N, its falsification arm
# (idea-stage/PREREG_AMENDMENT_3_ADDENDUM_13.md; EXPLORATORY: no hypothesis, no Holm family, no claim-admission role of its own) on ML-1M,
# the only dataset, Qwen3-8B, the only backbone: eight adapters trained like the registered adapters s0-s2 (the section-2 recipe, read from
# their recorded train_config.json: LoRA r 16, lr 1e-4, one epoch over the panel, effective batch 32, ...) on three kinds of panel built
# from the registered TRAIN panel by src/confrec/ftb_panel.py (FT-B: every item with both classes keeps min(n+, n-) examples of each class;
# FT-S: K examples drawn at random, K = FT-B's; FT-N: the FT-B panel with its labels permuted within the kept items), scored with the `like`
# question on the whole EVAL panel and with the swap-prior arm (--swap_k 8) on S_d's TEST rows exactly like the real s0's passes, and read by
# ftgrid_report (unchanged) per arm and by src/confrec/ftb_reading.py across arms. A NEW script (no bound file changes; nothing of
# run_ftlen.sh, ftlen_panel.py, run_ftq.sh or ftq_panel.py is edited: what is shared is imported read-only). GPU server; one job at a time
# through scripts/sigir/gpu_queue.sh, one job per dataset:
#     cd /root/autodl-tmp/lumen-rec && bash scripts/sigir/run_ftb.sh ml1m
#   usage: bash scripts/sigir/run_ftb.sh D       D = ml1m (the only argument; any other dataset is refused)
#   env:   MODEL      Qwen3-8B dir (default /root/autodl-tmp/lumen/models/Qwen3-8B); any other backbone is refused
#          OUT_ROOT   the FT-B root outputs/confrec/ftgrid_ftb, the only root of this script outside DRY_RUN and READ_ROOT. A relative OUT_ROOT
#                     resolves from the repo root (the script's working directory). It is compared canonically, in lower case on every
#                     platform (realpath -m: //, ., .., absolute paths, symlinks; on Windows also D:/... spellings): any spelling of that root
#                     is accepted, any other directory (the grid root outputs/confrec/ftgrid, ftgrid_ftb/sub, ...) is refused, and so is
#                     that root reached through a link (outputs, outputs/confrec or ftgrid_ftb itself a symlink)
#          READ_ROOT  the reading root outputs/confrec/ftb_reading (ml1m.json, ml1m.csv and the CF reference cache ml1m_refs.json), checked
#                     in the same way
#          VARIANT    default gate_ft_prompt of outputs/confrec/gatefix/dev/selection.json; any other value is refused
#          STAGES     comma list of 1-4 or all (default all)
#          PYTHON     interpreter (skips the conda activation)
#          DRY_RUN    0 or 1 (any other value is refused with exit 2); 1 = a CPU rehearsal, see below
#          FTB_ALLOW_RETRY, FTB_RETRY_REASON   a comma list of adapters (b0 b1 b2 r0 r1 r2 n0 n1) and a non-empty reason: a deliberate retry
#                     of an adapter whose like or swap pass failed E1 twice (FAILED_INTEGRITY is sticky per adapter, see stages 2 and 3); the
#                     retry is logged
#   layout (the reason for it): ONE root for the three arms, so that the guards below are the reviewed guards of run_ftlen.sh (one canonical
#          root, one link sweep, one DRY allow-list) and not three copies with a cross-root dependency (FT-S's K and FT-N's panel come from
#          FT-B's). The adapters keep the names of the addendum: adapters/ml1m/{b0,b1,b2} (FT-B), {r0,r1,r2} (FT-S), {n0,n1} (FT-N). The
#          unchanged ftgrid_report and ftgrid_extra accept the registered model names only (zeroshot, s0, s1, s2, p0, p1), so each arm has its
#          own scores tree scores/<b|r|n>/ml1m in which its adapters carry those names (b0 is scores/b/ml1m/s0, r1 is scores/r/ml1m/s1, n0 is
#          scores/n/ml1m/s0; report.json's lora records the adapter, and the reading refuses a tree scored with another arm's adapter) and
#          scores/<tag>/ml1m/zeroshot is a link to the registered zero-shot scores. Per arm: report/<tag>/ml1m.json, and by hand
#          extra/<tag>/ml1m.json; the reading across the arms is the only place where they are compared.
#   reads (never written): the grid root outputs/confrec/ftgrid (panels/ml1m: train.jsonl, eval.jsonl, eval_sd_test.jsonl, ftgrid_split.json;
#          scores/ml1m of the registered zero-shot and s0-s2: their like and swap passes, the recorded arguments of the real s0's passes and
#          the reference arms of the reading), Gate-FT's adapters outputs/confrec/gateft/adapters (the real adapters of ML-1M: their
#          train_config.json) and train.jsonl, and the raw ratings data/raw
#   writes (the FT-B root only, outputs/confrec/ftgrid_ftb): ftb/ml1m/{train_b, train_r0, train_r1, train_r2, train_n}.jsonl and their
#          .manifest.json; adapters/ml1m/{b0..b2, r0..r2, n0, n1}; scores/{b,r,n}/ml1m/zeroshot = a link to the registered zero-shot scores
#          (linked, never copied or rescored), scores/<tag>/ml1m/s<k>/{like,swap}; report/<tag>/ml1m.json (+ ml1m_tables.csv),
#          report/NOTE_FTB.txt and, when a retry is allowed, report/NOTE_FTB_overrides.txt; build/ml1m/ (step markers); and below READ_ROOT
#          (outputs/confrec/ftb_reading): ml1m.json, ml1m.csv, ml1m_refs.json. Nothing below outputs/confrec/ftgrid is written, and nothing is
#          written through a link. What is covered: (1) every path this script names below the two roots, with the temporary names the bound
#          tools write beside their products, must resolve, links included, to the same place below the canonical root, else the run is
#          refused (exit 2) before it writes anything; (2) a sweep for links (find ROOT -type l) at the start and before each stage allows
#          only the scores/<tag>/ml1m/zeroshot that this script makes itself (asserted when made, and resolved to the registered zero-shot
#          score directory at the start) and no link below READ_ROOT; (3) the temporary files of the report, of the panels and of a scoring
#          directory are removed before ftgrid_report, the builder or the scorer runs. The scorer's own temporary names (like/*.tmp,
#          like/parts/*.tmp) are not enumerated: only (2) and (3) cover them, and a bound tool that wrote another temporary name would be
#          covered by (2) alone
#   gates    (before stages 2-4, in this order; any failure exits 4 and starts nothing): the recorded Gate-FT decision is GATE_FT_PASS
#            (Amendment 3 section 0); `ftgrid_freeze --check --stage amendment` and `--stage core` (the full record, a CPU check run again at
#            every start; the addendum itself is part of the amendment record); the gate-fix decisions (selection.json, and gate.json when a
#            fix was found); the stage-1 manifests exist (exit 1 otherwise); and the FT-B record (addendum 13 section 5): the sha1 of this
#            script, of src/confrec/ftb_panel.py, of src/confrec/ftb_reading.py, of tests/test_confrec_ftb.py and of
#            tests/test_confrec_ftb_run.py, and of the five panel manifests (train_b, train_r0-r2, train_n), are in docs/sigir/PILOT_LOG.md.
#            Imported or used as they are, and recorded separately (not part of that record): src/confrec/ftq_panel.py and
#            src/confrec/ftlen_panel.py (the recipe validation, the run.key and report-config readers, the like and swap record checks, the
#            link plumbing), src/confrec/ftprune.py (tie_key, train_examples), ftgrid_data.py, split_panel.py, build_rated_panels.py,
#            stats.py, prompting.py, train_lora_yesno.py, pyes_scorer.py, ftgrid_report.py and src/confrec/ftgrid_extra.py
#   stage 1  (CPU, outcome-free, no gate) ftb_panel build: the five panels and their manifests (K, the kept items, the share of TRAIN
#            examples dropped, the label rate overall and per item, sha1 of the source and of every panel, the seeds, the code sha1), then
#            ftb_panel verify (also when the step was skipped: all ten files are recomputed from the registered train.jsonl and must equal the
#            directory byte for byte, exit 1 otherwise; neither a step marker nor a manifest is trusted) and the sha1 of every file is printed
#            as a pilot-log line (`FILE = sha1`). Record them with the code before stage 2 (the human step)
#   stage 2  (GPU) verify the panels again, then ftb_panel recipe: the trainer flags the real adapters s0-s2 recorded in their
#            train_config.json (Gate-FT's; s0-s2 must agree and be the registered recipe, one epoch; their TRAIN file must hold the bytes of the
#            registered train.jsonl), with --train, --out and --seed the only flags replaced. The records of the real s0's like and swap
#            passes are validated too before the first training (ftb_panel scoring), so a wrong record stops the job early. Then train b0-b2
#            on train_b.jsonl, r0-r2 on train_r<seed>.jsonl and n0, n1 on train_n.jsonl (seeds 0-2, 0-2, 0-1) with train_lora_yesno; the
#            trained config is checked against the real s0's (only --train, --out and --seed may differ; exactly the K examples of the panel)
#            and the adapter's provenance (ftb.json: arm, seed, the sha1 of the panel and of its manifest, K, the number of optimizer steps and
#            the registered SFT's) is written. An adapter with train_config.json and weights is never retrained, but must carry that
#            provenance of this panel. FAILED_INTEGRITY is sticky per adapter: while scores/<tag>/ml1m/s<k>/like or /swap, or a .stale.* of
#            either, holds it (that pass failed E1 twice), making the adapter again is refused (exit 1, naming the directory) unless
#            FTB_ALLOW_RETRY names it and a non-empty FTB_RETRY_REASON is set; the override is appended to report/NOTE_FTB_overrides.txt
#            (adapter, UTC date, reason). A finished failed state is resumed without refusal
#   stage 3  (GPU) per adapter: like on panels/ml1m/eval.jsonl (CAL and TEST rows), then the swap-prior arm (--swap_k 8) on
#            panels/ml1m/eval_sd_test.jsonl (S_d's TEST rows): pyes_scorer with exactly the arguments of the real s0's pass of the arm (its
#            run.key and the config of its report.json; ftb_panel scoring); no other arm is scored. E1 as run_ftgrid.sh (a failing run is moved
#            to DIR.e1fail.<time> and rerun once, a second failure of the same run.key leaves DIR/FAILED_INTEGRITY: the adapter is reported as
#            missing, never replaced; one deliberate difference: run_ftgrid.sh counts every DIR.e1fail.*, here only those whose run.key is the
#            current one, so a leftover of an earlier panel, adapter or argument vector does not turn a first failure into the final one); the
#            recorded scoring config of each pass equals the real s0's pass of the same arm except `lora` (ftb_panel verify_scores). The same
#            sticky rule: a pass that would be scored again under another run.key (a re-made adapter) while FAILED_INTEGRITY sits in like/,
#            swap/ or a .stale.* of that adapter is refused (exit 1) unless the override is set. Then the registered zero-shot scores are linked
#            into the three scores trees (ftb_panel link)
#   stage 4  (CPU) ftgrid_report (unchanged) per arm: --models zeroshot,s0,s1,s2 (FT-N: zeroshot,s0,s1) --scores_root
#            outputs/confrec/ftgrid_ftb/scores/<tag> --out outputs/confrec/ftgrid_ftb/report/<tag>/ml1m.json. Each arm's adapters are the fine-tuned
#            models s0-s2 of that report: every P1, E-B and Holm block of these files is a computation on that arm's adapters under the registered
#            names and no decision of it is registered (report/NOTE_FTB.txt says so). Then ftb_reading build -> READ_ROOT/ml1m.json and
#            ml1m.csv: the paired contrasts and the reading rule of addendum 13 section 4 (status exploratory), which is the only place where
#            the arms are compared. The registered report of ML-1M is never touched (checked)
# ftgrid_extra is NOT run by this script. Its runner run_ftextra.sh refuses this root (it takes the three registered roots only, and a DRY_RUN
# root must lie outside outputs/confrec/ftgrid*), so the module is run by hand, as it is, once per arm, for the blocks E-W (G_wu), E-J and
# E-G (the e-share); it reads the arm's like and swap passes and the registered zero-shot link (the decomposition arms it does not need are
# absent and are not loaded). The reading recomputes G_wu itself through the same module's functions and cross-checks these files when they
# exist (run them, then stage 4 again):
#     python -m src.confrec.ftgrid_extra build --domain ml1m --split outputs/confrec/ftgrid/panels/ml1m/ftgrid_split.json \
#       --panels outputs/confrec/ftgrid/panels/ml1m --scores_root outputs/confrec/ftgrid_ftb/scores/b --models zeroshot,s0,s1,s2 \
#       --raw data/raw --out outputs/confrec/ftgrid_ftb/extra/b/ml1m.json --root_label llama --n_boot 2000
#     (--scores_root .../scores/r --out .../extra/r/ml1m.json likewise; FT-N: scores/n, --models zeroshot,s0,s1, extra/n/ml1m.json)
# --root_label llama is deliberate: the roots are outside the registered ones, so any label is accepted by the module, and only `llama`
# keeps a file from being eligible for an A3-6 family (family_eligible_panel false, no FT_C_reading); the module also records that the Qwen
# backbone does not belong to that root as an input problem, which makes `summarize` refuse the file, as it must (an exploratory file is
# never summarized).
# Re-runnable: a finished step is skipped (markers newer than their inputs); a scoring dir is skipped when report.json exists and run.key
# (panel sha1, model, variant, adapter-weights sha1, args) is unchanged, else it is moved to DIR.stale.<time>.
# GPU time (addendum 13 section 5 says about 6 GPU-hours for the eight adapters and their scoring; checked against the file times of the real
# adapters and passes recorded by run_ftlen.sh's header): training scales with K, the number of examples of the panel: the registered
# ML-1M adapter took about 46 min on the full TRAIN panel, so an adapter of K examples takes about 46 min x K / N; scoring is the registered
# one: about 9 min for the like pass and about 12.5 min for the swap pass of one adapter. Eight adapters: 8 x 46 min x K / N + 8 x (9 + 12.5)
# min = 6.1 h x K / N + 2.9 h (the report prints K and N with the panels).
# DRY_RUN=1: the same chain, CPU only, on run_ftgrid.sh's own DRY_RUN world of ML-1M (synthetic raw data, panels, Gate-FT context, s0-s2 and
# their like and decomposition passes, the temporary pilot log), built first by `DRY_RUN=1 STAGES=0,1,2,3,4 run_ftgrid.sh ml1m`; skipped once
# it exists. Everything lies in a temporary directory outside outputs/confrec: tmp_outputs/ftb_dryrun/{ftgrid: the world, ftgrid_ftb: the
# FT-B root, ftb_reading: the reading root}. The DRY_RUN guard is an allow-list: an OUT_ROOT, a READ_ROOT and a world must, in any spelling
# and through any link, lie under tmp_outputs of the repo, or outside the repo's parent directory (so never in the repo: outputs/ with the
# registered roots ftgrid*, ftmethod, gateft, gatefix and every other result directory, data/, src/, scripts/, docs/, tests/, idea-stage/,
# Paper/; never beside it: a sibling checkout, the data next to it, ../outside_repo; never above it); anything else is refused with exit 2,
# and so is a root that is, or lies inside, the world it reads or another root of the run, and a world that holds a link leading anywhere
# else. A relative root resolves from the repo root; every comparison is made on lower-case canonical forms. The trainer and the scorer are
# run_ftgrid.sh's stand-ins (the real argparse, the real scorer code with a fake model); every other step is the real code, the freeze checks
# included (on the temporary pilot log). The rehearsal first shows that an empty pilot log fails the amendment, core and FT-B record checks;
# the human step of recording the FT-B files is done on the temporary log unless DRY_NO_RECORD=1 (the record is then missing and the run is
# refused). Without symlinks (a Windows machine) DRY_RUN copies the real zero-shot scores instead of linking them. DRY_GATE and DRY_E1_FAIL
# act as in run_ftgrid.sh (DRY_E1_FAIL=s0/swap: the pass of an adapter whose scores path contains it fails E1; note that every arm's tree
# names its adapters s0-s2: DRY_E1_FAIL=b/ml1m/s0/like fails FT-B's s0). DRY_NO_WORLD=1 neither builds nor repairs that world (a test hook): a
# damaged world is refused as in a real run, not healed first. DRY_NO_RECORD and DRY_NO_WORLD are 0 or 1 too.
# Exit codes (as run_ftgrid.sh): 0 done; 1 error or an inconsistent recorded input; 2 usage or refused input; 4 refused by the freeze /
# gate rules.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
D="${1:-}"
case "$D" in
  ml1m) ;;
  *) echo "usage: bash scripts/sigir/run_ftb.sh ml1m   (ML-1M is the only dataset of FT-B, FT-S and FT-N)" >&2; exit 2 ;;
esac
# the switches are 0 or 1 and nothing else: DRY_RUN=yes must not start a real job, nor DRY_RUN= a rehearsal
DRY_RUN="${DRY_RUN-0}"
DRY_NO_RECORD="${DRY_NO_RECORD-0}"
DRY_NO_WORLD="${DRY_NO_WORLD-0}"
for sw in DRY_RUN DRY_NO_RECORD DRY_NO_WORLD; do
  case "${!sw}" in 0|1) ;; *) echo "$sw must be 0 or 1 (it is '${!sw}'): nothing was started" >&2; exit 2 ;; esac
done
# a deliberate retry of an adapter whose like or swap pass failed E1 twice (see retry_guard): FTB_ALLOW_RETRY names the adapters, and
# FTB_RETRY_REASON, which must then be non-empty, is logged
FTB_ALLOW_RETRY="${FTB_ALLOW_RETRY-}"
FTB_RETRY_REASON="${FTB_RETRY_REASON-}"
if [ -n "$FTB_ALLOW_RETRY" ]; then
  IFS=',' read -r -a RETRY_LIST <<< "$FTB_ALLOW_RETRY"
  for x in "${RETRY_LIST[@]}"; do
    case "$x" in b0|b1|b2|r0|r1|r2|n0|n1) ;;
      *) echo "FTB_ALLOW_RETRY must be a comma list of b0, b1, b2, r0, r1, r2, n0, n1 (it is '$FTB_ALLOW_RETRY'): nothing was started" >&2
         exit 2 ;; esac
  done
  if [ -z "$FTB_RETRY_REASON" ]; then
    echo "FTB_ALLOW_RETRY=$FTB_ALLOW_RETRY needs a non-empty FTB_RETRY_REASON (it is logged): nothing was started" >&2; exit 2
  fi
fi
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  set +u; source /root/miniconda3/etc/profile.d/conda.sh; conda activate lumen; set -u
  PY=python
fi
REG=outputs/confrec/ftgrid        # the registered grid root: read, never written
REGB=outputs/confrec/ftgrid_ftb   # the FT-B root: everything this script writes except the reading
REGR=outputs/confrec/ftb_reading  # the reading root: ml1m.json, ml1m.csv, the CF reference cache
DRYB=tmp_outputs/ftb_dryrun       # DRY_RUN's world and roots: a temporary directory outside outputs/confrec
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
  OUT_ROOT="${OUT_ROOT:-$DRYB/ftgrid_ftb}"
  READ_ROOT="${READ_ROOT:-$DRYB/ftb_reading}"
else
  MODEL="${MODEL:-/root/autodl-tmp/lumen/models/Qwen3-8B}"
  GRID="$REG"
  OUT_ROOT="${OUT_ROOT:-$REGB}"
  READ_ROOT="${READ_ROOT:-$REGR}"
fi
while [ "${OUT_ROOT%/}" != "$OUT_ROOT" ]; do OUT_ROOT="${OUT_ROOT%/}"; done      # ftgrid_ftb// -> ftgrid_ftb
while [ "${READ_ROOT%/}" != "$READ_ROOT" ]; do READ_ROOT="${READ_ROOT%/}"; done
if [ -z "$OUT_ROOT" ]; then echo "OUT_ROOT is empty or the filesystem root: FT-B has one registered root, $REGB" >&2; exit 2; fi
if [ -z "$READ_ROOT" ]; then echo "READ_ROOT is empty or the filesystem root: the reading has one registered root, $REGR" >&2; exit 2; fi
# Everything this script writes below the FT-B root, as a path below it, with the temporary names the bound tools write beside their
# products (ftgrid_report: report/<tag>/D.json.tmp and D_tables.csv.tmp; the builder: <file>.tmp). The three links it makes itself,
# scores/<tag>/D/zeroshot, are not listed: ftb_panel link asserts them, and sweep_links below allows no other link. The scorer's like/*.tmp
# names are not enumerated: the sweep and the removal of stale .tmp files before each scorer run cover them
ADAPTERS=(b0 b1 b2 r0 r1 r2 n0 n1)
TAGS=(b r n)
PANEL_FILES=(train_b.jsonl train_b.manifest.json train_r0.jsonl train_r0.manifest.json train_r1.jsonl train_r1.manifest.json
             train_r2.jsonl train_r2.manifest.json train_n.jsonl train_n.manifest.json)
TREE=(ftb "ftb/$D" adapters "adapters/$D" scores report report/NOTE_FTB.txt report/NOTE_FTB.txt.tmp report/NOTE_FTB_overrides.txt build "build/$D"
      "build/$D/done")
for f in "${PANEL_FILES[@]}"; do TREE+=("ftb/$D/$f" "ftb/$D/$f.tmp"); done
for a in "${ADAPTERS[@]}"; do TREE+=("adapters/$D/$a"); done
for t in "${TAGS[@]}"; do
  TREE+=("scores/$t" "scores/$t/$D" "report/$t" "report/$t/$D.json" "report/$t/$D.json.tmp" "report/$t/${D}_tables.csv"
         "report/$t/${D}_tables.csv.tmp")
  for k in 0 1 2; do
    if [ "$t" = n ] && [ "$k" = 2 ]; then continue; fi
    TREE+=("scores/$t/$D/s$k" "scores/$t/$D/s$k/like" "scores/$t/$D/s$k/swap")
  done
done
TREE_READ=("$D.json" "$D.json.tmp" "$D.csv" "$D.csv.tmp" "${D}_refs.json" "${D}_refs.json.tmp")
if [ "$DRY_RUN" = 1 ]; then SPELL=$OUT_ROOT; SPELLR=$READ_ROOT; else SPELL=$REGB; SPELLR=$REGR; fi    # the spelling the files are written with
ARGS=(. .. "$OUT_ROOT" "$READ_ROOT" "$GRID")
for r in "${TREE[@]}"; do ARGS+=("$SPELL/$r"); done
for r in "${TREE_READ[@]}"; do ARGS+=("$SPELLR/$r"); done
# the canonical forms decide, never the spelling: a string comparison is defeated by //, ., .., an absolute path, a link or letter case
canon_all "${ARGS[@]}" || { echo "cannot resolve OUT_ROOT=$OUT_ROOT, READ_ROOT=$READ_ROOT or the paths below them" >&2; exit 2; }
ROOTC=${CANON[0]}; PARENTC=${CANON[1]}; OUTC=${CANON[2]}; READC=${CANON[3]}; GRIDC=${CANON[4]}
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
  for p in "$OUTC" "$READC" "$GRIDC"; do
    if ! dry_ok "$p"; then
      echo "DRY_RUN=1 never writes to a registered output root, nor anywhere in the repo or beside it ($p): a rehearsal root is under" \
        "$ROOTC/tmp_outputs (default $DRYB) or outside $PARENTC; a relative root resolves from the repo root $ROOTC" >&2
      exit 2
    fi
  done
  if under "$OUTC" "$GRIDC" || under "$READC" "$GRIDC" || under "$OUTC" "$READC" || under "$READC" "$OUTC"; then
    echo "DRY_RUN=1: OUT_ROOT=$OUT_ROOT resolves to $OUTC and READ_ROOT=$READ_ROOT to $READC: a root of the run is, or lies inside, the" \
      "synthetic world it reads or the other root" >&2; exit 2
  fi
elif [ "$OUTC" != "$ROOTC/$REGB" ]; then
  # canonical equality with the registered root below the repo root: no link may lie between the repo root and the FT-B root either
  echo "OUT_ROOT=$OUT_ROOT resolves to $OUTC, not to $ROOTC/$REGB: FT-B has one registered root, $REGB, reached through no link (the" \
    "grid root $REG is read, never written); use DRY_RUN=1 for a rehearsal" >&2
  exit 2
elif [ "$READC" != "$ROOTC/$REGR" ]; then
  echo "READ_ROOT=$READ_ROOT resolves to $READC, not to $ROOTC/$REGR: the reading has one registered root, $REGR, reached through no link;" \
    "use DRY_RUN=1 for a rehearsal" >&2
  exit 2
else
  OUT_ROOT=$REGB
  READ_ROOT=$REGR
fi
# nothing written below the roots may leave them through a link: every path of the trees resolves, links included, to the same place
# below the canonical root
for i in "${!TREE[@]}"; do
  expect="$OUTC/${TREE[$i]}"
  expect="${expect,,}"
  if [ "${CANON[$((i + 5))]}" != "$expect" ]; then
    echo "FT-B refused: $SPELL/${TREE[$i]} resolves to ${CANON[$((i + 5))]}, not to $expect: a link between the FT-B root and the" \
      "files this script writes would redirect them (nothing was written)" >&2
    exit 2
  fi
done
for j in "${!TREE_READ[@]}"; do
  expect="$READC/${TREE_READ[$j]}"
  expect="${expect,,}"
  if [ "${CANON[$((j + 5 + ${#TREE[@]}))]}" != "$expect" ]; then
    echo "FT-B refused: $SPELLR/${TREE_READ[$j]} resolves to ${CANON[$((j + 5 + ${#TREE[@]}))]}, not to $expect: a link between the reading" \
      "root and the files this script writes would redirect them (nothing was written)" >&2
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
# sweep_links [full]: the only links below the FT-B root are the scores/<tag>/ml1m/zeroshot that ftb_panel link makes (and asserts), and
# there is none below the reading root. Any other link, in particular a planted temporary name of a bound tool (report/<tag>/D.json.tmp,
# like/*.tmp, ml1m.json.tmp), is refused with exit 2 before a tool can write through it. "full" (the start of a run) also resolves each
# allowed link and compares it with the registered zero-shot score directory
sweep_links() {
  local -a found=() chk=() exp=()
  local l rel n i
  if [ -d "$READ_ROOT" ]; then
    mapfile -t found < <(find "$READ_ROOT" -type l 2> /dev/null)
    if [ "${#found[@]}" -gt 0 ]; then
      echo "FT-B refused: ${found[0]} is a link: there is no link below $READ_ROOT, and a tool would write through it (nothing was" \
        "written)" >&2
      exit 2
    fi
  fi
  [ -d "$OUT_ROOT" ] || return 0
  mapfile -t found < <(find "$OUT_ROOT" -type l 2> /dev/null)
  for l in "${found[@]}"; do
    rel=${l#"$OUT_ROOT"/}
    if ! [[ $rel =~ ^scores/(b|r|n)/ml1m/zeroshot$ ]]; then
      echo "FT-B refused: $l is a link: the only links below $OUT_ROOT are scores/<b|r|n>/ml1m/zeroshot, and a tool would write through" \
        "it (nothing was written)" >&2
      exit 2
    fi
    chk+=("$l"); exp+=("$GRID/scores/ml1m/zeroshot")
  done
  if [ "${1:-}" = full ] && [ "${#chk[@]}" -gt 0 ]; then
    n=${#chk[@]}
    canon_all "${chk[@]}" "${exp[@]}" || { echo "cannot resolve the links below $OUT_ROOT" >&2; exit 2; }
    for ((i = 0; i < n; i++)); do
      if [ "${CANON[$i]}" != "${CANON[$((i + n))]}" ]; then
        echo "FT-B refused: ${chk[$i]} resolves to ${CANON[$i]}, not to the registered zero-shot score directory ${CANON[$((i + n))]}" >&2
        exit 2
      fi
    done
  fi
}
sweep_links full
if [ "$(basename "$MODEL")" != Qwen3-8B ]; then
  echo "$MODEL is not Qwen3-8B: FT-B, FT-S and FT-N are registered for the Gate-FT backbone only (addendum 13 section 2; the Llama" \
    "replication is conditional and is not run by this script)" >&2; exit 2
fi
STAGES=" $(printf '%s' "${STAGES:-all}" | tr ',' ' ') "
for s in $STAGES; do
  case "$s" in 1|2|3|4|all) ;; *) echo "unknown stage '$s' in STAGES (1-4, all)" >&2; exit 2 ;; esac
done
case "$STAGES" in *" all "*) STAGES="$STAGES 1 2 3 4 " ;; esac
want() { case "$STAGES" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

P="$GRID/panels/$D"                 # the grid's panels of D (read)
SPLIT="$P/ftgrid_split.json"
RS="$GRID/scores/$D"                # the real scores: zeroshot (linked into the FT-B root) and s0-s2's like and swap passes (their records)
QA="$OUT_ROOT/adapters/$D"          # the adapters b0 ... n1
QS="$OUT_ROOT/scores"               # scores/<tag>/D/zeroshot: a link to the registered zero-shot scores; scores/<tag>/D/s<k>/{like,swap}
FTB="$OUT_ROOT/ftb/$D"              # the panels and their manifests
REGREP="$GRID/report/$D.json"       # the registered report of D: never touched
B="$OUT_ROOT/build/$D"              # step markers
DONE="$B/done"
READ_JSON="$READ_ROOT/$D.json"
REFS="$READ_ROOT/${D}_refs.json"
MANIFESTS=("$FTB/train_b.manifest.json" "$FTB/train_r0.manifest.json" "$FTB/train_r1.manifest.json" "$FTB/train_r2.manifest.json"
           "$FTB/train_n.manifest.json")
# addendum 13 section 5: recorded before the first adapter is trained (the manifests pin the built panels)
FTB_FILES=(scripts/sigir/run_ftb.sh src/confrec/ftb_panel.py src/confrec/ftb_reading.py tests/test_confrec_ftb.py tests/test_confrec_ftb_run.py)
SWAP_K=8                            # addendum 13 section 2: the swap-prior arm of the registered decomposition stage
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
RA="$GT/adapters"; TRAIN_REF="$GT/train.jsonl"     # the real adapters of ML-1M are Gate-FT's (never retrained, scored by their path)

# ---- helpers ----
# fresh OUT DEP...: OUT exists and is newer than every DEP (a missing DEP makes it stale)
fresh() { local o=$1 i; shift; [ -e "$o" ] || return 1; for i in "$@"; do { [ -e "$i" ] && [ "$o" -nt "$i" ]; } || return 1; done; }
# step PRODUCT DEP... -- CMD...: run CMD unless PRODUCT exists and its marker DONE/<sha1 of the path>.<basename>.done is newer than every
# DEP; the marker is touched only after CMD succeeds, so an interrupted step always reruns. (run_ftgrid.sh's marker is the basename alone;
# here the three arms' reports and the reading are all called ml1m.json, so the marker is named by the path as well: one deliberate
# difference)
step() {
  local prod=$1 deps=() mark; shift
  while [ "$1" != "--" ]; do deps+=("$1"); shift; done; shift
  mark="$DONE/$(printf '%s' "$prod" | sha1sum | cut -c1-12).$(basename "$prod").done"
  if [ -e "$prod" ] && fresh "$mark" "${deps[@]}"; then echo "[skip] $prod"; return 0; fi
  rm -f "$mark"
  "$@" || return 1
  mkdir -p "$DONE"; touch "$mark"
}
adapter_done() { [ -f "$1/train_config.json" ] && ls "$1"/adapter_model.* >/dev/null 2>&1; }
# e1_ok DIR: section 2 / E1 of a finished scorer run: censored-2 share <= 0.5%, no overlength prompt, mean Yes+No mass
# >= 0.95 on the main prompts; a swap run's donor prompts also need the censored-2 share and no overlength prompt
# (run_ftgrid.sh's function, verbatim: tests/test_confrec_ftb_run.py compares the two)
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
# scores_of NAME: the scores directory of adapter NAME (b0 -> scores/b/ml1m/s0): the arm's tree, the adapter's seed as the model name
scores_of() { echo "$QS/${1:0:1}/$D/s${1:1}"; }
# retry_guard NAME: FAILED_INTEGRITY is sticky per adapter. When a pass of NAME failed E1 twice (its scores directory's like or swap, or a
# .stale.* of either, holds FAILED_INTEGRITY), re-making the adapter or scoring it under another key would start a fresh E1 budget, and
# section 2 says it is reported as missing, never replaced: stages 2 and 3 refuse (exit 1, naming the directory) unless FTB_ALLOW_RETRY
# names the adapter and FTB_RETRY_REASON says why; the override is appended to report/NOTE_FTB_overrides.txt (once per run)
retry_guard() {
  local name=$1 d hit="" arm
  for arm in like swap; do
    for d in "$(scores_of "$name")/$arm" "$(scores_of "$name")/$arm".stale.*; do
      if [ -f "$d/FAILED_INTEGRITY" ]; then hit=$d; break 2; fi
    done
  done
  if [ -z "$hit" ]; then return 0; fi
  case ",$FTB_ALLOW_RETRY," in
    *",$name,"*) ;;
    *) echo "FT-B refused: a pass of $name failed E1 twice ($hit/FAILED_INTEGRITY): making $name again or scoring it under another key" \
         "would reset its E1 budget, and section 2 reports it as missing, never replaced. A deliberate retry needs" \
         "FTB_ALLOW_RETRY=$name and FTB_RETRY_REASON='why' (logged in $OUT_ROOT/report/NOTE_FTB_overrides.txt)" >&2
       exit 1 ;;
  esac
  case "${RETRY_LOGGED:-}" in *",$name,"*) return 0 ;; esac
  mkdir -p "$OUT_ROOT/report"
  printf '%s\t%s\t%s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FTB_RETRY_REASON" >> "$OUT_ROOT/report/NOTE_FTB_overrides.txt"
  RETRY_LOGGED="${RETRY_LOGGED:-},$name,"
  echo "[retry] $name ($hit/FAILED_INTEGRITY): FTB_ALLOW_RETRY, logged in $OUT_ROOT/report/NOTE_FTB_overrides.txt" >&2
}
# score DATA DIR --lora A [--swap_k K]: pyes_scorer with SARGS, the scorer flags of the real s0 passes (fp16, top-50 logprobs, max_model_len
# 4096, 100-user chunks, the selected variant, yes/no readout, like: run_ftgrid.sh's score(), as its record says); completion
# marker DIR/report.json, run.key = panel sha1 + model + variant + adapter weights sha1 + args (the format of run_ftgrid.sh)
score() {
  local data=$1 dir=$2 key old="" lora="" prev="" x wsha=-; shift 2
  for x in "$@"; do if [ "$prev" = --lora ]; then lora="$x"; fi; prev="$x"; done
  if [ -n "$lora" ]; then wsha=$(cat "$lora"/adapter_model.* | sha1sum | cut -d' ' -f1); fi
  key="$(sha1sum "$data" | cut -d' ' -f1) $MODEL $VARIANT $wsha $*"
  if [ -f "$dir/run.key" ]; then old=$(cat "$dir/run.key"); fi
  if [ "$old" = "$key" ] && [ -f "$dir/report.json" ]; then echo "[skip] $dir: scored"; return 0; fi
  if [ -n "${RETRY_NAME:-}" ]; then retry_guard "$RETRY_NAME"; fi      # a failed integrity is not escaped by a new key
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
check_record() { "$PY" -m src.confrec.ftb_panel record --pilot_log "${1:-$PILOT_LOG}" --files "${FTB_FILES[@]}" "${MANIFESTS[@]}"; }
# ensure_record: the FT-B record is in the pilot log (exit 4 otherwise). DRY_RUN does the human step on the temporary log first
# (--append writes the missing lines), unless DRY_NO_RECORD=1
ensure_record() {
  if [ "$DRY_RUN" = 1 ] && [ "$DRY_NO_RECORD" != 1 ]; then
    "$PY" -m src.confrec.ftb_panel record --pilot_log "$PILOT_LOG" --files "${FTB_FILES[@]}" "${MANIFESTS[@]}" --append || exit 4
  else
    check_record || exit 4
  fi
}
# rehearse: DRY_RUN only, once per world: an empty pilot log fails the amendment, core and FT-B record checks
rehearse() {
  if [ -f "$B/rehearsed" ]; then return 0; fi
  [ -f "$DRYD/PILOT_LOG.empty.md" ] || printf '# empty temporary pilot log\n' > "$DRYD/PILOT_LOG.empty.md"
  if check_amendment "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 || check_core "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1 \
      || check_record "$DRYD/PILOT_LOG.empty.md" > /dev/null 2>&1; then
    echo "DRY_RUN: a freeze or record check passed on an empty pilot log" >&2; exit 1
  fi
  echo "[dry] gate rehearsal: an empty pilot log fails the amendment, core and FT-B record checks"
  mkdir -p "$B"; touch "$B/rehearsed"
}
# gates: everything the registered text asks before a GPU job (Amendment 3 section 0, addendum 13 section 5)
gates() {
  if [ "$GATE_DECISION" != GATE_FT_PASS ]; then
    echo "FT-B refused: Gate-FT decision $GATE_DECISION (section 0: FT-B runs only after a recorded GATE_FT_PASS, $GT/gate_ft.json)" >&2
    exit 4
  fi
  if [ "$DRY_RUN" = 1 ]; then rehearse; fi
  check_amendment || { echo "FT-B refused: the freeze record of Amendment 3 (stage amendment) is not in $PILOT_LOG" >&2; exit 4; }
  check_core || { echo "FT-B refused: the full record of Amendment 3 (stage core) is not in $PILOT_LOG" >&2; exit 4; }
  if [ "$SEL_DECISION" = FIX_FOUND ] && [ ! -f "$G/confirm/gate.json" ]; then
    echo "FT-B refused: selection.json found a fix but the gate-fix confirm stage has recorded no gate.json (section 4)" >&2
    exit 4
  fi
  for f in "${MANIFESTS[@]}"; do
    [ -f "$f" ] || { echo "missing $f (stage 1 builds the panels and their manifests: record them with the code before stage 2)" >&2; exit 1; }
  done
  ensure_record
}
# read_info: SEL_DECISION, SEL_PROMPT (selection.json), SPLIT_VARIANT (the split) and GATE_DECISION (gate_ft.json; missing if absent)
read_info() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftb_panel info --selection "$SEL" --split "$SPLIT" --gate "$GT/gate_ft.json") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  { read -r SEL_DECISION; read -r SEL_PROMPT; read -r SPLIT_VARIANT; read -r GATE_DECISION; } < <(printf '%s\n' "$out" | tr -d '\r')
}
# verify_panels: the panels on disk are the panels. They are recomputed from the registered train.jsonl (seconds of CPU) and must equal the
# ten files of the directory byte for byte (exit 1 otherwise): the step marker and a manifest alone prove nothing about a file that was
# replaced afterwards. Once per run: stages 2-4 share it
verify_panels() {
  if [ -n "${PANELS_OK:-}" ]; then return 0; fi
  local f
  for f in "$P/train.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (the inputs of stage 1)" >&2; exit 1; }
  done
  "$PY" -m src.confrec.ftb_panel verify --domain "$D" --train "$P/train.jsonl" --split "$SPLIT" --out_dir "$FTB" || exit $?
  PANELS_OK=1
}
# recipe: RECIPE = the trainer flags the real adapters s0-s2 recorded in their train_config.json, except --train, --out and --seed
# (ftb_panel validates them: one registered recipe, this model, variant and TRAIN file; ML-1M: Gate-FT's adapters)
recipe() {
  local out rc=0
  out=$("$PY" -m src.confrec.ftb_panel recipe --adapters "$RA" --model "$MODEL" --variant "$VARIANT" --train_ref "$TRAIN_REF" \
    --train_file "$P/train.jsonl" --split "$SPLIT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  mapfile -t RECIPE < <(printf '%s\n' "$out" | tr -d '\r')
  echo "[recipe] trainer flags recorded by the real s0-s2 (--train, --out and --seed replaced): ${RECIPE[*]}"
}
# scoring_recipe: SARGS = the scorer flags of the real s0's like pass (its run.key and the config of its report.json); the swap pass
# of the real s0 must be recorded with the same flags (the arm adds only --swap_k 8 and its panel)
scoring_recipe() {
  local like swap rc=0
  if [ -n "${SARGS_READ:-}" ]; then return 0; fi
  like=$("$PY" -m src.confrec.ftb_panel scoring --pass like --scores "$RS" --adapters "$RA" --data "$P/eval.jsonl" --model "$MODEL" \
    --variant "$VARIANT") || rc=$?
  [ "$rc" = 0 ] || exit "$rc"
  swap=$("$PY" -m src.confrec.ftb_panel scoring --pass swap --scores "$RS" --adapters "$RA" --data "$P/eval_sd_test.jsonl" \
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
# verify_adapters NAMES [--write]: each adapter's recorded arguments equal the real s0's except --train, --out, --seed, and it trained on
# exactly the K examples of its panel; its provenance
verify_adapters() {
  local names=$1; shift
  "$PY" -m src.confrec.ftb_panel verify_adapter --adapters "$QA" --ref_adapters "$RA" --names "$names" --panels_dir "$FTB" "$@" || exit $?
}
# panel_of NAME: the panel an adapter trains on (b -> train_b, r<k> -> train_r<k>, n -> train_n)
panel_of() {
  case "$1" in
    b*) echo "$FTB/train_b.jsonl" ;;
    r*) echo "$FTB/train_r${1:1}.jsonl" ;;
    n*) echo "$FTB/train_n.jsonl" ;;
  esac
}
# train_adapter NAME: the recorded recipe on the adapter's panel; skipped when train_config.json and weights exist (their provenance
# was verified before the first training)
train_adapter() {
  local name=$1 out="$QA/$1" seed=${1:1} panel
  panel=$(panel_of "$name")
  if adapter_done "$out"; then echo "[skip] adapter $out exists"; return 0; fi
  [ -f "$panel" ] || { echo "missing $panel (stage 1)" >&2; exit 1; }
  retry_guard "$name"                  # before any training: a failed integrity is not escaped by making the adapter again
  rm -rf "$out"
  "$MPY" -m src.confrec.train_lora_yesno --train "$panel" --out "$out" --seed "$seed" "${RECIPE[@]}" 2>&1 \
    | grep -vE "it/s\]|s/it\]" || true
  adapter_done "$out" || { echo "training of $out did not finish" >&2; exit 1; }
  verify_adapters "$name" --write
}
# verify_arm TAG PASS SEEDS: the recorded scoring config of the arm's passes equals the real s0's pass of the same kind except `lora`
verify_arm() {
  "$PY" -m src.confrec.ftb_panel verify_scores --arm "$1" --pass "$2" --scores "$QS/$1/$D" --ref_scores "$RS" --adapters "$QA" \
    --seeds "$3" || exit $?
}
# link_scores: the registered zero-shot scores become a symlink of each arm's scores tree (never a copy; DRY_RUN without symlinks copies)
link_scores() {
  local t
  for t in "${TAGS[@]}"; do
    "$PY" -m src.confrec.ftb_panel link --real_scores "$RS" --q_scores "$QS/$t/$D" --models zeroshot $LINK_FLAG || exit $?
  done
}
# seeds_of TAG: the seeds of an arm's adapters
seeds_of() { if [ "$1" = n ]; then echo "0 1"; else echo "0 1 2"; fi; }
# models_of TAG: the --models of an arm's report
models_of() { if [ "$1" = n ]; then echo "zeroshot,s0,s1"; else echo "zeroshot,s0,s1,s2"; fi; }
# note_ftb: report/NOTE_FTB.txt, the sidecar that says what the files of this root are (written when missing or different)
note_ftb() {
  local f="$OUT_ROOT/report/NOTE_FTB.txt" tmp="$OUT_ROOT/report/NOTE_FTB.txt.tmp"
  mkdir -p "$OUT_ROOT/report"
  printf '%s\n' \
    "FT-B, FT-S and FT-N (Amendment 3 addendum 13; EXPLORATORY: no hypothesis, no Holm family, no claim-admission role), written by" \
    "scripts/sigir/run_ftb.sh." \
    "" \
    "1. The adapters b0-b2 (FT-B, item-balanced), r0-r2 (FT-S, size-matched random draw) and n0-n1 (FT-N, labels permuted within the kept" \
    "   items) are trained on panels built from the registered TRAIN panel (ftb/<D>/); they are not the registered adapters. Each arm has its" \
    "   own scores tree scores/<b|r|n>/<D>, in which its adapters carry the model names s0, s1, s2 that ftgrid_report accepts (b0 is" \
    "   scores/b/<D>/s0, r1 is scores/r/<D>/s1, n0 is scores/n/<D>/s0); zeroshot there is a link to the registered zero-shot scores." \
    "2. report/<b|r|n>/<D>.json and report/<b|r|n>/<D>_tables.csv are the unchanged ftgrid_report output for that arm. It reads the arm's" \
    "   adapters as the fine-tuned models s0-s2 of the registered grid: every P1, E-B and Holm block in them is a computation on that arm's" \
    "   adapters under the registered names, no decision of it is registered, and none is cited. The reading (outputs/confrec/ftb_reading/" \
    "   <D>.json) is the only place where the arms are compared and where the addendum's labels are given." \
    "3. ftgrid_extra is not run by this script (its runner refuses this root); see the header of scripts/sigir/run_ftb.sh for the command" \
    "   that builds its blocks on each arm and for the root label that keeps the files out of every A3-6 family." > "$tmp"
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
      && [ -f "$RS/s1/swap/report.json" ] && [ -f "$RS/s2/like/report.json" ] && [ -f "$RS/s2/swap/report.json" ] \
      && [ -f "$P/eval_sd_test.jsonl" ]; then
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
echo "run_ftb $D: model $MODEL, root $OUT_ROOT, variant $VARIANT, stages:$STAGES"

# ---- the registered gates come first: a refused run touches nothing ----
GATED=0
for s in 2 3 4; do if want "$s"; then GATED=1; fi; done
if [ "$GATED" = 1 ]; then gates; fi

# ================= stage 1: the FT-B, FT-S and FT-N panels (CPU) =================
if want 1; then
  sweep_links
  echo "== stage 1: the FT-B, FT-S and FT-N panels ($D)"
  for f in "$P/train.jsonl" "$SPLIT"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  mkdir -p "$FTB"
  find "$FTB" -name '*.tmp' -delete 2> /dev/null || true             # a stale temporary file of the builder is not written through
  step "${MANIFESTS[4]}" "$P/train.jsonl" "$SPLIT" src/confrec/ftb_panel.py src/confrec/ftprune.py src/confrec/ftgrid_data.py \
      src/confrec/split_panel.py -- \
    "$PY" -m src.confrec.ftb_panel build --domain "$D" --train "$P/train.jsonl" --split "$SPLIT" --out_dir "$FTB"
  verify_panels        # also when the step was skipped: the sha1 printed next is the recomputed panels', not a marker's
  echo "[stage 1] pilot-log lines for the panels of $D (record them with the code before stage 2):"
  for f in "${PANEL_FILES[@]}"; do echo "$FTB/$f = $(sha1sum "$FTB/$f" | cut -d' ' -f1)"; done
fi

# ================= stage 2: the adapters b0-b2, r0-r2, n0, n1 (GPU) =================
if want 2; then
  sweep_links
  echo "== stage 2: adapters b0-b2, r0-r2, n0-n1 ($D)"
  verify_panels        # before any training: the panels on disk are the recomputed panels, byte for byte
  recipe
  scoring_recipe       # pre-flight: the real s0's like and swap passes must be the registered ones before any GPU hour is spent
  have=""
  for a in "${ADAPTERS[@]}"; do if adapter_done "$QA/$a"; then have="${have:+$have,}$a"; fi; done
  if [ -n "$have" ]; then verify_adapters "$have"; fi
  for a in "${ADAPTERS[@]}"; do train_adapter "$a"; done
fi

# ================= stage 3: like and swap for the eight adapters (GPU) =================
if want 3; then
  sweep_links
  echo "== stage 3: like on eval.jsonl and swap (--swap_k $SWAP_K) on eval_sd_test.jsonl for b0-b2, r0-r2, n0-n1 ($D)"
  for f in "$P/eval.jsonl" "$P/eval_sd_test.jsonl"; do
    [ -f "$f" ] || { echo "missing $f (run_ftgrid.sh stage 0)" >&2; exit 1; }
  done
  verify_panels
  scoring_recipe
  for a in "${ADAPTERS[@]}"; do
    adapter_done "$QA/$a" || { echo "adapter $a of $D is missing or incomplete: $QA/$a (stage 2)" >&2; exit 1; }
  done
  names=""
  for a in "${ADAPTERS[@]}"; do names="${names:+$names,}$a"; done
  verify_adapters "$names"
  for t in "${TAGS[@]}"; do
    for k in $(seeds_of "$t"); do
      RETRY_NAME="$t$k" score "$P/eval.jsonl" "$QS/$t/$D/s$k/like" --lora "$QA/$t$k"
      verify_arm "$t" like "$k"
      RETRY_NAME="$t$k" score "$P/eval_sd_test.jsonl" "$QS/$t/$D/s$k/swap" --lora "$QA/$t$k" --swap_k "$SWAP_K"
      verify_arm "$t" swap "$k"
    done
  done
  link_scores
fi

# ================= stage 4: ftgrid_report per arm and the reading (CPU) =================
if want 4; then
  sweep_links
  echo "== stage 4: ftgrid_report per arm -> $OUT_ROOT/report/<b|r|n>/$D.json and the reading -> $READ_JSON ($D)"
  verify_panels
  for arm in like swap; do
    [ -f "$RS/zeroshot/$arm/report.json" ] || { echo "missing the registered zero-shot $arm pass $RS/zeroshot/$arm (run_ftgrid.sh $D)" >&2; exit 1; }
  done
  for t in "${TAGS[@]}"; do
    for k in $(seeds_of "$t"); do
      for arm in like swap; do
        { [ -f "$QS/$t/$D/s$k/$arm/report.json" ] || [ -f "$QS/$t/$D/s$k/$arm/FAILED_INTEGRITY" ]; } \
          || { echo "missing the $arm pass $QS/$t/$D/s$k/$arm (stage 3)" >&2; exit 1; }
      done
    done
  done
  link_scores
  note_ftb
  REG_BEFORE=$(registered_sha)
  RDEPS=()
  for t in "${TAGS[@]}"; do
    RDEPS=("$SPLIT" src/confrec/ftgrid_report.py)
    for f in "$QS/$t/$D"/*/*/report.json; do if [ -f "$f" ]; then RDEPS+=("$f"); fi; done
    mkdir -p "$OUT_ROOT/report/$t"
    rm -f "$OUT_ROOT/report/$t/$D.json.tmp" "$OUT_ROOT/report/$t/${D}_tables.csv.tmp"   # a stale temporary file of the report is not written through
    step "$OUT_ROOT/report/$t/$D.json" "${RDEPS[@]}" -- \
      "$PY" -m src.confrec.ftgrid_report --domain "$D" --split "$SPLIT" --panels "$P" --scores_root "$QS/$t" \
        --models "$(models_of "$t")" --raw "$RAW" --out "$OUT_ROOT/report/$t/$D.json" --n_boot "$N_BOOT" --seed 0
  done
  if [ "$(registered_sha)" != "$REG_BEFORE" ]; then
    echo "the registered report $REGREP changed: it is never touched (addendum 13)" >&2; exit 1
  fi
  RDEPS=("$SPLIT" src/confrec/ftb_reading.py src/confrec/ftgrid_report.py src/confrec/ftgrid_extra.py)
  for f in "$QS"/*/"$D"/*/*/report.json "$RS"/*/*/report.json "$OUT_ROOT"/report/*/"$D".json "$OUT_ROOT"/extra/*/"$D".json "${MANIFESTS[@]}"; do
    if [ -f "$f" ]; then RDEPS+=("$f"); fi
  done
  mkdir -p "$READ_ROOT"
  rm -f "$READ_JSON.tmp" "${READ_JSON%.json}.csv.tmp" "$REFS.tmp"          # a stale temporary file of the reading is not written through
  step "$READ_JSON" "${RDEPS[@]}" -- \
    "$PY" -m src.confrec.ftb_reading build --split "$SPLIT" --panels "$P" --grid_scores "$GRID/scores" --root "$OUT_ROOT" --raw "$RAW" \
      --refs "$REFS" --out "$READ_JSON" --n_boot "$N_BOOT" --seed 0
fi
for t in "${TAGS[@]}"; do
  for k in $(seeds_of "$t"); do
    for arm in like swap; do
      if [ -f "$QS/$t/$D/s$k/$arm/FAILED_INTEGRITY" ]; then
        echo "NOTE: $QS/$t/$D/s$k/$arm is FAILED_INTEGRITY: that pass of $t$k is reported as missing, never replaced (section 2)" >&2
      fi
    done
  done
done
echo "run_ftb $D: done (stages:$STAGES)"
