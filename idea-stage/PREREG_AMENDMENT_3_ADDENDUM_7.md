# Pre-registration amendment 3, addendum 7 (2026-10-05): S6 wording, composition diagnostics and a conditional item-stratified control

**Status: an extension of Amendment 3** (recorded with it; `ftgrid_freeze` treats every `PREREG_AMENDMENT_3_ADDENDUM_*.md` as part of the
amendment). Written before any S6 arm was built (stages A and B of `scripts/sigir/run_ftprune.sh` have not run), before any pruned adapter
existed and before any S6 outcome was read. **Read before writing:** the same material as addendum 6 (Gate-FT, the ML-1M grid report, the
two same-family reviews of the draft). The methodology review pointed out that S6 matches the arms per label class only, so a "beats random"
result could come from item composition, and that there is no noise ground truth.

## 1. Wording

RQ6 is asked as: **does pruning the training examples that the zero-shot model is most uncertain about help more than matched random
pruning?** P2 is called *uncertainty-selected pruning*; the phrase "noisy training examples" is dropped from the paper (nothing in the data
says which examples are noisy). The matched random arm P1 matches **class counts only**; the paper says so wherever the contrast is stated.

## 2. Composition diagnostics (descriptive, CPU)

For every pruned arm (P1, with the mean over seeds; P2; P3) and for P0, from the subset index files and the TRAIN rows named in the prune
manifest (no scoring, no outcome): (a) the share of TRAIN items that keep no example; (b) the mean absolute change of the item label rate
over the items with at least two TRAIN examples, weighted by their number of examples; (c) the share of the EVAL users' TEST pairs of ML-1M
whose item keeps at least one TRAIN example (the "seen" share of Amendment 3 section 1, recomputed on the arm's examples); (d) the share
of the arm's examples that belong to the head tercile of items by TRAIN popularity (terciles cut over the full TRAIN set). A new file,
`src/confrec/ftprune_compose.py`, computes them from the manifest; its sha1 is recorded in PILOT_LOG before the `prune` record. No bound
file changes.

## 3. A conditional control: the item-stratified random arm P1s

P1s is run **if and only if** the registered confirmatory contrast P2 - P1 reads BEATS (section 6 of Amendment 3). It removes, for every
cell (item, label class), exactly as many TRAIN examples as P2 removed from that cell, chosen uniformly at random within the cell by a draw
that depends on the seed (5 seeds, the section-2 recipe, the same scoring as P1 and P2), so the per-item class counts, the item label rates and
the item coverage equal P2's and only *which* examples of a cell are removed differs. The contrast **P2 - P1s** (seed-averaged TEST UAUC,
user-bootstrap CI) is descriptive; it reads **BEATS_WITHIN_ITEM** iff the CI lower bound is above 0 and all five paired differences are
positive, and otherwise the paper states that the advantage of P2 over P1 may come from item composition. The code is written after the
BEATS outcome only if it is triggered, and must implement exactly this definition (no choice is left to the implementer); the arm is outside
every Holm family.

## 4. Scope

S6 stays single-backbone, ML-1M only, and reaches the abstract only with a Llama replication (Amendment 3 section 6 unchanged); all other
rules of that section (the signals, the arms, the confirmatory test P2 - P1, the claim labels, the order) are unchanged.
