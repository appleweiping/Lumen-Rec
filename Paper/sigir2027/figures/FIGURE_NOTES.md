# Figure notes

Everything in this directory except `decomp.tex` and this file is written by `scripts/sigir/make_figures.py`. No number is typed
in the generator, in the captions below or in these notes: every plotted value is read from a committed result file at generation
time, and `figures_manifest.json` lists, for every figure and panel, the source files (relative path and sha1), every plotted
record as `<file>:<dotted key path>` with the plotted fields, the flags, the withheld values and the panels not run yet.
The live state (which panel has data, which file is missing) is the manifest's `status`, `missing` and `panels[].status`.

## 1. Files and commands

| file | what |
|---|---|
| `tracks.pdf` (F1), `shares.pdf` (F2), `serving.pdf` (F3) | vector PDFs, 7.0 in wide (double column), fonts embedded as TrueType, no creation date, byte-identical for identical inputs |
| `decomp.tex` (F4) | hand-written TikZ schematic, 3.3 in wide (one column), no data; loaded with `\input` |
| `figures_manifest.json` | provenance of the three plots and the sha1 of every file (see above) |

```
python scripts/sigir/make_figures.py --results docs/sigir/results --out Paper/sigir2027/figures [--only tracks,shares,serving,decomp]
python -m pytest tests/test_confrec_figures.py -q -p no:cacheprovider
```

`--only` keeps the manifest entries of the figures not asked for as long as their files still carry the recorded sha1. Exit code 2:
a result file that exists but is malformed (the message names the file and the key) or an unknown figure id.

## 2. What each figure shows, and the keys it reads

Field names are those that `scripts/sigir/fill_paper.py` reads for the same cells (`docs/sigir/PAPER_DATA_MAP.md`).

**F1 `tracks`** (about the same cells as Table `tab:tracks`, block A). One panel per (backbone, rated domain): Qwen3-8B on ML-1M, Toys,
Video Games, Sports, then Llama on ML-1M and Toys. Rows: zero-shot UAUC of the confidence, LoRA UAUC (seed mean, the three seeds as small
dots), then, in a shaded band, the references on the same rows: item mean, information-matched item mean, temporal MF.
- `grid/<qwen|llama>/<d>.json`: `meta.domain`, `meta.backbone`; `E_A.ZS.UAUC_TEST.per_model.zeroshot.{est,lo,hi}`;
  `E_A.FT.UAUC_TEST.mean_over_seeds.{est,lo,hi}` and `E_A.FT.UAUC_TEST.per_model.{s0,s1,s2}.est`;
  `E_A.<ZS|FT>.UAUC_TEST.references.{q_hat,mf}.{est,lo,hi}` (zero-shot rows first, as `fill_paper.ref_val`); `E_A.UAUC_TEST.rows.n_users`
  (panel header); `complete` and `available` of the regime blocks.
- `extra/<d>.json` (Qwen) or `extra/llama/<d>.json`: `E_J.<ZS|FT>.dUAUC_L_minus_q_hat_T.per_seed.<zeroshot|s0>.UAUC_b` (the UAUC of the
  matched mean; a point estimate, the file stores no interval), `E_J.<reg>.rows_E_A` (checked against the grid rows), `status.items_2_to_7`.
- `gft/gate_ft.json`: `decision` (LoRA values are drawn only after GATE_FT_PASS).

**F2 `shares`** (Table `tab:tracks`, block B). One column per Qwen rated panel. Top row: shares of the within-user variance of the
confidence logit: item prior (rho^2/r_8), non-prior (its complement) and the e-share; each for zero-shot and LoRA. Bottom row: the
information gain G (E-D, pooled) and G_wu (E-W, within-user) for zero-shot and LoRA, beside the gain of the MF residual with no LLM
feature (G_CF, G_CF_wu).
- grid: `E_D.ZS.shares.per_model.zeroshot.{item_prior_share,non_prior_share,shares_reading}`;
  `E_D.FT.shares.mean_over_seeds.*` and `E_D.FT.shares.per_model.<m>.*` (same keys); `E_D.<reg>.shares.rows.n_users`;
  `E_D.ZS.information_gain.per_model.zeroshot.G`; `E_D.FT.information_gain.{G_mean_over_seeds, per_model.<m>.G}`;
  `E_D.<reg>.information_gain.G_CF`.
- extra: `E_G.ZS.e_share.per_model.zeroshot.e_share`; `E_G.FT.e_share.{mean_over_seeds, per_model.<m>.e_share}`; `E_G.descriptive`;
  `E_W.ZS.G_wu.per_model.zeroshot`; `E_W.FT.G_wu.{mean_over_seeds, per_model.<m>}`; `E_W.<reg>.G_CF_wu`; `status.items_2_to_7`.

**F3 `serving`** (Table `tab:anatomy`, block C, as a curve). One panel per family unit of the next-item audit (Sports events 1001-10000,
Toys, Home, Tools): NDCG@10 among the served events against the fraction served, for serving the most confident events first (`p_max`)
and for the fixed-seed random subset.
- `aud/<d>.json`: `domain`; `segments.<seg>.n_events`; `segments.<seg>.questions.next.D_selective_serving.signals.{p_max,random}.curve.ndcg10[i].{coverage,est,lo,hi,n}`.
- Not drawn although the file holds them: the other signals (`margin`, `neg_entropy`, `max_logit`), the Sports quarantine segment
  `events_1_1000`, `gain_at_50_vs_full` (the registered contrast stays in the table), and every `p`, `p_boot`, `null` field.

**F4 `decomp`**: conceptual schematic, no data (a TikZ picture, see section 5).

None of the plots reads a p-value, a Holm-family field, `ci_excludes_0` or an admission label: a plot never implies a registered
decision. Intervals are the files' own (percentile bootstrap over users on rated panels and over events on next-item panels).

## 3. How uncertainty, status and absence are drawn

| situation | drawing | manifest |
|---|---|---|
| any interval stored in the file | bars (F1, F2) or a band (F3); a record without `lo`/`hi` is a point | `fields.lo`, `fields.hi` |
| `status.items_2_to_7 = exploratory` (ML-1M addendum-6 values) | hollow marker | `flags: ["exploratory"]`, `flag_sources` |
| `E_G.descriptive`, `descriptive_min_n`, fewer than 150 users or events | hollow marker | `flags: ["descriptive"]` |
| reference computed on other rows than the LLM score | hollow marker | `flags: ["rows_differ"]` |
| `shares_reading = uninterpretable` | no marker, `n/i` | `withheld[]`, the value is not listed |
| result file missing | hatched panel, `not run yet`, the reason and the file | panel `status: not_run`, `missing[]` with `kind: file` |
| one element missing inside a panel (e.g. no `extra` file) | `not run yet` at its place | panel `status: partial`, `missing[]` |
| block with `available: false`, null value | `not run yet` with the file's own reason | `missing[]` with `kind: block` |
| LoRA series before a recorded GATE_FT_PASS | not drawn | `missing[]` with `kind: gate` (or `file`) |
| LoRA regime incomplete (a seed missing or excluded) | not drawn, never replaced | `missing[]`, `reason` quotes the file |

The UAUC axes of F1 do not start at zero: the start is printed under the panels (it starts below chance and is lowered only if a value
needs it); NDCG@10 (F3), the shares (F2 top) and the gains (F2 bottom, zero line drawn) are not truncated.

Colour and marks: blue = zero-shot, orange = LoRA (as in every figure), neutral gray = non-LLM reference or random subset. The two hues
are slots 1 and 2 of the validated default palette (CVD separation passes; checked with the dataviz validator). Every series also has
its own marker (circle, diamond, square, triangle, plus) and row or position, so the figures read in greyscale; hollow versus filled
carries the exploratory/descriptive flag.

## 4. Where a figure still waits for data

The panels read `not run yet` for any file that is not in `docs/sigir/results`. The files a full set needs are exactly the ones
`PAPER_DATA_MAP.md` lists: `grid/{qwen,llama}/<d>.json`, `extra/{<d>,llama/<d>}.json` (the matched mean, the e-share, G_wu), `aud/<d>.json`
for the four next-item domains and `gft/gate_ft.json`. After `pull_results.ps1`, re-run the generator, then `fill_paper.py`.

## 5. Proposed LaTeX

Floats follow FILL RULE 6 (`[tp]` for double-column, `[!htbp]` for single-column). Captions carry no result number; `\Description`
is the ACM alt text (acmart warns without it). `main.tex` needs `\usepackage{tikz}` and `\usetikzlibrary{arrows.meta,positioning}`
for F4 only.

F1, after the paragraph that opens "What the confidence tracks":

```latex
% figure-tracks
\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{figures/tracks.pdf}
\Description{One panel per rated dataset and backbone. Rows: UAUC of the zero-shot confidence, of the LoRA confidence with its three seeds, and of the item mean, the information-matched item mean and matrix factorisation on the same rows, each with its interval.}
\caption{UAUC of the confidence $\ell$ beside the non-LLM references on the same TEST rows (Qwen3-8B; Llama for ML-1M and Toys). Zero-shot (circle) and LoRA (diamond: seed mean, seeds as small dots) against the item mean $m$, the matched item mean $m_T^{\dagger}$ (no interval stored) and temporal MF (shaded). Bars: 95\% user-bootstrap intervals from the result files; hollow: exploratory (addendum 6, ML-1M) or descriptive; ``not run yet'': no result file. The axis does not start at zero. Descriptive only: no test or admission rule is read off the plot (Table~\ref{tab:tracks}A).}
\label{fig:tracks}
\end{figure*}
```

F2, directly after F1 (or beside Table `tab:tracks`B):

```latex
% figure-shares
\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{figures/shares.pdf}
\Description{Two rows of small plots, one column per rated dataset. Top: shares of the within-user variance of the confidence logit that the item prior explains, that it leaves, and that the pair-specific residual carries, for zero-shot and LoRA. Bottom: the information gain of the residual, pooled and within-user, beside the gain of a matrix-factorisation residual.}
\caption{What the confidence logit $\ell$ is made of (top) and what it adds (bottom), Qwen3-8B. Top: shares of the within-user variance of $\ell$ that the item prior explains ($\rho^2/r_8$), that it leaves (non-prior) and that the residual $e$ carries ($e$-share$^{\dagger}$); n/i: the report marks the share uninterpretable and the value is withheld. Bottom: information gain $\mathcal G$ (E-D pooled; E-W$^{\dagger}$ within-user) beside the MF residual's $\mathcal G_{\rm CF}$. Circle: zero-shot; diamond: LoRA seed mean, seeds as dots; bars: 95\% user-bootstrap intervals; hollow: exploratory or descriptive. No Holm family or confirmation is implied (Table~\ref{tab:tracks}B).}
\label{fig:shares}
\end{figure*}
```

F3, after the paragraph "Static exposure and selective serving (S3)":

```latex
% figure-serving
\begin{figure*}[tp]
\centering
\includegraphics[width=\textwidth]{figures/serving.pdf}
\Description{One panel per next-item dataset. NDCG at ten among the served events against the fraction of events served, when the most confident events are served first and when a random subset is served; the two curves meet when every event is served.}
\caption{Selective serving on the next-item panels (Qwen3-8B zero-shot, $V_0$, \textsf{next}): NDCG@10 among the served events when only the most confident fraction by $p_{\max}$ is served, against a random subset (fixed seed); both curves meet when every event is served. Bands: 95\% event-bootstrap intervals of each curve; overlapping bands are not a test. Descriptive: the registered contrast and the admission rule are in Table~\ref{tab:anatomy}C and Section~\ref{sec:protocol}. ``Not run yet'': no audit file.}
\label{fig:serving}
\end{figure*}
```

F4, in Section "Setting and Notation", after the paragraph "Decomposition and audit quantities":

```latex
% figure-decomp
\begin{figure}[!htbp]
\centering
\input{figures/decomp}
\Description{Flow diagram: a yes/no prompt for a user-item pair goes through the language model to the confidence logit, which splits into a user offset, an item prior and a pair-specific residual; a question box asks what the residual adds to an item mean and to matrix factorisation.}
\caption{Decomposition of the Yes/No confidence logit of a pair $(u,i)$ into a user offset, an item prior (estimated by the swap prior from $K$ donor histories) and a pair-specific residual, and the question put to $e$ (Section~\ref{sec:prelim}); schematic, no data.}
\label{fig:decomp}
\end{figure}
```

## 6. Page cost in the sigconf layout

PAGE_COST_PLACEHOLDER

## 7. What has to change outside this directory (described, not edited)

1. `scripts/sigir/fill_paper.py`, `run()`: the filled copy is `main.tex`, `references.bib`, `sections/*.tex` and the three JSON files;
   nothing under `Paper/sigir2027/figures/` is copied into `filled/`. `\includegraphics{figures/tracks.pdf}` and `\input{figures/decomp}`
   would fail there ("File not found"). Proposed: in `run()`, add every `paper/figures/*.pdf` and `paper/figures/*.tex` to `outputs`
   (bytes, under `figures/<name>`); the existing loop already rewrites a file only when its bytes change. Do not copy
   `figures_manifest.json` or this file (they are not part of the build). `scripts/sigir/page_budget.py` calls `fill.run` into
   `<out>/filled` and compiles there, so it needs no change once `run()` copies the figures; it already records the end of every
   `figure` and `figure*` (`env/figure/end`, `env/figure*/end`).
2. `Paper/sigir2027/main.tex`: add `\usepackage{tikz}` and `\usetikzlibrary{arrows.meta,positioning}` after `\usepackage{subcaption}`
   (only for F4; acmart does not load TikZ).
3. `sections/experiments.tex`, `sections/preliminaries.tex`: insert the snippets of section 5 (and the sentences that cite
   `Figure~\ref{fig:tracks}`, `\ref{fig:shares}`, `\ref{fig:serving}`, `\ref{fig:decomp}`). The skeleton cites none today.
4. Page budget: the three plots and the schematic are new floats on top of a paper that is already near the nine-page limit (section 6):
   something moves to the artefact or into the figures (for example block A and B of Table `tab:tracks` become the figures'
   source and keep only the rows the text cites).
5. Freshness: a figure is stale as soon as one of its source files changes. The manifest lists the sha1 of each source; a check in
   `fill_paper.checks()` (or a test) that compares them with `UNFILLED.json["sources"]` / `MANIFEST.json` would catch a figure that
   outlived its inputs. Not implemented here.
6. `tests/test_confrec_fillpaper.py` asserts the skeleton is byte-identical after a run; it is unaffected by copying figures, but any test
   that lists the files `run()` writes needs the new entries.

## 8. Tests

`tests/test_confrec_figures.py` (CPU only, Agg): manifest values equal the files' values and the sha1s equal the files' hashes on a
synthetic tree with the real schemas; the drawn marks equal the manifest; a missing panel is a placeholder with a `not_run` entry; a
malformed file raises an error naming file and key; exploratory and descriptive flags, withheld values and the Gate-FT rule; no read
outside the results root (a decoy outside it is never opened); byte-identical output in one process and across processes with
different hash seeds; PDF is vector, TrueType, exact size, text inside the canvas; no result-like literal in the script; no result
number in the captions above; `decomp.tex` compiles in the paper class (when LaTeX is installed); and a round trip on a snapshot of
the real result tree.
