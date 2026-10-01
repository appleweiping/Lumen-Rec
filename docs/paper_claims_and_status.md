# Paper Claims and Experiment Status

This file freezes the paper scope so the project does not drift into a mixed
system log.

> **REFRAME NOTICE (authoritative).** This file is the designated source of
> truth for the frozen claim. After the major reframe (branch
> `paper/reframe-major-revision`, 8-domain commit `932d393`) the headline is the
> **pointwise relevance posterior**, and the calibrated-uncertainty/risk
> decomposition is a **characterized negative result**, not a positive
> contribution. Any earlier wording in this repo that says "uncertainty improves
> ranking/reliability" is **stale and reversed** -- do not reuse it. The frozen
> claim below replaces it.

## Working title

A Pointwise LLM Relevance Posterior Is a Strong Reranker (a controlled
same-candidate study of uncertainty-adjusted LLM recommendation). The
submission metadata config must match this reframed title unless the manuscript
title itself changes. No version of the title may be read as a full-catalog or
generic recommender SOTA claim.

## Primary claim (frozen, dual framing)

The frozen claim has **two parts**, exactly as the reframed paper states them:

1. **Positive result -- the pointwise posterior is the working ranking signal.**
   A task-grounded **pointwise LLM relevance posterior** (the model's own
   per-candidate relevance probability, used directly as the ranking score)
   ranks **first in 6 of 8 Amazon domains** (Books, Electronics, Sports, Toys,
   Home, Tools), improving NDCG@10 over the strongest official baseline by
   **+21.6% to +53.2%**. It is **competitive but not first** in the remaining
   two domains: **rank 2 on Beauty** (-11.0% NDCG@10 behind ProEx, 0.1341 vs.
   0.1506) and **rank 5 on Movies** (-24.2% behind LLMEmb, 0.1281 vs. 0.1690),
   reported honestly rather than dropped. On the four new domains with full
   per-event signal rows (Sports/Toys/Home/Tools), all 56 per-domain paired
   tests are positive and Holm-significant.

2. **Negative result -- the calibrated-uncertainty / risk decomposition does
   NOT improve ranking.** The principled uncertainty decomposition (boundary
   ambiguity, calibration gap, evidence support, counterevidence) and the
   one-parameter risk-adjusted family `r = p * (1 - U)^eta` do **not** improve
   same-candidate ranking over the bare posterior: `eta = 0` is test-best in all
   four diagnostic domains, a confidence-only variant matches or exceeds the
   full family in 3 of 4, boundary uncertainty is exactly inert (4/4), and
   ranking on the raw `p-hat` tracks the reported configuration within <= 0.008.
   This is reported as a **characterized negative result**.

This is a scoped **same-candidate reranking** claim, not a full-catalog
recommender SOTA claim, and **not** an ECE / calibration-guarantee claim. The
method name expands to **Candidate-Conditioned Relevance Posterior** (not
"Calibrated"). The contribution is not a new model backbone and not a generic
recommender system.

## Primary contribution statements

1. Protocol: a unified same-candidate reranking benchmark (10,000 users / 101
   candidates per domain [Beauty 973], shared Qwen3-8B backbone, shared
   importer, paired Holm-corrected bootstrap, full provenance) over **8 official
   -code-level baselines** and **8 Amazon domains**.
2. Positive finding: the **task-grounded pointwise relevance posterior** is the
   strongest single reranking signal, ranking first in 6 of 8 domains (see the
   frozen claim above). The win is attributed to the posterior itself by a
   raw-`p-hat` attribution ablation, not to the uncertainty machinery.
3. Negative finding: the calibrated-uncertainty / risk decomposition, evaluated
   as leave-one-component-out ablations and a raw-probability attribution, does
   **not** improve same-candidate ranking. The event-level uncertainty signal
   does stratify reliability, but this is reported as a descriptive,
   partly-circular observation, **not** a ranking-improvement claim.

C-CRP is the main internal method line. SRPD is the trainable
framework/ablation line and becomes paper-facing only after leakage-clean
teacher generation, weighted-loss training when claimed, exact same-candidate
score export, and paired-test gates pass.

## Paper-critical readiness modules

Official-baseline completion is necessary but no longer sufficient for paper
readiness. Before final writing or GPT-5.5/Codex xhigh review, the paper must
also include:

**2026-10-01 ProMax page range closed.** Crossref now exposes the final ACM
record for DOI `10.1145/3805712.3809600` with pages `2431--2441`; the ProMax
entry in `Paper/references.bib` now carries that page range. The live probe
`outputs/summary/paper_critical/promax_public_metadata_probe_20261001.{json,md}`
closes `promax:final_page_range_missing_in_bib` and
`promax:crossref_registry_not_visible` (Crossref 200). The DOI is registered
(doi.org handle API, timestamp 2026-07-10; doi.org 302 -> dl.acm.org), but the
probe's followed-redirect check still sees ACM DL bot-blocking (403), so
`promax:doi_resolver_not_visible` stays open and
`ProMax public metadata ready=false` under the unchanged evidence standard.
Target formatting and private manual submission confirmation remain open;
`final_submission_ready=false`.

**2026-06-15 private/manual gate stamp-aware default refresh.** Codex repaired
the date-sensitive defaults in
`scripts/audit/main_build_manual_submission_private_confirmation_request_packet.py`,
`scripts/audit/main_validate_manual_submission_private_confirmation_json.py`,
`scripts/audit/main_audit_final_blocker_consistency.py`, and
`scripts/audit/main_audit_final_blocker_doc_status.py`. These tools now use
the supplied `--stamp YYYYMMDD` to derive their same-date input artifacts when
explicit paths are omitted, instead of silently falling back to stale 20260613
manual/final-blocker files. Regression tests cover the stamp-derived manual
checklist, private-confirmation JSON/request packet, final-blocker consistency
inputs, and doc-status consistency input. Codex reran
`outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260615.{json,md}`
and
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260615.{json,md}`
plus
`outputs/summary/paper_critical/final_blocker_doc_status_audit_20260615.{json,md}`
with the repaired defaults. This is an evidence-handoff freshness fix only:
manual submission remains open, explicit Claude Opus coverage remains missing,
ProMax public metadata remains blocked, target formatting remains blocked,
doc-status is `ok=true`, and `final_submission_ready=false`.

**2026-06-15 final-blocker consistency semantics refresh.** Codex updated
`scripts/audit/main_audit_final_blocker_consistency.py` so the audit distinguishes
between a consistent blocked state and a packet inconsistency. The refreshed
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260615.{json,md}`
is now `ok=true` and `final_blocker_consistency_ok=true` while preserving
`final_submission_ready=false`. It explicitly accepts the current
`blocking_status=local_artifact_repair_required` as consistent with the open
target-formatting, ProMax metadata, manual confirmation, and Claude Opus
coverage blockers. This is an evidence-hygiene fix only, not a final-readiness
claim.

**2026-06-15 private manual-confirmation request refresh.** Codex refreshed the
public-safe handoff for the private submission-system confirmation:
`outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260615.{json,md}`.
The packet is `ok=true`, `request_packet_ready=true`,
`manual_confirmation_needed=true`, `manual_submission_system_ready=false`, and
`final_submission_ready=false`. It recommends the ignored path
`artifacts/private/manual_submission_private_confirmation_20260615.json`,
requires the current source-manifest hash
`2acac6e54318be410e9e216429195cad580fd870b91ef95a6bddb9f361909a08`, and only
allows public-safe confirmation fields: booleans, `source_manifest_sha256`, and
completed item IDs. It explicitly forbids author identities, affiliations,
conflicts, reviewer preferences, declarations, account metadata, or private
payloads in git. The request packet remains a handoff artifact only:
`confirm_anonymous_shell` and `confirm_external_proceedings_metadata` are still
blocked by target formatting and ProMax metadata, and the manual submission gate
does not close until a validated untracked private JSON is consumed by the
manual checklist.

**2026-06-15 ProMax public proceedings and final-stack refresh.** Codex reran
the live ProMax public metadata probe and the external proceedings metadata
audit:
`outputs/summary/paper_critical/promax_public_metadata_probe_20260615.{json,md}`
and
`outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260615.{json,md}`.
Both audits keep final readiness blocked. ProMax still has no final ACM page
range in BibTeX, Crossref `/works/10.1145/3805712.3809600` returns `404`, and
the DOI resolver returns `404`; ACM DL is inaccessible with `403`. Public source
checks still support accepted-paper provenance (arXiv HTML ACM metadata, SIGIR
2026 accepted papers, UQ author/publication pages, and UQ Experts), but those
sources do not close the exact page-range/Crossref/DOI gates. Codex refreshed
the full local final-stack handoff at
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260615.{json,md}`,
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260615.{json,md}`,
and
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260615.{json,md}`.
The stack remains `ok=false`, `local_release_candidate_ready=false`,
`blocking_status=local_artifact_repair_required`, and
`final_submission_ready=false`. Remaining blocker classes are target formatting
(`15 > 9` pages and 8 overfull hbox warnings), ProMax public metadata, private
manual submission-system confirmation, and explicit Claude Opus review coverage.

**2026-06-15 Claude Opus coverage gate refresh.** Codex repaired the Claude Opus
request path so it no longer carries the stale positive "uncertainty improves
ranking" claim as the live claim: the generated prompt now uses the current dual
framing, where the pointwise posterior is the positive result and
uncertainty/risk is a characterized negative result. The refreshed
`outputs/summary/paper_critical/claude_opus_review_request_packet_20260615.{json,md}`
is `ok=true`, `claude_review_needed=true`, has no hard failures, and is a
request artifact only. Two additional direct `mcp__claude_review.review`
attempts with `model=opus` and tools disabled failed:
`claude_opus_review_attempt_fourteenth_20260615.json` records the full-packet
prompt failure (`The command line is too long`), and
`claude_opus_review_attempt_fifteenth_20260615.json` records the compressed
current-claim prompt failure (`Claude CLI did not return JSON output`). The
refreshed connector-health packet
`outputs/summary/paper_critical/claude_review_connector_health_20260615.{json,md}`
records failed Claude attempts `15`, reports `failed_attempt_count=15`,
`valid_review_evidence_count=0`, `connector_unhealthy=true`,
`same_route_retry_recommended=false`, and
`recommended_next_route=external_claude_opus_json_via_request_packet_and_validator`.
This does **not** close explicit Claude Opus coverage or final readiness:
`review_continuation_packet_20260615.{json,md}` still records
`explicit_claude_opus_present=false`, `final_panel_coverage_complete=false`, and
`final_submission_ready=false`.

**2026-06-14 inline framework figure and package-polish smoke audit.** The
current branch `paper/reframe-major-revision` now contains an inline TikZ
framework overview at Figure~`fig:framework` (upstream commit `ee98fec`),
replacing the stale raster figure path. Codex refreshed the local package
polish around that change: `Paper/references.bib` now produces
`Paper/main.blg` with `warning$ -- 0` after
`pdflatex -> bibtex -> pdflatex -> pdflatex`, and
`scripts/audit/main_audit_submission_package.py` now treats either
external `framework_overview.pdf/svg` assets or the inline TikZ figure as a
valid framework-overview source. The smoke audit
`outputs/summary/paper_critical/submission_package_tikz_smoke_20260614.{json,md}`
reports `framework_overview.accepted_mode=inline_tikz`,
`bibtex_warning_count=0`, `page_count=15`, `overfull_hbox_count=8`,
`ok=false`, and `final_submission_ready=false`. This closes the BibTeX and
false missing-framework-asset blockers for the current manuscript shape; it
does **not** close the target-formatting blocker because the paper still
exceeds the 9-page profile and has overfull hbox warnings.

**2026-06-14 live ProMax proceedings and final-stack refresh.** Codex reran the
read-only live ProMax public metadata probe and external proceedings metadata
audit:
`outputs/summary/paper_critical/promax_public_metadata_probe_20260614.{json,md}`
and
`outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260614.{json,md}`.
Both audits keep final readiness blocked. ProMax still has no final ACM page
range in BibTeX, Crossref `/works/10.1145/3805712.3809600` returns `404`, and
the DOI resolver returns `404`; ACM DL is inaccessible with `403`. The public
source probes continue to support accepted-paper provenance (arXiv HTML ACM
metadata, SIGIR accepted papers, UQ profile/publication pages), but those
sources do not close the exact page-range/Crossref/DOI gates. Codex refreshed
the local release-candidate stack and closure packet at
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260614.{json,md}`
and
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260614.{json,md}`.
The refreshed stack remains `ok=false`, `local_release_candidate_ready=false`,
and `final_submission_ready=false`. Codex also updated
`configs/paper_submission_metadata.json` to match the current reframed
manuscript title, so the stale title-mismatch failure is gone; remaining
failures are target formatting (`15 > 9` pages and 8 overfull hbox warnings),
ProMax public proceedings metadata, manual submission-system confirmation, and
explicit Claude Opus review coverage. The associated consistency audit
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260614.{json,md}`
is `ok=false` because the release stack now has
`blocking_status=local_artifact_repair_required`; this is stricter and more
accurate than the previous local-ready handoff, not a final-readiness closure.

**2026-06-12 external metadata discovery enhancement.** Codex strengthened
`scripts/audit/main_audit_external_proceedings_metadata.py` with advisory
Crossref title discovery for ProEx/ProMax. The discovery layer records candidate
DOI/title/container/page/year/URL/type/score information and is visible in the
generated Markdown report, but it remains separate from the exact final gates:
BibTeX page range, direct Crossref `/works/{doi}`, DOI resolver, and configured
required source checks. The refreshed live audit
`outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260612.{json,md}`
still reports `external_proceedings_metadata_ready=false` and
`final_submission_ready=false`. ProMax discovery currently finds `5` Crossref
title-search candidates, but `0` candidates matching the expected DOI
`10.1145/3805712.3809600`; therefore the ProMax page-range, Crossref, and DOI
resolver blockers remain final-blocking rather than being silently replaced by
fuzzy search evidence. Codex then added a required ProMax arXiv HTML
ACM-metadata source check and updated the local ProMax BibTeX entry with ISBN
`979-8-4007-2599-9` and location `Melbourne, VIC, Australia`. The refreshed
audit records source check `arxiv_html_promax_acm_metadata` as passing, and the
external metadata report now exposes BibTeX `isbn` and `location` fields. This
strengthens the provenance for the expected DOI/ISBN/venue metadata, but it is
still not final proceedings metadata: ProMax has no final page range in BibTeX,
Crossref `/works/10.1145/3805712.3809600` remains 404, and the DOI resolver
remains 404.

**2026-06-12 final blocker closure packet.** Codex added
`scripts/audit/main_build_final_submission_blocker_closure_packet.py` to turn
the remaining final-readiness blockers into a single actionable handoff. The
generated
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260612.{json,md}`
reports `closure_packet_ready=true`, `local_release_candidate_ready=true`,
`ready_for_human_handoff=true`, and `final_submission_ready=false`. It
classifies the state as local artifact handoff ready, external proceedings
metadata blocked, and manual submission system private-pending. The packet
records the current ProMax DOI/ISBN/location/source-check evidence, exact
public metadata closure conditions, source-manifest sha256
`795aa65895cf06452a3cf48b6519bece473cd61fc2ee33c2becb7c2d8b2e79cc`, and the
manual item IDs needed in a future untracked private confirmation file. It
does not store private author/COI/reviewer/account data and does not close
final submission readiness.

**2026-06-12 ProMax public metadata probe.** Codex added
`scripts/audit/main_probe_promax_public_metadata.py` as a lightweight live
monitor for the public ProMax metadata blocker. The generated
`outputs/summary/paper_critical/promax_public_metadata_probe_20260612.{json,md}`
reports `promax_public_metadata_ready=false`, `final_submission_ready=false`,
Crossref status `404`, DOI resolver status `404`, ACM DL status `403`, and
passing arXiv HTML / SIGIR accepted-paper source probes. The probe is now
extended with a UQ author-profile announcement source check for the accepted
ProMax title. These source checks strengthen accepted-paper provenance, but the
probe still cannot by itself mark the paper final-submission-ready.

**2026-06-13 ProMax public metadata recheck.** The latest live probe
`outputs/summary/paper_critical/promax_public_metadata_probe_20260613.{json,md}`
still reports `promax_public_metadata_ready=false`: Crossref `/works` for
`10.1145/3805712.3809600` returns `404`, the DOI resolver returns `404`, and
ACM DL returns `403`; arXiv HTML ACM metadata, the SIGIR accepted-paper page,
the UQ author-profile source probe, the author Google Sites publications
source probe, and the UQ Experts profile source probe remain passing.
Follow-up live probes at `2026-06-13T00:49:05Z`,
`2026-06-13T01:59:36Z`, `2026-06-13T02:32:01Z`,
`2026-06-13T03:16:11Z`, `2026-06-13T04:11:37Z`,
`2026-06-13T04:30:39Z`, `2026-06-13T04:49:27Z`,
`2026-06-13T05:11:34Z`, `2026-06-13T05:31:00Z`,
`2026-06-13T05:42:19Z`, `2026-06-13T06:04:37Z`,
`2026-06-13T06:52:21Z`, and `2026-06-13T07:37:42Z` found the same direct
blocker state: Crossref `404`,
DOI resolver `404`, ACM DL `403`, and all `5/5` public source probes passing.
The closure packet was refreshed after the latest probe at
`2026-06-13T07:38:58Z`; this is stronger public accepted-paper provenance, not
a readiness upgrade. The closure packet Markdown now lists latest public source
probes in addition to direct Crossref/DOI/ACM statuses and explicitly carries
the review-panel blockers.
Codex also refreshed the public-safe private manual submission confirmation
request packet at `2026-06-13T08:10:35Z` and reran the closure/consistency
handoff at `2026-06-13T08:11Z`; the private manual gate remains open with
`manual_submission_system_ready=false`, while the consistency audit keeps the
validator-before-checklist route intact.
Codex also refreshed
the complete local release-candidate stack
as
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260613.{json,md}`;
the latest `2026-06-13T04:31:25Z` stack reports `ok=true`,
`local_release_candidate_ready=true`, `refresh_artifact_fresh=true`,
`blocking_status=external_manual_or_review_blocked`, `failures=[]`, and
`final_submission_ready=false`. Its freshness audit checks `23` input
fingerprints and `14` generated gate files with zero mismatches; the source
package has `21` files with manifest sha256
`91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424`, and the
independent rebuild produces a `9`-page PDF with zero BibTeX and overfull hbox
warnings. The refreshed closure packet
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`
now uses the 2026-06-13 final gate, external metadata audit, manual checklist,
release-candidate stack, and latest probe while preserving
`final_submission_ready=false`.

**2026-06-13 review-continuation packet.** A fresh GPT-5.5 xhigh reviewer pass
after the handoff refresh rated the paper `8.0/10` with verdict
`CONDITIONAL_PASS`, table eligibility limited to the scoped controlled
same-candidate reranking claim, and `final_submission_ready_claim_allowed=false`.
Codex applied the actionable non-experimental fixes: the abstract now uses
`\method{} ranks` to avoid the PDF-visible `C-CRPranks` spacing issue; the
post-section-review claim-audit CSV compatibility path
`outputs/summary/paper_critical/final_paper_claim_audit_post_section_review_20260612.csv`
exists; and the current citation spot-check was refreshed as
`outputs/summary/paper_critical/final_citation_spot_check_20260613.{json,md}`
with `cited_key_count=21`, `bibliography_entry_count=21`,
`missing_in_bib=[]`, `uncited_in_bib=[]`, `bibtex_warning_count=0`, and
`must_add_count=0`. After the source change, Codex reran the LaTeX/BibTeX
chain and the complete 2026-06-13 release-candidate stack before rebuilding
the final blocker closure packet. The review-continuation handoff
`outputs/summary/paper_critical/review_continuation_packet_20260613.{json,md}`
reports `ok=true`, `review_continuation_ready=true`, score floor `8.0`,
`local_release_candidate_ready=true`, `ready_for_human_handoff=true`, and
`final_submission_ready=false`. It also keeps
`final_panel_coverage_complete=false` because explicit Claude Opus reviewer
output has not yet been captured. The attempted Claude Opus review job failed
with `Claude CLI did not return JSON output`; the failure record is
`outputs/summary/paper_critical/claude_opus_review_attempt_20260613.json` and
is intentionally not counted as valid reviewer evidence. A second retry and a
third tool-discovered retry failed with the same error and are recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_retry_20260613.json`
and
`outputs/summary/paper_critical/claude_opus_review_attempt_third_20260613.json`.
The review-continuation packet builder now records failed reviewer attempts
separately from valid reviewer JSONs. A fourth synchronous no-tools Claude
review call failed with the same `Claude CLI did not return JSON output` error
and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_sync_notools_20260613.json`,
and a fifth minimal JSON-oriented no-tools call failed the same way at the
Claude CLI/connector layer. The fifth failure is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_minimal_json_20260613.json`,
so the packet at that point exposed five failed Claude attempts while preserving
`explicit_claude_opus_present=false`. Codex added
`scripts/audit/main_build_claude_review_request_packet.py` and generated
`outputs/summary/paper_critical/claude_opus_review_request_packet_20260613.{json,md}`
as a public-safe prompt/schema handoff for the next valid Claude Opus review.
After the ninth failed connector attempt, the request packet was refreshed at
`2026-06-13T03:28:45Z` and, at that point, recorded nine failed Claude
attempts. The tenth-attempt refresh below supersedes that count. The
request packet is not reviewer coverage and does not close final
readiness. The review-continuation packet builder now validates additional
reviewer JSONs before counting them for panel coverage: a Claude/Opus review
must explicitly identify a Claude Opus reviewer and be valid substantive
evidence with `valid_review_evidence=true`,
`claim_boundary_ok=true`, `final_submission_ready_claim_allowed=false`, a
kill-argument, major concerns, required changes, and acknowledged remaining
blockers. This prevents an empty Claude name+score JSON from closing the
explicit Claude Opus review blocker.
Codex then retried a sixth synchronous JSON-only Claude call with `model=opus`
and tools disabled; it failed with the same connector-layer error and is
recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_sixth_20260613.json`.
Codex then retried a seventh synchronous JSON-only Claude call with `model=opus`,
tools disabled, and a shorter structured JSON-only prompt; it failed with the
same connector-layer error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_seventh_20260613.json`.
Codex then retried an eighth asynchronous Claude call with
`mcp__claude_review.review_start` plus `review_status`, `model=opus`, tools
disabled, and a short JSON-only prompt; the job
`b1b88420168a4e498029a00a8695098a` failed with the same connector-layer error
and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_eighth_20260613.json`.
At that point, the refreshed review-continuation packet reported eight
failed attempts while keeping `explicit_claude_opus_present=false`.
Codex then retried a ninth asynchronous Claude call through
`mcp__claude_review.review_start`; job
`a3863723466147e9b9b849cf994ca8fd` failed with the same connector-layer
`Claude CLI did not return JSON output` error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_ninth_20260613.json`.
Codex then retried a tenth asynchronous Claude Opus call through
`mcp__claude_review.review_start`; job
`b6e19654680c457d8be4845e168ce251` failed with the same connector-layer
`Claude CLI did not return JSON output` error and is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_tenth_20260613.json`.
At that historical tenth-attempt point, the refreshed review-continuation packet
and Claude request packet recorded ten failed Claude attempts, still with
`explicit_claude_opus_present=false`, `final_panel_coverage_complete=false`,
and `final_submission_ready=false`.
Codex later tightened this import path further: explicit Claude Opus coverage
now requires the reviewer identity to contain both `claude` and `opus`, and
`scripts/audit/main_validate_claude_opus_review_json.py` provides a read-only
preflight for future external Claude JSONs. The validator checks the expected
schema, explicit Claude Opus identity, substantive evidence fields, scoped
claim boundary, `final_submission_ready_claim_allowed=false` while blockers
remain, and `score_0_to_10 >= 8.0` before the review is attached through
`--additional-review-json`. The refreshed
`claude_opus_review_request_packet_20260613.{json,md}` now includes the
validation command before the review-continuation command, and the closure
packet's review-panel next commands mirror that order.
Codex then retried an eleventh asynchronous Claude Opus call through
`mcp__claude_review.review_start`; job
`bf4b6b8145404ffa881cd99ed3c73429` used `model=opus`, tools disabled, and a
short schema-aligned ARIS hostile-review prompt, but again failed with
`Claude CLI did not return JSON output`. The failure is recorded at
`outputs/summary/paper_critical/claude_opus_review_attempt_eleventh_20260613.json`.
That eleventh-attempt refresh still left
`explicit_claude_opus_present=false`,
`final_panel_coverage_complete=false`, and `final_submission_ready=false`.
Codex then added
`scripts/audit/main_audit_claude_review_connector_health.py` as a local
read-only health audit for the repeated Claude connector failures. The current
`outputs/summary/paper_critical/claude_review_connector_health_20260613.{json,md}`
now reports failed attempt count `13`, valid review evidence count `0`,
`same_error_tail_streak=1`, `connector_unhealthy=false`,
`same_route_retry_recommended=true`, and recommends
`retry_connector_or_refresh_request_packet`. This does not close the explicit
Claude Opus blocker. The changed health verdict is diagnostic: the twelfth
direct `mcp__claude_review.review` attempt exposed a prompt shell-escaping
failure, and the thirteenth safe-prompt retry returned to the prior
`Claude CLI did not return JSON output` failure mode.
Codex then hardened the external JSON intake path: the refreshed
`claude_opus_review_request_packet_20260613.{json,md}` now carries a fillable
`response_template` with `valid_review_evidence=false` by default, while
`scripts/audit/main_build_review_continuation_packet.py` and
`scripts/audit/main_validate_claude_opus_review_json.py` require a current
Claude Opus review JSON to acknowledge both the open ProMax public metadata
blocker and the private manual submission-system blocker before it can count as
explicit Claude Opus coverage. The refreshed review-continuation packet exposes
`required_claude_blocker_ack_groups=["manual_submission_system",
"promax_public_metadata"]`. At that point, it records failed Claude attempts
`13` and keeps `final_submission_ready=false`.
The final-blocker consistency audit is now schema
`2026-06-13.final_blocker_consistency_audit.v3` and explicitly guards this
handoff path plus the private manual-confirmation path: it fails if the Claude
request packet loses the response template, if the template defaults
`valid_review_evidence` to true, if ProMax/manual blocker-acknowledgement rules
disappear, if the current review-continuation packet stops exposing the
required Claude blocker acknowledgement groups, or if the manual request and
closure packets stop routing a private confirmation JSON through
`main_validate_manual_submission_private_confirmation_json` before the public
manual checklist consumes it.

**2026-06-13 final submission gate review-coverage hardening.** Codex updated
`scripts/audit/main_build_final_submission_gate.py` so the final local
submission gate directly consumes the review-continuation packet instead of
leaving Claude Opus coverage only in a side handoff. The refreshed
`outputs/summary/paper_critical/final_submission_gate_20260613.{json,md}`
reports `ok=true`, `all_local_artifact_gates_ok=true`,
`review_continuation_ready=true`, `review_panel_coverage_complete=false`, and
`final_submission_ready=false` with verdict
`LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED`. The
release-candidate packet and stack now expose
`blocking_status=external_manual_or_review_blocked`, and
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`
adds a separate `review_panel_coverage` closure group containing
`review_panel_coverage_not_complete` and `explicit_claude_opus_review`. This is
a guardrail hardening, not a readiness upgrade: the paper remains locally
handoff-ready but not final-submission-ready.
After an independent GPT-5.5 xhigh sidecar audit, Codex fixed one stale-input
risk in `scripts/audit/main_refresh_pre_submission_gates.py`: the
pre-submission refresh now fingerprints both the review-continuation packet and
`scripts/audit/main_build_review_continuation_packet.py`, because the final gate
reads that packet. The refreshed
`outputs/summary/paper_critical/pre_submission_gate_refresh_20260613.json`
records those fingerprints and
`outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_20260613.json`
reports `refresh_artifact_fresh=true`. Codex also updated
`scripts/audit/main_build_review_continuation_packet.py` so a future state where
the ProMax probe, closure packet, or release-candidate stack has legitimately
closed no longer causes a review-continuation failure; the final submission
decision remains delegated to the final gate.
Codex later fixed a release-stack artifact hygiene bug in the same gate family:
recursive warning prefixes from nested artifacts were growing across refreshes.
The builders now normalize known aggregator prefixes before adding the current
layer prefix in review-continuation, final-gate, pre-submission-refresh,
release-candidate, stack, and closure packet outputs. The refreshed artifacts
have compact warning lists with no detected recursive warning chains. This is
an evidence-hygiene fix only; the final gate remains blocked by ProMax public
metadata, private manual submission-system confirmation, and explicit Claude
Opus review coverage.
Codex later fixed a stale-stamp footgun in
`scripts/audit/main_build_final_submission_blocker_closure_packet.py`: when
`--stamp` is omitted, the script now infers the input stamp from a dated
`--output-json` or `--output-md` path. This prevents a 20260613 closure packet
from silently reading 20260612 gate inputs. The regenerated 20260613 closure
packet now uses the 20260613 final gate, release stack, manual checklist,
external metadata audit, and ProMax probe; its review-panel closure group
explicitly lists `review_panel_coverage_not_complete` and
`explicit_claude_opus_review`, and total remaining blocker count is `13`.
Codex later fixed the matching ProMax-probe handoff guard: the closure builder
now defaults to `promax_public_metadata_probe_<stamp>.json` when present, and
the final-blocker consistency audit fails if the closure packet omits that
probe or records direct statuses inconsistent with the standalone probe
artifact. The refreshed consistency audit now reports
`closure_promax_probe_provided=true`, closure/probe direct status `404/404/403`,
and `final_submission_ready=false`.
Codex then added
`scripts/audit/main_audit_final_blocker_consistency.py` and
`outputs/summary/paper_critical/final_blocker_consistency_audit_20260613.{json,md}`
as a cross-packet consistency check over the final gate, release stack, closure
packet, review-continuation packet, Claude request packet, ProMax probe, and
manual request packet. At that point, the audit reports `ok=true`, `final_blocker_consistency_ok=true`, and failed Claude attempts `13`,
`explicit_claude_opus_present=false`, ProMax public metadata still false with
Crossref `404` / DOI resolver `404` / ACM DL `403`, manual confirmation still
needed, recursive warning regressions `0`, and
`final_submission_ready=false`. This prevents stale handoff artifacts from
contradicting each other, but it is not a readiness upgrade.
Codex later upgraded this audit to schema
`2026-06-13.final_blocker_consistency_audit.v3` and added
`scripts/audit/main_validate_manual_submission_private_confirmation_json.py`
as the preflight for any future untracked private confirmation JSON. The
refreshed request packet and closure packet both list this validator before the
manual checklist command, and the consistency audit now records
`manual_request_has_private_confirmation_validator=true` and
`closure_manual_group_has_private_confirmation_validator=true`.
Codex then added
`scripts/audit/main_audit_final_blocker_doc_status.py`,
`tests/test_audit_final_blocker_doc_status.py`, and
`outputs/summary/paper_critical/final_blocker_doc_status_audit_20260613.{json,md}`
as the matching canonical-doc status guard. The audit reports `ok=true` and
`final_blocker_doc_status_ok=true` for the current sections of the active TODO,
claim/status doc, milestone README, and server runbook, while requiring the
latest failed-Claude count `11`, explicit Claude missing state, ProMax
`404/404/403` blocked state, private manual blocked state, recursive warning
regressions `0`, and `final_submission_ready=false`. It prevents historical
eight-attempt or two-blocker wording from reading like current truth.

**2026-06-12 submission release-candidate packet.** Codex added
`scripts/audit/main_build_submission_release_candidate_packet.py` as a local
handoff index over the final gate, refresh freshness audit, anonymous source
package, source-package rebuild, metadata packet, manual checklist, and
external proceedings metadata audit. The generated
`outputs/summary/paper_critical/submission_release_candidate_20260612.{json,md}`
reports `ok=true`, `readiness_scope=local_artifacts_only`,
`local_release_candidate_ready=true`, `blocking_status=external_or_manual_blocked`,
and `final_submission_ready=false`. The packet deliberately copies
`final_submission_ready` from
`outputs/summary/paper_critical/final_submission_gate_20260612.json` rather than
recomputing it. It verifies the dated gate inputs share stamp `20260612`, the
source manifest sha256 values agree across package/rebuild/metadata artifacts,
the staged source package contains `21` files / `652691` bytes, and the
independent rebuild produced a `9`-page PDF with zero BibTeX and overfull hbox
warnings. This is a local artifact release candidate only; final submission
remains blocked by ProMax final page-range / DOI-Crossref visibility and
private manual submission-system confirmation. In the refreshed 2026-06-13
packet this status is widened to `external_manual_or_review_blocked` because
explicit Claude Opus review coverage is also final-blocking.

**2026-06-12 submission release-candidate stack refresh.** Codex added
`scripts/audit/main_refresh_submission_release_candidate_stack.py` as the
preferred local handoff wrapper over the pre-submission refresh, freshness
audit, and release-candidate packet. The generated
`outputs/summary/paper_critical/submission_release_candidate_stack_refresh_20260612.{json,md}`
reports `ok=true`, `refresh_ok=true`, `freshness_ok=true`,
`refresh_artifact_fresh=true`, `release_candidate_ok=true`,
`local_release_candidate_ready=true`, `readiness_scope=local_artifacts_only`,
`blocking_status=external_or_manual_blocked`, and
`final_submission_ready=false`. The stack artifact has zero failures and keeps
the same nine remaining blockers instead of treating local consistency as
final submission approval. Future agents should use this wrapper for routine
status refreshes, then inspect sub-gates only when the stack fails. The
refreshed 2026-06-13 stack now reports
`blocking_status=external_manual_or_review_blocked`.

**2026-06-12 pre-submission gate refresh.** Codex added
`scripts/audit/main_refresh_pre_submission_gates.py` as the preferred
single-command refresh for the final submission gate stack. It runs the external
proceedings metadata audit, submission package audit, anonymous source-package
staging, source-package rebuild audit, submission metadata packet, manual
submission checklist, and final submission gate in dependency order, then writes
`outputs/summary/paper_critical/pre_submission_gate_refresh_20260612.{json,md}`.
The latest live refresh reports `ok=true`, `final_submission_ready=false`, and
final verdict `LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED`. This
improves reproducibility and prevents stale or out-of-order gate artifacts. The
refresh summary now also records the Git HEAD before refresh, tracked dirty
state, sha256 fingerprints for paper/source/config/evidence/script inputs, and
sha256/size for each generated gate JSON/MD. It still does not close final
readiness while ProMax public proceedings metadata and private manual
submission-system fields remain unresolved.
Codex then added
`scripts/audit/main_audit_pre_submission_refresh_freshness.py` as the companion
freshness gate for that refresh artifact. It checks the stored
`input_fingerprints` and every generated step JSON/MD sha256/size against the
current worktree, fails closed on mismatches or missing files, and keeps
`final_submission_ready=false` while external/manual blockers remain. The
recorded Git HEAD is interpreted as generation provenance, not as a strict
current-HEAD equality condition after committing generated artifacts.
The current freshness artifact
`outputs/summary/paper_critical/pre_submission_gate_refresh_freshness_20260612.{json,md}`
reports `ok=true`, `refresh_artifact_fresh=true`, `21` input fingerprints and
`14` generated gate files checked, zero hash mismatches, and
`final_submission_ready=false`. The refreshed external metadata audit now also
passes a required SIGIR 2026 accepted-papers source check for the ProMax title
and author line, strengthening the accepted-paper evidence while leaving final
page-range/DOI-registry blockers intact.

**2026-06-12 anonymous source-package staging.** Codex added
`scripts/audit/main_build_submission_source_package.py` to turn the audited
source manifest into a local anonymous source-package staging directory without
expanding the file set. The generated
`outputs/summary/paper_critical/submission_source_package_20260612.{json,md}`
reports `ok=true`, `submission_source_package_ready=true`,
`final_submission_ready=false`, `21` copied files, `652691` copied bytes, and a
copied manifest sha256 matching the package-audit source manifest sha256
`4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa`. The copied
tree is kept in ignored `artifacts/submission_source_package_20260612/files/`
rather than committed. The builder is fail-closed on non-OK package audits,
non-false `final_submission_ready`, nonempty audit failures, anonymous-source
leak-scan failure, external source references, invalid or duplicate manifest
entries, missing/changed source hashes, path escapes, private/manual/COI/
reviewer/account path fragments, and untracked/ignored files outside the
explicit generated-paper allowlist. This is local artifact staging only; it does
not close final submission readiness while ProMax metadata and manual
submission-system gates remain open.

**2026-06-12 anonymous source-package rebuild audit.** Codex added
`scripts/audit/main_audit_submission_source_package_rebuild.py` as the local
rebuildability gate for the staged anonymous source package. It verifies the
staged package file tree exactly matches the manifest, rejects hash/size drift,
copies the package into an independent ignored worktree, deletes stale LaTeX
build outputs, and runs `pdflatex -> bibtex -> pdflatex -> pdflatex`. The
generated
`outputs/summary/paper_critical/submission_source_package_rebuild_20260612.{json,md}`
reports `ok=true`, `submission_source_package_rebuild_ready=true`,
`final_submission_ready=false`, `21` verified package files, four successful
build commands, a rebuilt 9-page / 546669-byte `main.pdf`, actual PDF page
count `9`, `bibtex_warning_count=0`, and `overfull_hbox_count=0`. The final
submission gate now treats this rebuild audit as a required local artifact
gate. Passing it means only that the staged anonymous package rebuilds cleanly
in this local environment; it does not imply ProMax metadata readiness, manual
submission-system completion, or final submission readiness.

**2026-06-12 final submission gate.** Codex added
`scripts/audit/main_build_final_submission_gate.py` as the final local
pre-submission aggregator over the package audit, metadata packet,
source-package rebuild audit, external proceedings metadata audit, manual
submission checklist, and current review-continuation packet. The generated
`outputs/summary/paper_critical/final_submission_gate_20260612.{json,md}`
reports `ok=true`, `all_local_artifact_gates_ok=true`,
`external_proceedings_metadata_ready=false`,
`manual_submission_system_ready=false`, `final_submission_ready=false`, and
verdict `LOCAL_PACKAGE_READY_BUT_EXTERNAL_OR_MANUAL_BLOCKED`; the refreshed
2026-06-13 gate additionally requires review-panel coverage and now reports
`LOCAL_PACKAGE_READY_BUT_EXTERNAL_MANUAL_OR_REVIEW_BLOCKED`. This is the
current authoritative local submission-status summary: it confirms that the
repository-side artifacts are ready enough for target formatting, but final
submission is still blocked by ProMax final page-range/DOI registry visibility,
private manual submission-system fields that cannot be stored in git, and the
missing explicit Claude Opus review.

**2026-06-12 manual submission checklist packet.** Codex added
`scripts/audit/main_build_manual_submission_checklist.py` and
`configs/paper_manual_submission_checklist.json` to make the remaining
submission-system work auditable without storing private author/reviewer/COI
data in the repository. The generated packet
`outputs/summary/paper_critical/manual_submission_checklist_20260612.{json,md}`
reports `ok=true`, `manual_submission_checklist_ready=true`,
`manual_submission_system_ready=false`, and `final_submission_ready=false`.
It lists `14` manual actions, pre-fills only public metadata from the existing
submission metadata packet, records the source manifest sha256, and explicitly
keeps author identities, conflicts, reviewer preferences, declarations, and
private submission account metadata out of git. This reduces the manual blocker
from a prose reminder to a concrete checklist, but it cannot close final
readiness until a human completes the submission-system fields and the ProMax
external metadata blockers are resolved.
Codex also added
`configs/paper_manual_submission_private_confirmation.template.json` and
optional `--private-confirmation-json` support. A future untracked local
confirmation file can mark submission-system items completed while recording
only a file hash, source-manifest match, and completed item IDs; the checker
rejects forbidden private keys such as authors, affiliations, conflicts,
reviewer preferences, and account metadata. This gives the manual gate a
privacy-preserving closure path but does not change the current committed
status: `manual_submission_system_ready=false`.

**2026-06-13 manual private-confirmation request packet.** Codex added
`scripts/audit/main_build_manual_submission_private_confirmation_request_packet.py`
and generated
`outputs/summary/paper_critical/manual_submission_private_confirmation_request_packet_20260613.{json,md}`
as the public-safe handoff into the private submission-system step. The packet
reports `ok=true`, `request_packet_ready=true`,
`manual_confirmation_needed=true`, `manual_submission_system_ready=false`, and
`final_submission_ready=false`. It records the current source manifest sha256
`91d1d6495fe3fa85608d7711fb5873730d907237242b3b3fa489c6f1ed516424`, a safe
confirmation JSON skeleton under the ignored recommended path
`artifacts/private/manual_submission_private_confirmation_20260613.json`, all
manual item IDs needed for the full manual gate, currently blocked item
`confirm_external_proceedings_metadata`, rejected private JSON keys, and exact
follow-up commands for validating the private confirmation JSON before
rerunning the manual checklist, release-candidate stack, and final gate after a
human completes the submission-system fields. Codex also linked this request
packet from
`scripts/audit/main_build_final_submission_blocker_closure_packet.py` and
refreshed
`outputs/summary/paper_critical/final_submission_blocker_closure_packet_20260613.{json,md}`.
This request packet is not a private confirmation and does not waive the ProMax
public metadata blocker, private manual confirmation blocker, or explicit
Claude Opus review blocker.

**2026-06-13 manual private-confirmation validator.** Codex added
`scripts/audit/main_validate_manual_submission_private_confirmation_json.py`
and `tests/test_validate_manual_submission_private_confirmation_json.py` as a
local read-only preflight for any future untracked private confirmation JSON.
It reports only booleans, item IDs, key names, hashes, and path state; it
rejects forbidden private keys, schema/source-manifest/profile/checklist
mismatches, in-repo paths outside ignored `artifacts/private/`, unknown or
duplicate item IDs, missing current required item IDs, and attempts to mark a
currently blocked item such as `confirm_external_proceedings_metadata`
complete before ProMax public metadata closes. The refreshed request packet and
closure packet now place this validator before the manual checklist command,
and the v3 consistency audit fails if that order disappears. This is privacy
and handoff hardening only; `manual_submission_system_ready=false` and
`final_submission_ready=false` remain the live committed state.

**2026-06-12 external proceedings metadata recheck.** Codex added
`scripts/audit/main_audit_external_proceedings_metadata.py` and
`configs/paper_external_proceedings_metadata_checks.json` to turn the remaining
ProEx/ProMax citation-metadata caution into a reproducible ARIS
`aris-citation-audit` artifact. The generated live audit
`outputs/summary/paper_critical/external_proceedings_metadata_recheck_20260612.{json,md}`
reports `ok=true`, `external_proceedings_metadata_ready=false`, and
`final_submission_ready=false`. ProEx is supported by the local BibTeX page
range plus DBLP search API evidence for KDD 2026 pages `1940-1951` and DOI
`10.1145/3770854.3780284`; Crossref/DOI resolver `404` responses are retained
as warnings. ProMax remains final-blocking because the local BibTeX entry has
`numpages=11` but no final `pages` range, and live Crossref/DOI resolver checks
for `10.1145/3805712.3809600` return `404`. The external audit now verifies two
accepted-paper sources for ProMax: arXiv `2604.26231` confirms the SIGIR 2026
acceptance/11-page metadata, and the official SIGIR 2026 accepted-papers page
contains the ProMax title and author line. The submission-package and
metadata-packet audits now consume this artifact and carry the concrete
blockers:
`promax:final_page_range_missing_in_bib`,
`promax:crossref_registry_not_visible:status=404`,
`promax:doi_resolver_not_visible:status=404`, and the manual
submission-system checklist. This improves auditability but does not close
final submission readiness.

**2026-06-12 submission metadata packet.** Codex added
`scripts/audit/main_build_submission_metadata_packet.py` and
`configs/paper_submission_metadata.json` to prepare the submission-system fields
that can be safely stored in the anonymous repo. The generated packet
`outputs/summary/paper_critical/submission_metadata_packet_20260612.{json,md}`
reports `ok=true`, `submission_metadata_packet_ready=true`,
`final_submission_ready=false`, title
`Actionable Uncertainty for LLM-Based Recommendation`, a 212-word abstract,
five suggested keywords, four topic areas, and the same paper/source manifest
sha256 as the package audit. Author identities, conflicts, reviewer
suggestions/exclusions, and submission-system declarations remain manual-only
fields and are not stored in the repository. The packet reduces the final
manual submission checklist but does not close final readiness.

**2026-06-12 submission-package audit.** Codex added
`scripts/audit/main_audit_submission_package.py` as the local, read-only gate
between manuscript evidence readiness and target-conference formatting. The
current output
`outputs/summary/paper_critical/submission_package_audit_20260612.{json,md}`
reports `ok=true`, `submission_package_ready_for_target_formatting=true`,
`final_submission_ready=false`, and verdict
`READY_FOR_TARGET_FORMATTING_NOT_FINAL_SUBMISSION`. It verifies the paper source
closure, local framework figure files, anonymous ACM paper shell, 9-page
`Paper/main.pdf`, zero BibTeX warnings, no undefined citations/references, zero
overfull hbox warnings, final-panel score floor `8.0`, and the scoped
claim-audit state. It now includes a privacy-preserving anonymous source leak
scan over the TeX closure, failing closed on email addresses, ORCID identifiers,
acknowledgment/acks macros, absolute local paths, non-anonymous author macros,
or non-anonymous affiliation macros. Sensitive matches are reported only by
file/line/type and short sha256 token, not by raw private text. The current
audit reports `source_leak_scan_ok=true` and zero hits for all leak categories.
The audit also emits a source-package manifest with
`file_count=21`, `total_bytes=652691`, manifest sha256
`4f2a9856f722c98ffaf6b7073af27f6890c3086fffe23fa596ebe9fc62aa3cfa`, and no
external source references; it covers `main.tex`, included section/table
sources, `references.bib`, `main.bbl`, `main.pdf`, and the paper-local framework
figure. `configs/paper_submission_profiles.json` now defines the working
`sigir2026_full_paper_acm_anonymous` profile from official SIGIR 2026 full-paper
and ACM template sources; the current audit reports
`target_formatting_profile_ok=true` for the ACM `acmart`/`sigconf` anonymous
9-page source package. This advances the paper package through the local target
profile gate, but it does not close final submission readiness: ProMax final ACM
page range/Crossref visibility and the final manual submission-system
metadata/format checklist remain blocking conditions.

**2026-06-12 final PDF polish/metadata follow-up.** The latest ARIS
claim/citation follow-up is saved at
`outputs/summary/paper_critical/final_pdf_polish_metadata_followup_20260612.{json,md}`.
The server preflight found no matching running experiment and an idle GPU. The
current manuscript polish removes the remaining local overfull hbox warnings
(`overfull_hbox_count=0`) while preserving the audited claim boundary; only
underfull float/table/bibliography line-wrap warnings remain. `Paper/main.pdf`
is 546669 bytes and `Paper/main.blg` reports `warning$ -- 0`. ProEx proceedings
metadata is now resolved by a visible DBLP spot-check for KDD 2026 V.1, pages
1940-1951, DOI `10.1145/3770854.3780284`. ProMax remains a final-submission
metadata caution: arXiv `2604.26231` lists `11 pages, 8 figures, accepted by
SIGIR 2026`, and the current BibTeX uses ACM DOI `10.1145/3805712.3809600`, but
the final ACM page range/registry visibility must be rechecked immediately
before submission. `final_submission_ready` remains `false`.

**2026-06-12 manuscript/evidence gate.** The Phase 2.5 evidence layer is now
ready for strict manuscript-level claim/citation review, but the paper is not
submission-ready. Codex added a reproducible final claim gate at
`scripts/audit/main_build_final_paper_claim_audit.py`; the generated
`outputs/summary/paper_critical/final_paper_claim_audit_20260612.{json,md}` and
claims CSV report `ok=true`, `paper_evidence_ready_for_drafting=true`,
`final_submission_ready=false`, and verdict
`READY_FOR_MANUSCRIPT_LEVEL_CLAIM_AND_CITATION_AUDIT`. The gate allows six
supported claims (same-candidate official comparison, full evidence ledger,
uncertainty stratification, diagnostic component ablation, NDCG@10
hyperparameter stability, and framework figure use) and explicitly marks two
strong mechanism claims as contradicted: every C-CRP component is necessary,
and positive eta/risk penalty is necessary. The stale `paper/` draft was
rewritten to the current C-CRP same-candidate evidence spine, stale calibration
table removed, current main/module tables added, and
`scripts/analysis/main_build_paper_result_tables.py` now generates visible
full-baseline and significance tables from existing local evidence. The table
audit at
`outputs/summary/paper_critical/paper_result_tables_audit_20260612.{json,md}`
reports `ok=true`, `method_row_count=36`, eight official rows per domain, C-CRP
rank 1 by NDCG@10 in every domain, and `56/56` positive Holm-significant paired
tests in every domain. Citation repair replaced placeholder/incomplete
BibTeX with 19 used references; `Paper/main.blg` reports `warning$ -- 0`, and
`outputs/summary/paper_critical/citation_audit_repair_20260612.{json,md}`
reports `overall_citation_health=MINOR_ISSUES_AFTER_REPAIR`,
`must_add_count=0`, `bibtex_warning_count=0`, and recency verdict `Good`.
`pdflatex -> bibtex -> pdflatex -> pdflatex` now produces `Paper/main.pdf`
(8 pages, 533021 bytes) after the first structural expansion pass. The
expansion added detailed method notation, protocol/fairness details, an
uncertainty-stratification table, numeric diagnostic summaries, and updated
scientific limitations. The structural expansion audit is saved at
`outputs/summary/paper_critical/manuscript_structural_expansion_audit_20260612.{json,md}`.
The paper-critical pytest subset reports `66 passed`; full local pytest
collection remains blocked by pre-existing import-path issues in historical
official-runner tests, so it is not used as the manuscript gate. A follow-up
section-level ARIS review after local claim-boundary fixes is saved at
`outputs/summary/paper_critical/section_review_followup_after_local_review_20260612.{json,md}`.
Meitner first returned `26/35 = 7.4/10`; after risk-term wording downgrades,
stability/sensitivity wording, signal-row schema/parser/vLLM reproducibility
details, selective-classification citations, and canonical-doc SOTA cleanup,
the re-review returned `28/35 = 8.0/10`, a bare conditional section-readiness
pass. The regenerated claim audit
`outputs/summary/paper_critical/final_paper_claim_audit_after_local_review_20260612.{json,md}`
still reports `final_submission_ready=false`, so final submission readiness
remains blocked on citation metadata cautions and a final full review panel
including the specifically requested Claude Opus perspective if available. The
manual citation/proceedings recheck is saved at
`outputs/summary/paper_critical/final_citation_proceedings_recheck_20260612.{json,md}`:
it reports `cited_key_count=21`, `bibliography_entry_count=21`,
`missing_in_bib=[]`, `uncited_in_bib=[]`, and `placeholder_hits=[]`. It also
updates ProEx to the KDD 2026 `V.1` booktitle detail and upgrades ProMax from
arXiv-only `@misc` to SIGIR 2026 `@inproceedings` with ACM DOI
`10.1145/3805712.3809600`, retaining arXiv `2604.26231` and `numpages=11` from
arXiv metadata; remaining citation cautions are ProMax final ACM page range and
ACM/Crossref registry visibility immediately before submission, plus a
ProEx ACM-visible metadata spot-check because Crossref returned 404 in this
environment. A two-reviewer final citation/claim follow-up is saved at
`outputs/summary/paper_critical/final_panel_citation_claim_followup_20260612.{json,md}`:
Faraday returned `8.0/10` conditional pass and Meitner returned `8.5/10`
conditional pass/pass-with-scope-guards. Both reviewers agreed that the
citation/claim gate is conditionally passed, no new experiment is required, and
`final_submission_ready` must remain `false` until the remaining metadata
cautions and final manuscript signoff are closed. Full-manuscript panel review
is saved at
`outputs/summary/paper_critical/final_full_manuscript_panel_review_20260612.{json,md}`:
Faraday returned `28/35 = 8.0/10`, Avicenna returned `29/35 = 8.3/10`, and
Meitner returned `28/35 = 8.0/10`, all weak-accept/conditional-pass under
scope guards. No new experiment is required. The actionable panel fixes were
applied in the manuscript: `Paper/sections/method.tex` now names
`calibrated_relevance_probability` as the paper-facing score field consumed by
the scorer rather than implying a new test-fitted calibrator, and
`Paper/figures/framework_overview.{pdf,svg}` makes the framework figure local
to the paper package. The regenerated
`outputs/summary/paper_critical/final_paper_claim_audit_after_full_panel_review_20260612.{json,md}`
remains `ok=true`, `paper_evidence_ready_for_drafting=true`, and
`final_submission_ready=false`.
A previous ARIS paper-claim-audit pass on the expanded manuscript is saved at
`outputs/summary/paper_critical/manuscript_claim_audit_after_structural_expansion_20260612.{json,md}`;
it reports `claim_text_verdict=READY_WITH_SCOPE_GUARDS`,
`submission_gate_verdict=NEEDS_SECTION_REVIEW_BEFORE_SUBMISSION`,
`SUPPORTED=11`, `WEAKLY_SUPPORTED=1`, `UNSUPPORTED=0`, `OVERCLAIMED=0`, and
`CONTRADICTED=0`. The manuscript audit
at
`outputs/summary/paper_critical/manuscript_claim_citation_audit_20260612.{json,md}`
reported `NEEDS_SECTION_REVIEW_BEFORE_SUBMISSION` before the final spot-check
and section-review edits. Later section-review passes advanced the section gate,
but final submission readiness is still not closed:
`outputs/summary/paper_critical/final_citation_spot_check_20260612.{json,md}`
reports `overall_citation_health=HEALTHY_WITH_PROCEEDINGS_METADATA_CAUTION`,
`must_add_count=0`, all 19 cited keys resolved, all eight official baselines
cited, and `Paper/main.blg` still at `warning$ -- 0`; ProEx/ProMax proceedings
metadata remains a pre-submission recheck caution. GPT-5.5 xhigh section-level
review returned `8.0/10` conditional pass, not submission-ready. The follow-up
audit at `outputs/summary/paper_critical/section_level_review_20260612.{json,md}`
records the applied fixes: visible official baseline provenance table,
all-metric C-CRP rank table, four-domain ablation summary table, method
provenance/selection-grid detail, exact bootstrap/Holm wording, risk-penalty
necessity downgrades, and expanded limitations. After these edits,
`pdflatex -> bibtex -> pdflatex -> pdflatex` produces `Paper/main.pdf`
(9 pages, 541654 bytes) with no undefined citation/reference or BibTeX blocker.
The evidence-to-claim gate was rerun at
`outputs/summary/paper_critical/final_paper_claim_audit_post_section_review_20260612.{json,md,csv}`
and remains `ok=true`, `paper_evidence_ready_for_drafting=true`,
`final_submission_ready=false`, with the same explicit contradicted claims:
every component necessary and positive eta/risk penalty necessary. A fresh ARIS
claim-text pass at
`outputs/summary/paper_critical/manuscript_claim_audit_after_section_review_20260612.{json,md}`
reports `claim_text_verdict=READY_WITH_SCOPE_GUARDS`,
`submission_gate_verdict=NEEDS_FINAL_PANEL_REVIEW_BEFORE_SUBMISSION`, 12
supported claims, and no weakly supported, unsupported, overclaimed, or
contradicted manuscript claims. Do not claim final submission readiness until
paper-critical tests/readiness checks remain green and another section-level
top-conference review passes on this latest draft; Claude Opus reviewer
perspective was still unavailable in this session.

**2026-06-11 observation/motivation closure.** The Phase 2.5
observation/motivation module is now closed as descriptive motivation-only
evidence. Sports/Toys/Home/Tools packages at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_{sports,toys,home,tools}/`
each pass the hardened package audit with `ok=true`,
`paper_claim_ready=true`, and `failures=[]`. The audit now independently
requires `1,010,000` finite uncertainty rows, exactly `101` rows/event,
`invalid_uncertainty_rows=0`, event-bin `candidate_rows=101`, exact event
joins, `same_candidate_alignment.json`, same-candidate candidate key count
`1,010,000`, score coverage `1.0`, zero missing/extra/duplicate/invalid score
keys, and local/server hash evidence. The four-domain aggregate at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/observation_four_domain/`
reports `ok=true`, `paper_claim_ready=true`,
`claim_status=uncertainty_stratifies_reliability`, and
`table_eligibility=motivation_only_not_main_table_sota`. The claim gate passes
because high-uncertainty C-CRP bins have lower `NDCG@10`, `MRR`, and `HR@10`
than low-uncertainty bins in all four domains. Paper wording may say
"C-CRP event-level uncertainty stratifies ranking reliability under the
same-candidate protocol" and may use the five-bin trend and high-minus-low
figures as motivation evidence. It must not claim causality, statistical
significance, baseline calibration, exhaustive baseline behavior, or main-table
SOTA evidence from this module. GPT-5.5 xhigh post-module review returned
**PASS, 8.4/10**; an engineering review veto was lifted after gate hardening
with **8.6/10**. Claude Opus reviewer tooling was unavailable in this session.

**2026-06-11 status update.** Home now has a completed local Phase 2.5
component-ablation evidence package at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_ablation_home/`.
The package audit
`phase2_5_component_ablation_package_audit.{json,md}` reports `ok=true`,
`paper_claim_ready=true`, and `failures=[]`. The package was repaired before
being accepted: the component summary now freezes the preregistered main C-CRP
configuration (`eta=1.0`, `tie_break_seed=20260607`) rather than the
validation-sensitivity best row (`eta=0.5`), and the same-candidate import was
rerun with `--tie_break_seed 20260607` so imported ranking metrics match
`selected_test_metrics.csv`. This does not make the whole paper ready. It also
does not support a strong component-necessity claim: Home, like Sports/Toys,
shows neutral-to-slightly-better leave-one-component-out rows for some
uncertainty/risk components. Tools now repeats that pattern and is complete
through the same package gate:
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_ablation_tools/phase2_5_component_ablation_package_audit.{json,md}`
reports `ok=true`, `paper_claim_ready=true`, and `failures=[]`. Tools selector
used the preregistered main config (`eta=1.0`, weights `0.5,0.3,0.2`,
`tie_break_seed=20260607`) with score coverage `1.0` and imported
`NDCG@10=0.16949770813562767`, but `without_counterevidence` and
`without_risk_penalty` are higher on NDCG@10. The paper must report this as
weak or redundant component evidence, not as a strong component-necessity
claim. The selector command surface was corrected on 2026-06-11 so server-side
`--import_scores` calls the real importer at
`scripts/misc/main_import_same_candidate_baseline_scores.py` and preserves the
seeded tie-break via `--tie_break_seed`.
Post-module GPT-5.5 xhigh sidecar review rated the Tools component-ablation
package **CONDITIONAL PASS, 7.5/10** and limited its table eligibility to
supplementary/diagnostic ablation evidence. Later on 2026-06-11, Sports and
Toys component packages were backfilled from existing full-scale server signal
rows without LLM re-query, passed package audit with `ok=true`,
`paper_claim_ready=true`, and `failures=[]`, and were aggregated with Home and
Tools at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_component_ablation_four_domain/`.
The aggregation provenance reports `ok=true`, `paper_claim_ready=true`,
`delta_convention=removal_minus_full`, `tie_epsilon=1e-12`, and
`table_eligibility=supplementary_diagnostic_only`. The paper-facing wording
must be cautious: boundary uncertainty is inert across all four domains;
counterevidence and risk-penalty removal are nonworse or better on NDCG@10;
evidence support is directionally supportive but small; calibration gap is
mixed. A GPT-5.5 xhigh post-module review rated the completed component module
**CONDITIONAL PASS, 8.1/10**. The real hyperparameter-curve package has since
closed as supplementary stability/sensitivity evidence on 2026-06-12; the next
paper-readiness step is final paper-facing evidence consolidation plus
claim/overclaim/citation audit, not another default experiment launch.

1. An observation/motivation study explaining why uncertainty should be used in
   this framework. It should use representative completed baselines and fair
   same-candidate settings, not paid/SOTA general-model exhaustiveness, and it
   must produce a paper-ready figure or table showing the uncertainty
   phenomenon. Script entry:
   `scripts/analysis/main_build_uncertainty_observation_study.py`; it requires
   real event-level uncertainty fields and intentionally rejects score-only
   C-CRP files. Current status: closed on 2026-06-11 for
   Sports/Toys/Home/Tools as `motivation_only_not_main_table_sota` evidence
   after package audits, exact four-domain aggregation, GPT-5.5 xhigh review,
   and engineering gate-hardening review. The older audit history below is
   retained to explain why score-only formal outputs were insufficient before
   the full-scale signal/selector packages existed. Historical artifact audit
   status: the server had a complete
   `ccrp_selected_test_scored_rows.csv`/`ccrp_internal_provenance.json` pair
   only for the older Beauty supplementary smaller-N selector run; the
   Sports/Toys/Home/Tools formal C-CRP outputs currently visible are score-only
   and cannot support the motivation figure until real signal/scored rows are
   located or regenerated without LLM re-query leakage. Fixed-filter discovery
   and full audits on 2026-06-04 confirmed that each visible new-domain formal
   C-CRP `scores.csv` has 1,010,000 rows, 10,000 events, and candidate coverage
   `1.0`, but all four are `score_only_not_uncertainty` because they lack an
   uncertainty column. Preflight entries:
   `scripts/audit/main_discover_ccrp_uncertainty_sources.py`,
   `scripts/audit/main_remote_discover_ccrp_uncertainty_sources.py`, and
   `scripts/audit/main_audit_ccrp_uncertainty_sources.py`. A 2026-06-04
   project-root broad scan also found no additional full-scale signal rows
   outside `outputs/`. Static trace
   `scripts/audit/main_trace_ccrp_formal_signal_path.py` shows the formal v3
   runner only requested `relevance_probability` and did not preserve
   evidence/counterevidence, `ccrp_uncertainty`, selected scored rows, or
   internal provenance, so the paper-critical uncertainty rows cannot be
   reconstructed from formal `scores.csv` alone. Guarded planning entry:
   `scripts/audit/main_plan_ccrp_signal_generation.py` writes a planning-only
   JSON/shell command package for Sports/Toys signal discovery, source audit,
   validation selection, ablation, observation, and hyperparameter plots. The
    generated shell exits before running commands and remains
    `planning_only_not_executed` until real full-scale signal paths pass audit.
    On 2026-06-05 01:50 CST, the generator and plan artifact were hardened to
    require no active official baseline row or matching baseline Python process,
    replacing stale row-specific wording while preserving the non-executing
    shell guard.
    Consolidated module audit entry:
   `scripts/audit/main_audit_paper_critical_modules.py` writes
    `outputs/summary/paper_critical/paper_critical_module_audit_20260604.{json,md}`
    without running experiments. The current audit reports
    `paper_ready=false`, `framework_overview_scaffold_ready=true`,
    `guarded_plan_ready=true`, and `signal_rows_available=false`, so the
    observation, component-ablation, and hyperparameter modules remain blocked
    by missing full-scale uncertainty or recomputable signal rows.
    Post-Phase2 checkpoint:
    `outputs/summary/paper_critical/paper_critical_module_audit_post_phase2_20260605.{json,md}`
    and
    `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_phase2_20260605/`
    update this state after the Sports/Toys/Home/Tools official+C-CRP gate
    certificate. The guarded plan now covers all four new domains and records
    `will_start_experiment=false`; the generated shell exits with `exit 2`
    before any command. The audit still reports `paper_ready=false` and
    `signal_rows_available=false`, so observation, component ablation, and
    hyperparameter claims remain blocked until full-scale valid/test
    uncertainty signal rows are located or regenerated and audited under the
    same-candidate protocol.
    Post-cleanup ARIS checkpoint on 2026-06-06 06:36-06:38 CST: after the
    completed-row checkpoint cleanup cleared the storage floor, Codex reran
    read-only server/storage and C-CRP signal-source audits. Storage/process is
    no longer the immediate blocker:
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_after_cleanup_final_20260606_0650.{json,md}`
    reports zero active project Python processes, idle GPU, `/` at
    `25,656,160,256` free bytes / `88%` used, and
    `experiment_launch_allowed=true`. The signal-row gate remains closed:
    `outputs/summary/paper_critical/ccrp_signal_source_discovery_after_cleanup_20260606_0640.{json,csv}`
    found only the four formal C-CRP `scores.csv` files, and per-domain audits
    `outputs/summary/paper_critical/ccrp_signal_source_audit_{sports,toys,home,tools}_after_cleanup_20260606_0645.{json,csv}`
    classify each as `score_only_not_uncertainty` with candidate-key coverage
    `1.0` and failure `missing_uncertainty_column`. No Phase 2.5 experiment or
    baseline was launched.
    Signal-row runner wiring checkpoint on 2026-06-06 07:20 CST: performance
    evidence is not the current blocker. A fresh server audit confirmed C-CRP
    v3 reports exist for all eight domains, and Sports/Toys/Home/Tools each
    have `8/8` official completed baselines with `implementation_status=
    official_completed` and `blockers=[]`. The paper-facing ledger already
    reports `official_row_count=32`, `ccrp_row_count=4`, and per-domain
    `8 official + 1 C-CRP` rows. Codex added
    `experiments/rsc/run_ccrp_v3_signal_rows.py`, a server-side runner that can
    generate recomputable C-CRP valid/test signal rows with identity,
    raw/calibrated relevance probability, evidence support, counterevidence
    strength, provenance, row counts, and parse-failure audit fields. The
    guarded plan generator now emits valid/test signal-row generation templates
    but remains non-executing by default. Current consolidated audit:
    `outputs/summary/paper_critical/paper_critical_module_audit_post_signal_runner_plan_20260606_0720.{json,md}`.
    It reports `four_domain_evidence_consistent=true`,
    `phase2_5_storage_launch_allowed=true`, and `guarded_plan_ready=true`, but
    keeps `paper_ready=false` because no completed full-scale signal rows have
    passed audit yet. No server experiment was launched.
    A 2026-06-06 15:55 CST certificate-audit fix updated
    `scripts/audit/main_audit_cross_domain_official_ccrp_certificate.py` so
    `phase2_5_storage_launch_allowed=true` is accepted as a valid current
    storage state rather than reported as a compact-certificate warning. The
    cross-domain certificate still requires `paper_ready=false`,
    `four_domain_evidence_consistent=true`, and `signal_rows_available=false`
    for this pre-signal-row boundary. Focused verification passed with
    `python -m pytest tests\test_audit_cross_domain_official_ccrp_certificate.py`
    (`4 passed`), and a real-input temporary rerun reported `ok=true`,
    `comparison_certificate_ready=true`, `paper_ready=false`, and `warnings=[]`.
    This changes audit interpretation only; it does not complete any Phase 2.5
    signal-row, observation, ablation, or hyperparameter artifact.
    Phase 2.5 Sports-valid launch on 2026-06-06: before server use, Codex
    hardened `experiments/rsc/run_ccrp_v3_signal_rows.py` with a strict
    generation-count guard and pushed commit `70f2f0d`. The first launch attempt
    used base `/home/ajifang/miniconda3/bin/python` and failed before writing
    rows because vLLM could not import `libcudart.so.13`. The corrected bounded
    launch uses `/home/ajifang/miniconda3/envs/qwen_vllm/bin/python`, PID
    `3543564`, log `ccrp_signal_rows_sports_valid_20260606_071906.log`, and
    output dir
    `outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_signal_rows_sports`.
    Local launch manifest:
    `outputs/summary/paper_critical/ccrp_signal_rows_sports_valid_launch_20260606_071906.{json,md}`.
    This running valid-split job is not paper-ready evidence yet. After
    completion, its `valid_ccrp_signal_rows.csv` must pass the C-CRP uncertainty
    source audit against the Sports valid `candidate_items.csv` before selector
    use; test/observation/ablation/hyperparameter use remains blocked until the
    relevant gates pass.
    Packaging hardening while the row is active: at the 2026-06-06 07:40 CST
    monitor, PID `3543564` was still unique and healthy, GPU was `94%`,
    root disk had `25,982,668,800` bytes free / `87%` used, fatal log scan was
    clean, and the first chunk had reached `25149/505000` processed prompts.
    Codex added `scripts/audit/main_sync_ccrp_signal_evidence_package.py` so
    completed signal rows are synced into a local lightweight-but-complete
    package with server/local sha256 checks, row-count/provenance checks,
    parse-failure checks, and source-audit candidate-key coverage checks before
    any paper-critical module consumes them. Verification passed with
    `tests\test_sync_ccrp_signal_evidence_package.py`,
    `tests\test_ccrp_v3_signal_rows_runner.py`, and
    `tests\test_ccrp_uncertainty_source_audit.py`, plus the Phase 2.5 module
    package auditor tests (`40 passed`). This changes packaging discipline
    only; it does not mark the running signal row complete and does not create
    paper-ready evidence.
    Guarded-plan hardening while the row is active: at the 2026-06-06 07:52 CST
    monitor, PID `3543564` was still active/unique, GPU was `93%`, root disk
    had `25,982,087,168` bytes free / `87%` used, fatal scan was clean, and the
    first chunk had reached `41214/505000` processed prompts. Codex updated
    `scripts/audit/main_plan_ccrp_signal_generation.py` and regenerated the
    tracked guarded plan artifacts so generated valid/test signal rows have
    split-specific source-audit commands and local post-completion sync/package
    audit templates before selector use. The generated shell still exits before
    execution and does not include local sync commands. Verification passed with
    `tests\test_plan_ccrp_signal_generation.py`,
    `tests\test_audit_paper_critical_modules.py`, and
    `tests\test_sync_ccrp_signal_evidence_package.py`, plus related signal
    runner/source/package-audit tests (`51 passed`). This is
    planning hardening only; it does not complete the signal row or any
    paper-critical module.
    Paper-facing comparison ledger checkpoint on 2026-06-06 02:35-02:40 CST:
    a GPT-5.5 xhigh read-only sidecar reviewer confirmed the compact
    four-domain comparison certificate was supported but lacked a strict
    paper-facing provenance ledger. Codex added
    `scripts/audit/main_audit_cross_domain_official_ccrp_certificate.py` and
    `scripts/audit/main_build_new_domains_paper_facing_evidence_ledger.py`.
    Artifacts:
    `outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_20260606_0235.{json,md,sha256}`,
    `outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.{csv,json,md}`,
    and
    `outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_20260606_0240.sha256`.
    The certificate audit reports `ok=true` and
    `comparison_certificate_ready=true`; the ledger reports `ok=true`,
    `comparison_ledger_ready=true`, `row_count=36`, `official_row_count=32`,
    `ccrp_row_count=4`, complete HR@5/@10/@20, NDCG@5/@10/@20, MRR,
    row-count, status, provenance, gate, paired-test, score-audit,
    server-final, and local-light evidence columns. This supports drafting the
    four-domain official comparison table, but `paper_ready=false` remains
    because the uncertainty observation, component ablation, and hyperparameter
    modules still need full-scale uncertainty signal rows.
    Follow-up monitor at 2026-06-06 08:37 CST: the same Sports-valid
    signal-row PID `3543564` remained active and unique, GPU was `100%` with
    `42863 MiB / 49140 MiB`, root disk had `25,979,711,488` bytes free / `87%`
    used, fatal scan was clean, and the first chunk had reached
    `103908/505000` processed prompts. The signal output directory still had
    zero files; `valid_ccrp_signal_rows.csv`, provenance, and source-audit
    artifacts were absent. No source audit, local-light sync, cleanup, or new
    experiment was run.
    Follow-up monitor at 2026-06-06 10:04 CST: PID `3543564` was still active
    and unique, elapsed `02:45:07`, duplicate signal-runner process count was
    `1`, GPU was `100%` with `42863 MiB / 49140 MiB`, and root disk had
    `25,975,029,760` bytes free / `87%` used. Fatal scan remained clean, and
    progress had reached `222693/505000` prompts (`44%`). The signal output
    directory still had zero files; `valid_ccrp_signal_rows.csv`, provenance,
    and source-audit artifacts were absent. No completion gate, source audit,
    local-light sync, cleanup, or new experiment was run.
    Follow-up monitor at 2026-06-06 10:24 CST: PID `3543564` was still active
    and unique, elapsed `03:05:36`, duplicate signal-runner process count was
    `1`, GPU was `100%` with `42863 MiB / 49140 MiB`, and root disk had
    `25,973,907,456` bytes free / `87%` used. Fatal scan remained clean, and
    progress had reached `250729/505000` prompts (`50%`). The signal output
    directory still had zero files; `valid_ccrp_signal_rows.csv`, provenance,
    and source-audit artifacts were absent. No completion gate, source audit,
    local-light sync, cleanup, or new experiment was run.
    Follow-up monitor at 2026-06-06 10:57 CST: PID `3543564` was still active
    and unique, elapsed `03:38:49`, duplicate signal-runner process count was
    `1`, GPU was `100%` with `42863 MiB / 49140 MiB`, and root disk had
    `25,972,076,544` bytes free / `87%` used. Fatal scan remained clean, and
    progress had reached `296362/505000` prompts (`59%`). The signal output
    directory still had zero files; `valid_ccrp_signal_rows.csv`, provenance,
    and source-audit artifacts were absent. No completion gate, source audit,
    local-light sync, cleanup, or new experiment was run.
    Follow-up monitor at 2026-06-06 11:38 CST: PID `3543564` was still active
    and unique, elapsed `04:19:20`, duplicate signal-runner process count was
    `1`, GPU was `100%` with `42863 MiB / 49140 MiB`, and root disk had
    `25,969,831,936` bytes free / `87%` used. Fatal scan remained clean, and
    progress had reached `351871/505000` prompts (`70%`). The signal output
    directory still had zero files; `valid_ccrp_signal_rows.csv`, provenance,
    and source-audit artifacts were absent. No completion gate, source audit,
    local-light sync, cleanup, or new experiment was run.
    Follow-up monitor at 2026-06-06 12:53 CST: PID `3543564` was still active
    and unique, elapsed `05:34:45`, duplicate signal-runner process count was
    `1`, GPU was `93%` with `42863 MiB / 49140 MiB`, and root disk had
    `25,957,179,392` bytes free / `87%` used, above the 10 GiB / 97% alert
    floor. Fatal scan remained clean, and progress had reached
    `455609/505000` prompts (`90%`). The signal output directory still had
    zero files; `valid_ccrp_signal_rows.csv`, provenance, and source-audit
    artifacts were absent. No completion gate, source audit, local-light sync,
    cleanup, or new experiment was run.
    Follow-up monitor/correction at 2026-06-06 13:35 CST: PID `3543564`
    remained active and unique, elapsed `06:16:29`, duplicate signal-runner
    process count was `1`, GPU was `100%` with `42863 MiB / 49140 MiB`, and
    root disk had `25,954,861,056` bytes free / `87%` used. Fatal scan
    remained clean. The apparent `Processed prompts: 505000/505000` line is
    only chunk-local because `chunk_users=5000` and
    `expected_candidates_per_event=101`; Sports valid has `10000` events and
    expects `1,010,000` final signal rows. The first chunk had completed, while
    the second chunk was still adding requests at `337289/505000`. The signal
    output directory still had zero files; `valid_ccrp_signal_rows.csv`,
    provenance, and source-audit artifacts were absent. No completion gate,
    source audit, local-light sync, cleanup, or new experiment was run.
    Follow-up monitor at 2026-06-06 16:11 CST: PID `3543564` remained active
    and unique, elapsed `08:52:39`, with duplicate signal-runner process count
    `1`. GPU was `93%` with `42863 MiB / 49140 MiB`, root disk remained safe at
    `25,946,750,976` bytes free / `87%` used, and fatal scan stayed clean. The
    first chunk had completed; the second chunk had reached `221282/505000`
    prompts, about `71.9%` of the expected `1,010,000` signal rows overall. The
    output directory still had zero files and no `valid_ccrp_signal_rows.csv`,
    provenance, or source-audit artifact, so no completion gate, source audit,
    cleanup, or local package sync was run. While waiting, Codex rechecked the
    signal evidence sync helper and verified it does not impose a too-small
    full-CSV ceiling by default (`--max_signal_rows_mb=900`) while still
    requiring sha256/size, `1,010,001` CSV lines for Sports valid,
    provenance-count agreement, parse-failure bounds, and exact candidate-key
    coverage before any package can pass. Focused verification passed with
    `python -m pytest tests\test_sync_ccrp_signal_evidence_package.py
    tests\test_ccrp_uncertainty_source_audit.py -q` (`8 passed`).
    Follow-up monitor at 2026-06-06 16:16 CST: PID `3543564` remained active
    and unique, elapsed `08:57:33`, with duplicate signal-runner process count
    `1`. GPU was `92%` with `42863 MiB / 49140 MiB`, root disk remained safe at
    `25,946,492,928` bytes free / `87%` used, and fatal scan stayed clean. The
    first chunk had completed; the second chunk had reached `227991/505000`
    prompts, about `72.6%` overall for the expected `1,010,000` Sports-valid
    signal rows. The output directory still had zero files and no
    `valid_ccrp_signal_rows.csv`, provenance, or source-audit artifact, so no
    completion gate, source audit, cleanup, local package sync, or new
    experiment was run.
    Follow-up monitor at 2026-06-06 16:20 CST: PID `3543564` remained active
    and unique, elapsed `09:01:10`, with duplicate signal-runner process count
    `1`. GPU was `93%` with `42863 MiB / 49140 MiB`, root disk remained safe at
    `25,946,288,128` bytes free / `87%` used, and fatal scan stayed clean. The
    second chunk had reached `232950/505000` prompts after the first chunk
    completed, about `73.1%` overall. The output directory still had zero files
    and no signal CSV, provenance, or source-audit artifact, so no completion
    gate, source audit, cleanup, local package sync, or new experiment was run.
    Follow-up monitor at 2026-06-06 16:23 CST: PID `3543564` remained active
    and unique, elapsed `09:04:09`, with duplicate signal-runner process count
    `1`. GPU was `91%` with `42863 MiB / 49140 MiB`, root disk remained safe at
    `25,946,136,576` bytes free / `87%` used, and fatal scan stayed clean. The
    second chunk had reached `236988/505000` prompts after the first chunk
    completed, about `73.5%` overall. The output directory still had zero files
    and no signal CSV, provenance, or source-audit artifact, so no completion
    gate, source audit, cleanup, local package sync, or new experiment was run.
    Follow-up monitor at 2026-06-06 16:26 CST: PID `3543564` remained active
    and unique, elapsed `09:07:10`, with duplicate signal-runner process count
    `1`. GPU was `92%` with `42863 MiB / 49140 MiB`, root disk remained safe at
    `25,945,972,736` bytes free / `87%` used, and fatal scan stayed clean. The
    second chunk had reached `241182/505000` prompts after the first chunk
    completed, about `73.9%` overall. The output directory still had zero files
    and no signal CSV, provenance, or source-audit artifact, so no completion
    gate, source audit, cleanup, local package sync, or new experiment was run.
    Post-CCRP local evidence backfill on 2026-06-06 02:55 CST copied missing
    C-CRP evidence from the project server only: Sports/Toys
    `user_ranks.jsonl` and the missing Sports imported C-CRP same-candidate
    tables. The copied rank files each have `10,000` lines, and Sports
    `ranking_eval_records.csv` has `10,001` lines including header. Updated
    scripts now compute C-CRP local event-restatement readiness from actual
    local files rather than the stale compact-CSV flag. Current artifacts:
    `outputs/summary/paper_critical/cross_domain_official_ccrp_certificate_audit_post_ccrp_backfill_20260606_0255.{json,md}`,
    `outputs/summary/paper_critical/new_domains_paper_facing_full_metric_evidence_ledger_post_ccrp_backfill_20260606_0255.{csv,json,md}`,
    and
    `outputs/summary/paper_critical/new_domains_ccrp_local_evidence_backfill_20260606_0255.sha256`.
    Both regenerated artifacts report `ok=true`, zero failures, and zero
    warnings. This closes the local C-CRP event-evidence packaging warning for
    the four-domain comparison table; it still does not provide the missing
    uncertainty signal rows required for the observation, ablation, and
    hyperparameter modules.
    Storage preflight/cleanup for this next gate is recorded in
    `outputs/summary/paper_critical/server_storage_preflight_20260605.*`,
    `outputs/summary/tools_llmesr_completed_adapter_embedding_cleanup_20260605.{json,sha256}`,
    and `outputs/summary/paper_critical/server_storage_postcleanup_20260605.*`.
    The cleanup removed only completed Tools LLM-ESR paper-adapter staging
    files after final evidence/local-light/domain gates passed, recovering
    `/` to `12,342,898,688` bytes free / `94%` used. Final scores,
    provenance, audits, imported tables, final model checkpoints, task splits,
    and other projects were preserved. This is enough to clear the 10GiB danger
    floor but not enough for the recommended Phase 2.5 regeneration target, so
    signal-row generation remains gated on additional disk or another
    archive-backed cleanup decision.
    A follow-up read-only storage-retention audit is recorded in
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_20260605.{json,md,sha256}`.
    It found no sufficient routine safe-now cleanup: small completed staging
    remnants are only about `57M`, while high-yield candidates are protected
    completed artifacts such as the Tools LLM2Rec upstream embedding
    (`5,662,687,360` bytes) or final LLMEmb/LLM-ESR checkpoints (`3.8G`-`6.8G`).
    Those require an explicit archive/retention decision before deletion. No
    deletion was performed and no Phase 2.5 experiment was launched.
    Guarded planner:
    `scripts/audit/main_plan_phase2_5_retention_cleanup.py` now creates a
    planning-only retention cleanup package. Current artifact:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260605/tools_llm2rec_upstream_embedding_retention_cleanup_plan_20260605.{json,sh,sha256}`.
    It targets the completed Tools LLM2Rec upstream embedding only as an
    approval-required option, records `will_delete=false`,
    `will_delete_files=false`, and `will_execute_cleanup=false`, and its shell
    exits with `exit 2` before `sha256sum` or `rm --`. The planned expected
    embedding sha256 is
    `306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`.
    Current read-only retention helper:
    `scripts/audit/main_audit_phase2_5_storage_retention.py`. The 2026-06-05
    23:42 CST artifact
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260605.{json,md,sha256}`
    confirms no active project Python process, GPU idle, `12,342,640,640`
    free bytes, `64,611,717` safe-now recoverable bytes, and
    `experiment_launch_allowed=false`; routine cleanup still cannot clear the
    `15GiB` Phase 2.5 floor, and high-yield cleanup remains approval-required.
    No deletion was performed and no Phase 2.5 experiment was launched.
    Safe-now cleanup on 2026-06-06 00:10-00:15 CST then removed only the
    audited disposable staging/temp targets with a fixed-target helper:
    `outputs/baselines/paper_adapters/tools_large10000_100neg_llm2rec_official_adapter`,
    `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter`,
    and `tmp_llm2rec_sync`. The pre-delete manifest records file-level sha256
    and size, total deleted bytes `64,574,853`, and post-delete absence for all
    three paths. Evidence:
    `outputs/summary/paper_critical/phase2_5_safe_now_cleanup_manifest_20260605.json`,
    `outputs/summary/paper_critical/phase2_5_safe_now_cleanup_post_domain_gate_20260605.{json,csv}`,
    `outputs/summary/paper_critical/phase2_5_safe_now_cleanup_post_comparison_20260605/`,
    `outputs/summary/paper_critical/server_storage_phase2_5_safe_now_postcleanup_20260605.{json,md}`,
    and `outputs/summary/paper_critical/phase2_5_safe_now_cleanup_evidence_20260605.sha256`.
    Post-cleanup Tools gate still passes with eight official rows plus C-CRP,
    and the post-cleanup comparison still reports C-CRP observed-best on all
    seven metrics with all 56 paired tests positive and Holm-significant.
    Disk remains below the Phase 2.5 floor at `12,407,840,768` free bytes, so
    no signal-row regeneration was launched.
    Ranked retention audit on 2026-06-06 00:20 CST:
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_ranked_20260606.{json,md,sha256}`
    adds deterministic risk tiers and a `recommended_approval_candidate`.
    After safe-now cleanup, no safe-now bytes remain, disk is still below the
    strict floor at `12,407,414,784` free bytes, and the lowest-risk
    approval-required candidate is the completed Tools LLM2Rec upstream
    embedding
    `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
    It is an external embedding cache rather than a final score/provenance/
    audit/table/checkpoint artifact; deleting it would still require explicit
    archive/retention approval and the guarded manifest/gate sequence, but it
    would raise expected free space to `18,070,102,144` bytes and clear the
    minimum Phase 2.5 gate.
    Guarded ranked-plan refresh on 2026-06-06 00:30 CST:
    `scripts/audit/main_plan_phase2_5_retention_cleanup.py` now carries the
    ranked audit metadata directly. New planning-only artifact:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606/tools_llm2rec_upstream_embedding_ranked_retention_cleanup_plan_20260606.{json,sh,sha256}`.
    It records `recommended_by_ranked_audit=true`,
    `retention_risk_tier=approval_required_external_embedding_cache`,
    `retention_risk_rank=20`,
    `ranked_audit_current_free_bytes=12,407,414,784`, and
    `ranked_audit_expected_free_bytes_after_delete=18,070,102,144`. The
    generated shell still exits with `exit 2` before any manifest or delete
    command, and the JSON remains `will_delete=false`,
    `will_delete_files=false`, `requires_explicit_approval=true`. A fresh
    read-only preflight immediately before the refresh found no matching
    project Python process, GPU idle, and `/` at `12,407,390,208` free bytes /
    `94%` used. No deletion was performed and no Phase 2.5 experiment was
    launched. Guard verification passed with the broader paper-critical suite
    (`29 passed`).
    Local/server evidence consistency audit on 2026-06-06 00:40 CST:
    `scripts/audit/main_audit_local_server_evidence_consistency.py` now audits
    local lightweight official-baseline packages against their copied server
    large-artifact manifests without SSH, copying, deletion, or experiment
    launch. Tools evidence:
    `outputs/summary/paper_critical/local_server_evidence_consistency_tools_20260606.{json,md,sha256}`.
    It reports `ok=true`, `row_count=8`, `ok_count=8`, and
    `failure_count=0`, confirming all eight Tools official local packages have
    required local-light files, matching sync-manifest hashes, copied
    server-only large artifact manifests for scores/predictions/model files,
    and no server-only bulk artifacts copied locally. This strengthens evidence
    packaging integrity but does not unblock Phase 2.5 signal generation while
    disk remains below the launch floor.
    Four-new-domain consistency checkpoint on 2026-06-06 00:50 CST:
    `outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_20260606.{json,md,sha256}`
    audits Sports/Toys/Home/Tools together. It reports `ok=false`,
    `row_count=32`, `ok_count=11`, and `failure_count=51`. This does not
    invalidate completed official rows: the failures are local evidence-package
    gaps where older Sports/Toys/Home rows lack copied
    `server_large_artifact_manifest.json` and/or `.sha256`. Representative
    rows still have `server_final_evidence_audit.json` with `ok=true`,
    complete full metrics, exact score coverage, and row counts. The next
    evidence-integrity action is to backfill or regenerate the small
    server-large manifests for those older rows and rerun this audit. Tools
    remains `8/8` consistent.
    Post-backfill consistency checkpoint on 2026-06-06 01:45 CST:
    `outputs/summary/paper_critical/local_server_evidence_consistency_new_domains_post_backfill_20260606.{json,md,sha256}`
    reports `ok=true`, `row_count=32`, `ok_count=32`, and `failure_count=0`
    after regenerating and syncing the small server-large manifests for the
    older Sports/Toys/Home rows. Missing server-only prediction JSONLs are
    accepted only as `certified_missing_after_post_gate_cleanup` when the
    row-local `server_final_evidence_audit.json` proves the file existed with
    `10,000` lines before approved cleanup. This fixes the local evidence
    packaging gap; it does not change scientific metrics, does not start a new
    baseline, and does not unblock Phase 2.5 signal generation while disk
    remains below the launch floor.
    Consolidated paper-critical go/no-go audit on 2026-06-06 01:55 CST:
    `scripts/audit/main_audit_paper_critical_modules.py` now consumes both the
    four-domain local/server evidence consistency artifact and a current
    Phase 2.5 storage audit. Artifacts:
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0155.{json,md,sha256}`
    and
    `outputs/summary/paper_critical/paper_critical_module_audit_post_evidence_backfill_20260606_0155.{json,md,sha256}`.
    The module audit reports `ok=true`, `paper_ready=false`,
    `four_domain_evidence_consistent=true`,
    `framework_overview_scaffold_ready=true`, `component_inventory_ready=true`,
    `guarded_plan_ready=true`, `signal_rows_available=false`, and
    `phase2_5_storage_launch_allowed=false`. The storage audit found no active
    project Python process, GPU idle, `12,406,644,736` free bytes,
    `3,699,482,624` bytes deficit to the 15GiB floor, no safe-now recoverable
    bytes, and eight high-yield approval-required candidates. The current
    lowest-risk candidate remains the completed Tools LLM2Rec upstream embedding
    cache at
    `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`;
    deleting it would clear the minimum gate but still requires explicit
    archive/retention approval and post-delete gate checks. No deletion was
    performed and no experiment was launched.
    Current non-destructive retention decision packet on 2026-06-06 02:00 CST:
    `scripts/audit/main_plan_phase2_5_retention_cleanup.py` now accepts
    `--retention_audit_json` and can emit a Markdown approval-decision memo.
    Fresh storage audit:
    `outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0200.{json,md,sha256}`.
    Current guarded packet:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_plan_20260606_0200.{json,sh,md,sha256}`.
    The packet records `will_delete=false`, `will_start_experiment=false`,
    `requires_explicit_approval=true`, current free bytes `12,406,620,160`,
    expected free bytes after deleting the candidate `18,069,307,520`, and
    `expected_to_clear_min_free_gate=true`. The generated shell still exits
    before `sha256sum` and `rm --`, so no manifesting, deletion, or experiment
    launch occurred.
    Retention packet audit on 2026-06-06 02:05 CST:
    `scripts/audit/main_audit_phase2_5_retention_decision_packet.py` audits the
    packet locally without SSH, deletion, manifesting, or experiment launch.
    Artifact:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_current_retention_decision_packet_audit_20260606_0205.{json,md,sha256}`.
    It reports `ok=true`, `read_only=true`, `will_delete=false`,
    `will_start_experiment=false`, and `failures=[]`, confirming the current
    packet is internally consistent and still non-authorizing. The storage gate
    remains closed until explicit archive/retention approval or disk expansion.
    Live remote pre-approval audit on 2026-06-06 02:12 CST:
    `scripts/audit/main_remote_phase2_5_retention_preapproval_audit.py` checks
    the live server target and evidence without deletion, manifesting, or
    experiment launch. Artifact:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0212.{json,md,sha256}`.
    It reports `preapproval_checks_ready_except_disk=true`; the only failure is
    `disk_below_min_free_before_cleanup`. The live target size is
    `5,662,687,360`, sha256 matches
    `306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`,
    active process count is `0`, provenance is `official_completed` with
    `blockers=[]` and `score_coverage_rate=1.0`, and server-final evidence is
    `ok=true` with scores, prediction, and ranking eval records present. This
    strengthens the approval surface but still does not authorize deletion.
    Guarded cleanup dry-run action on 2026-06-06 02:25 CST:
    `scripts/audit/main_execute_phase2_5_retention_cleanup.py` validates the
    retention plan, packet audit, and a fresh live preapproval audit before
    rendering the ordered cleanup steps. Fresh artifacts:
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_preapproval_audit_20260606_0225.{json,md}`,
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.{json,md}`,
    and
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current/tools_llm2rec_upstream_embedding_cleanup_action_dry_run_20260606_0225.sha256`.
    The action artifact reports `mode=dry_run`, `ok=true`,
    `will_delete=false`, and
    `execution_status=dry_run_no_remote_commands`. No cleanup command was
    executed, no artifact was deleted, and no experiment was launched.
    Current retention-decision checkpoint on 2026-06-06 03:00 CST:
    a fresh read-only storage audit found no active project Python process,
    GPU idle, and `/` at `12,406,411,264` free bytes / `94%` used, still below
    the `15GiB` Phase 2.5 launch floor. The consolidated module audit
    `outputs/summary/paper_critical/paper_critical_module_audit_current_20260606_0300.{json,md}`
    reports `ok=true`, `paper_ready=false`,
    `four_domain_evidence_consistent=true`,
    `framework_overview_scaffold_ready=true`, `component_inventory_ready=true`,
    `guarded_plan_ready=true`, `signal_rows_available=false`, and
    `phase2_5_storage_launch_allowed=false`. Codex fixed
    `scripts/audit/main_plan_phase2_5_retention_cleanup.py` so generated
    retention decision packets emit the `.sha256` packet manifest required by
    the packet auditor. The refreshed non-destructive packet under
    `outputs/summary/paper_critical/retention_cleanup_plan_20260606_current_0300/`
    records the completed Tools LLM2Rec upstream embedding as the same
    approval-required candidate, expected to raise free space to
    `18,069,098,624` bytes if explicitly approved and deleted. Packet audit
    now reports `ok=true` and `failures=[]`; live preapproval reports
    `preapproval_checks_ready_except_disk=true` with only
    `disk_below_min_free_before_cleanup`; dry-run action rendering reports
    `ok=true`, `will_delete=false`, and
    `execution_status=dry_run_no_remote_commands`. No server cleanup,
    deletion, post-delete gate, experiment launch, or baseline launch occurred.
    Checkpoint manifest:
    `outputs/summary/paper_critical/phase2_5_retention_decision_checkpoint_20260606_0300.sha256`.
   Observation-builder guard hardening on 2026-06-04: the motivation script now
   rejects duplicate ranking-eval events, eval events absent from the C-CRP
   uncertainty input, invalid positive ranks, and `num_candidates` mismatches
   when ranking records include that column. Its provenance labels the output
   as `artifact_class=paper_critical_observation_motivation` with
   `paper_claim_scope=motivation_only_not_main_table_sota`, records the full
   required metric set (MRR, HR@5/@10/@20, NDCG@5/@10/@20), and includes
   explicit claim limits so representative motivation evidence cannot be
   mistaken for main-table SOTA evidence.
2. A component ablation study over every nontrivial C-CRP design component
   found in the implementation/docs. At minimum, audit score mode, boundary
   uncertainty (`without_boundary_uncertainty`), calibration gap
   (`without_calibration_gap`), evidence support/insufficiency
   (`without_evidence_support`), counterevidence, risk penalty, eta,
   confidence weight, and C-CRP weight triples where the code path supports
   them. A component whose removal is neutral or better must be reported as
   weak or misdesigned rather than hidden.
3. A hyperparameter analysis over real method controls, with curves and
   validation-only selection clearly separated from test reporting. Candidate
   controls include eta, confidence weight, weight triples, uncertainty
   thresholds/gates, anchor conflict penalties, and SRPD learning-rate/lambda
   controls when SRPD rows are used. C-CRP sweep plotting entry:
   `scripts/analysis/main_plot_ccrp_hyperparameter_sweep.py`, consuming
   validation `valid_ccrp_sweep.csv` outputs and requiring at least three values
   per paper-facing curve by default. Paper-facing stability wording requires
   validation and test curves to be reported separately, using `--test_sweep_csv`
   when the test sweep is available. Guard hardening on 2026-06-04 requires
   `audit_ok` and `degeneracy_audit_ok` columns by default, labels valid-only
   outputs as `validation_only_hyperparameter_selection_curve` with
   `paper_claim_scope=validation_only_not_stability_claim`, labels incomplete
   or audit-not-enforced curves as diagnostic-only, and reserves
   `paper_critical_hyperparameter_curve_ready` for audited valid+test curves.
4. A clean framework overview figure showing the full pipeline and where
   uncertainty and C-CRP components enter the ranking decision. Figure builder:
   `scripts/analysis/main_build_framework_overview_figure.py`; generated SVG
   is the editable source and PDF/PNG are export artifacts. Draft package:
   `outputs/summary/paper_critical/framework_overview/` with SVG, PDF, PNG,
   caption, provenance, and `framework_overview_manifest.sha256`. This package
   is paper-critical evidence scaffolding, not yet a final camera-ready figure.
   It was regenerated at 2026-06-04 17:31 CST against git commit `9badd19`;
   `python -m pytest tests\test_framework_overview_figure.py
   tests\test_audit_paper_critical_modules.py` passed (`4 passed`) and the PNG
   was visually checked as nonblank/readable.
   Review-ready checkpoint 2026-06-05 23:20 CST: the framework package was
   regenerated and the module audit now treats it as a figure-only
   review-ready artifact when the SVG/PDF/PNG/caption/provenance files match
   the sha256 manifest, the SVG contains the required pipeline labels, the PNG
   is valid and at least `1800x900`, provenance records
   `status_label=paper_critical_framework_overview_review_ready`,
   `paper_claim_ready=true`, the controlled same-candidate claim boundary, and
   the multiplicative risk formula
   `risk_score = base_score * (1 - uncertainty)^eta`. Evidence:
   `outputs/summary/paper_critical/framework_overview/` and
   `outputs/summary/paper_critical/paper_critical_module_audit_post_framework_review_20260605.{json,md,sha256}`.
   The audit reports framework overview `status=review_ready` and
   `paper_claim_ready=true`, but keeps overall `paper_ready=false` because
   observation, component-ablation, and hyperparameter claims still lack
   full-scale valid/test uncertainty signal rows.

Each module needs status labels, row counts, commands, configs, seeds when
applicable, provenance notes, plots/tables, and a lightweight local evidence
package before it can support paper claims.
Package-audit hardening on 2026-06-06: Codex added
`scripts/audit/main_audit_phase2_5_module_package.py`, a local read-only audit
for finished observation/motivation, component-ablation, and hyperparameter
packages. It enforces the Evidence Package Standard in
`docs/paper_critical_experiment_plan_2026-06-03.md`: command/log/config/hash
records, git commit, row-count and join-count evidence, metrics/tables,
figures, provenance/status labels, and local/server manifest comparison. It
also fails closed if a component-ablation package lacks an explicit
`component_ablation_summary.csv` covering every expected leave-one-component-out
ablation with full metrics and row counts, so a validation-only C-CRP sweep
cannot be mislabeled as completed component evidence. The component gate also
checks selected-valid/test artifacts, imported same-candidate tables, exact
coverage, and audit/degeneracy flags. The hyperparameter gate now requires the
main expected controls (`eta` and `weight_grid_label`), treats
`confidence_weight` as diagnostic-only under `confidence_plus_evidence`, and
requires valid and test curves, producer audit-summary fields, and
package-contained figures. This is a packaging gate, not a completed Phase 2.5
result; the modules still need full-scale signal rows and server execution
after a fresh process/GPU/disk preflight passes.

Component-ablation execution support on 2026-06-06: Codex added
`scripts/analysis/main_build_ccrp_component_ablation_summary.py`, tightened
`scripts/misc/main_select_ccrp_variant_on_valid.py` so selected test metrics
include `HR@5/@10/@20`, `NDCG@5/@10/@20`, and `MRR`, and refreshed the guarded
C-CRP signal-generation plan artifact. The builder freezes the
validation-selected config, evaluates expected leave-one-component-out rows on
test without selecting on test, writes summary/provenance/figure files, and
fails closed when validation metadata, full-score-mode eligibility, exact
coverage, audit/degeneracy flags, or valid-sweep ablation coverage are missing.
This improves the Phase 2.5 command surface but is not evidence of completed
component necessity; paper claims remain blocked until full-scale valid/test
signal rows exist, the module package is synced/audited, and a fresh
process/GPU/disk preflight passes. The earlier storage-floor violation was
later superseded by the 2026-06-06 06:23 CST completed-row model-checkpoint
cleanup.

Consolidated audit hardening on 2026-06-06: `scripts/audit/main_audit_paper_critical_modules.py`
now verifies the component-ablation execution-support layer explicitly:
selector full-reporting metrics, dedicated component summary builder,
component package audit requirements, and guarded-plan command templates. The
refreshed checkpoint
`outputs/summary/paper_critical/paper_critical_module_audit_post_component_execution_support_20260606_0348.{json,md,sha256}`
reports `ok=true`, `paper_ready=false`,
`component_ablation_execution_support_ready=true`,
`four_domain_evidence_consistent=true`, `signal_rows_available=false`, and
`phase2_5_storage_launch_allowed=false`. The paired storage audit
`outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0348.{json,md,sha256}`
reports no active experiment, GPU idle, `12,406,190,080` free bytes, and a
`3,699,937,280` byte deficit to the 15GiB launch floor. No cleanup or
experiment was run.

All-module execution-support audit hardening on 2026-06-06:
`scripts/audit/main_audit_paper_critical_modules.py` now verifies execution
support for all three paper-critical modules, not only component ablation. It
checks observation builder guards, component-ablation selector/builder/package
guards, hyperparameter test-sweep/audit/figure guards, and guarded-plan command
templates. The refreshed checkpoint
`outputs/summary/paper_critical/paper_critical_module_audit_post_all_module_execution_support_20260606_0432.{json,md,sha256}`
reports `ok=true`, `paper_ready=false`,
`observation_execution_support_ready=true`,
`component_ablation_execution_support_ready=true`,
`hyperparameter_execution_support_ready=true`,
`four_domain_evidence_consistent=true`, `signal_rows_available=false`, and
`phase2_5_storage_launch_allowed=false`. The paired storage audit
`outputs/summary/paper_critical/server_storage_phase2_5_retention_audit_current_20260606_0432.{json,md,sha256}`
reports no active experiment, GPU idle, `12,397,707,264` free bytes, no
safe-now recoverable bytes, and a `3,708,420,096` byte deficit to the 15GiB
launch floor. No cleanup or experiment was run.

Storage-actionable audit update on 2026-06-06: the consolidated module audit
now promotes the storage cleanup/archive decision into its top-level summary
and Markdown. Artifact:
`outputs/summary/paper_critical/paper_critical_module_audit_storage_actionable_20260606_0440.{json,md,sha256}`.
It reports `storage_safe_now_total_recoverable_bytes=0`,
`storage_recommended_candidate_size_bytes=5662687360`,
`storage_recommended_candidate_would_clear_min_free_gate=true`,
`storage_approval_decision_required=true`, and
`storage_cleanup_decision_required=true` for
`/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
This makes the disk blocker actionable but does not authorize deletion; no
cleanup or experiment was run.

Retention-decision checkpoint refresh on 2026-06-06 04:47 CST: Codex
regenerated the guarded approval packet under
`outputs/summary/paper_critical/retention_cleanup_plan_20260606_current_0447/`
using the 04:32 storage audit as its source, and recorded
`outputs/summary/paper_critical/phase2_5_retention_decision_checkpoint_20260606_0447.sha256`.
The packet remains non-destructive: `will_delete=false`,
`requires_explicit_approval=true`, and the packet audit reports `ok=true` with
no failures. The live read-only preapproval audit confirms target size
`5662687360`, matching sha256
`306618d974eb4133d9cda87bae3251e17d793aa6f5a8cb38d558b549ed31d56e`, no active
project Python process, completed Tools LLM2Rec evidence still present, and
`preapproval_checks_ready_except_disk=true` with only
`disk_below_min_free_before_cleanup`. The dry-run action audit reports
`execution_status=dry_run_no_remote_commands` and `will_delete=false`. This
does not unblock paper-critical signal generation until the disk expansion or
explicit archive/retention decision is made.

Framework overview refresh on 2026-06-06 05:05 CST: after GPT-5.5 xhigh
sidecar review, Codex changed the figure wording from a potentially
overclaiming `Paper-critical method evidence` label to
`Required method-evidence gates`. Codex regenerated the framework overview
figure package at generator commit
`e81d1a8404b2a90467e1912c53e082e1e3c46dae`, updating the editable SVG, PDF/PNG
exports, caption, provenance, and manifest under
`outputs/summary/paper_critical/framework_overview/`. The refreshed figure
keeps the claim boundary visible and uses the multiplicative C-CRP risk formula
`risk_score = base_score * (1 - uncertainty)^eta`. The post-refresh audit
`outputs/summary/paper_critical/paper_critical_module_audit_post_framework_gate_wording_20260606_0505.{json,md,sha256}`
reports `ok=true`, `framework_overview_scaffold_ready=true`, and framework
status `review_ready`, while preserving `paper_ready=false` because the
observation, component-ablation, and hyperparameter modules still require
full-scale uncertainty signal rows. The storage condition present at that
checkpoint was later superseded by the 2026-06-06 06:23 CST completed-row
model-checkpoint cleanup. No experiment or cleanup was run at the framework
checkpoint.

Framework overclaim guard hardening on 2026-06-06 05:18 CST: the consolidated
paper-critical module audit now enforces that the framework overview does not
claim completion of the still-blocked observation/motivation,
component-ablation, or hyperparameter modules. It checks required caption and
SVG phrases, forbids completion wording in those fields, verifies the
provenance caption matches the caption file, requires the
`Does not make observation, ablation, or hyperparameter evidence complete.`
claim limit, and requires every `evidence_gate_status` entry to be
`required_not_claimed_by_figure`. Fresh artifact:
`outputs/summary/paper_critical/paper_critical_module_audit_post_framework_overclaim_guards_20260606_0518.{json,md,sha256}`.
The audit remains `ok=true` for the framework package while preserving
`paper_ready=false` because full-scale signal rows are still missing. The
storage-floor violation present at that checkpoint was later superseded by the
2026-06-06 06:23 CST completed-row model-checkpoint cleanup. No experiment or
cleanup was run at the overclaim-guard checkpoint.

Framework producer-test hardening on 2026-06-06 05:23 CST: the figure-builder
test now verifies the conservative caption phrase, caption/provenance
consistency, and the provenance claim limit that the framework figure does not
complete observation, ablation, or hyperparameter evidence. Focused tests
passed (`9 passed`). This protects the figure generator side of the same
claim-boundary guard; it does not alter paper readiness.

Phase 2.5 package-audit hardening on 2026-06-06 05:29 CST: after GPT-5.5 xhigh
sidecar review, the completed-module package audit now enforces additional
lightweight-complete evidence requirements before observation, ablation, or
hyperparameter packages can support paper claims. It requires seed provenance,
scans log snippets for fatal markers, recursively rejects bulk scores and
prediction JSONL files inside the package, warns on nested large/binary model
artifacts, and validates full ranking metrics plus hyperparameter curve metric
values as finite values in `[0,1]`. Focused tests passed (`18 passed`). This is
local audit hardening only; paper readiness remains blocked by missing
full-scale signal rows and a fresh server preflight before any module launch.

Component-ablation package table-count hardening on 2026-06-06 05:36 CST: the
Phase 2.5 module-package audit now checks imported component-ablation tables
for expected same-candidate row counts and coverage totals. It requires
`ranking_eval_records.csv` to match the expected event count, coverage
`ranking_events` to match expected events, and `total_candidates` /
`matched_candidates` to match `expected_events * expected_candidates_per_event`;
it also rejects unexpected status labels in the same-candidate summary. Focused
tests passed (`20 passed`). This prepares the package gate for real full-scale
ablation evidence after signal rows and storage are unblocked.

Phase 2.5 package manifest-hash hardening on 2026-06-06 05:45 CST: the
completed-module package audit now rejects local/server manifest comparisons
that only report `ok=true`, file presence, row counts, or file sizes. A package
must provide at least one concrete file identity plus valid hash evidence:
either a checked 64-character `sha256` with `sha256_ok`/`hash_ok`, or matching
local/server or expected/actual SHA-256 values. Regression coverage rejects the
old presence-only shape and accepts both row-level local/server hash equality
and `manifest_checks` expected/actual equality. Focused tests passed
(`24 passed` for the package/critical-module/component-builder suite, plus
`5 passed` for local/server consistency). This is local evidence-gate
hardening only; paper readiness still requires full-scale uncertainty signal
rows and a fresh process/GPU/disk preflight before any module launch.

Hyperparameter package summary-shape hardening on 2026-06-06 05:54 CST: after
GPT-5.5 xhigh sidecar review, the completed-module package audit now verifies
`ccrp_hyperparameter_curve_summary.csv` directly for paper-facing stability
evidence. It requires provenance to declare the plotted `metric` and a
`min_values>=3` gate, requires both `valid` and `test` split rows for every
expected control, requires at least three distinct `control_value` entries per
expected control and split, checks metric-name consistency, requires row-level
`audit_ok`, `degeneracy_audit_ok`, exact `score_coverage_rate=1.0`, expected
`candidate_key_count`, one fixed-filter source row per plotted point via
`candidate_rows_for_value=1`, no duplicate `(split, control, control_value)`
curve points, real 64-hex valid/test sweep hashes, different valid/test hashes,
a plotted metric from the project full-metric set, and cross-checks provenance
`control_reports.curve_values` against the summary CSV. Regression coverage
rejects missing row-audit columns, false audit flags, incomplete coverage/key
counts, reused valid/test hashes, missing valid-sweep hashes, best-of-many
curve points, duplicate curve points, unsupported metrics, a missing-test
summary, and too-short curves even when provenance claims
`paper_critical_hyperparameter_curve_ready`. Focused tests passed
(`36 passed`). This is still only local evidence-gate hardening; the real
hyperparameter module remains blocked until full-scale uncertainty signal rows
exist and the resulting package is synced/audited under a fresh server
preflight.

Hyperparameter stability-report hardening on 2026-06-06 06:09 CST: after
GPT-5.5 xhigh sidecar review, the hyperparameter curve producer and package
audit now require an explicit per-control stability report before a package can
support a paper-facing stability claim. The producer emits valid/test best
values, test metric at the validation-best value, relative drop from the
test-best point, test rank of the validation-best value, and
`stable_within_tolerance`; unstable valid/test curves are downgraded to
diagnostic status. The package audit recomputes those fields from
`ccrp_hyperparameter_curve_summary.csv`, rejects mismatched provenance,
missing/duplicate/extra stability controls, missing valid/test evidence,
nonfinite metrics, tolerance above `0.05`, and relative drops above tolerance.
The producer now also requires paper-facing sweep rows to include
`score_coverage_rate` and `candidate_key_count`. Focused tests passed
(`41 passed`). This remains evidence-gate hardening only; the real
hyperparameter analysis still requires full-scale uncertainty signal rows and a
fresh process/GPU/disk preflight before any module launch.

Server storage cleanup on 2026-06-06 06:23-06:25 CST cleared the immediate
Phase 2.5 disk-floor violation without deleting paper-table evidence. After
explicit user direction, Codex deleted only two completed Home official-row
model checkpoint files:
`outputs/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/llmemb_official_model.pt`
and
`outputs/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/llmesr_official_model.pt`.
The cleanup preserved `scores.csv`, `fairness_provenance.json`, score audits,
run summaries, imported tables, `server_final_evidence_audit.json`, and
`server_large_artifact_manifest.json` for both rows. Corrected server cleanup
record:
`outputs/summary/server_cleanup/cleanup_large_completed_model_checkpoints_20260605T222339Z.corrected.{summary.txt,tsv}`.
Freed bytes: `13,259,007,313`; post-cleanup `/` was `25,656,266,752` bytes
free / `88%` used. Paper readiness remains blocked by missing full-scale
uncertainty signal rows and module packages, not by the prior 12.4 GB storage
floor, subject to a fresh process/GPU/disk preflight before launch.

The current execution specification is
`docs/paper_critical_experiment_plan_2026-06-03.md`.

## Not primary claims unless completed

- Generative title recommendation.
- Full-catalog SOTA.
- Universal cross-domain winner.
- LoRA distillation as main novelty.
- Shadow v2-v6 as independent main methods.
- SRPD as a main-method substitute for external baselines unless its formal
  train/eval gates are completed.
- Proxy comparisons against reported numbers from incompatible protocols.

## Status labels

Every paper-facing summary table must include `status_label`.

- `completed_result`: runnable and completed under the stated protocol.
- `runnable_not_complete`: code path exists, but the result is not complete
  enough for a main table.
- `design_only`: method or plan is documented but not fully run under the
  frozen protocol.
- `proxy_only`: useful for related-work positioning or fairness audit, but not
  same-schema evidence.
- `future_extension`: intentionally outside the main paper claim.

Rows with `design_only`, `proxy_only`, or `future_extension` must not enter the
main result table. Rows from fallback internal validation splits are not main
table eligible.

Agent drift guard: every new experiment, baseline, model module, or generated
table must receive a status label and milestone eligibility before it is
described as evidence. If a future agent cannot prove eligibility from
provenance and audits, the row is diagnostic or supplementary.

## Milestone claim eligibility

The canonical milestone map lives in `docs/milestones/README.md`.

| milestone | paper role | main-table eligible |
| --- | --- | --- |
| M0 Week1-4 / pony12 | diagnosis and motivation | no |
| M1 Pony framework | method and protocol bridge | yes, if completed under frozen protocol |
| M2 Light series | precursor / negative-control ablation | supplementary unless explicitly completed under main protocol |
| M3 Shadow series | task-grounded signal family | diagnostic unless large-scale protocol and validation gates are complete |
| M4 Baseline system | fairness and reviewer defense | yes for completed classical rows; official external rows only after provenance passes |
| M5 Four-domain validation | scoped stability/sensitivity evidence | yes when 100neg outputs, audits, and paired tests are complete |
| M6 Complete recommender system | future full-system roadmap | no until official baselines, Shadow, LoRA, and generated-title verification are completed |

The `*_style_*` LLM-rec rows are paper-style supplementary rows, not official
reproductions. Official external-baseline claims require the
`*_official_qwen3base_*` family plus provenance and score coverage.

Internal Shadow/Light/LoRA ablations cannot substitute for missing external
baselines. If official external rows are unfinished, the paper wording must
state the baseline gap and keep those rows out of main official claims.

## Main table eligibility

A row can enter the main ranking table only if all are true:

- `artifact_class == completed_result`
- `status_label` matches the row type, for example
  `same_schema_external_baseline`, `same_schema_internal_method`, or a
  documented completed ablation label.
- Same-schema data, prompt, candidate construction, and metric definitions are
  used.
- External baseline rows follow the declared comparison variant, usually
  `official_code_qwen3base_default_hparams_declared_adaptation`.
- Official external rows require `implementation_status=official_completed`;
  `style_adapter_only` and `partial_official_adapter_exists` rows stay
  supplementary.
- Internal formal rows require exact score export and import. C-CRP uses
  `status_label=same_schema_internal_method`; SRPD uses
  `status_label=same_schema_internal_ablation` unless all trainable-framework
  gates are satisfied.
- C-CRP formal rows must include validation-only selection provenance, input
  hashes, exact score coverage, and score-degeneracy audit output.
- SRPD rows that use rank-order fallback scores are internal ablations only;
  they must not be presented as native trainable scorer results or promoted to
  the main method table.
- Runner or plan filenames containing `official` are not enough. The row must
  have unblocked provenance from the pinned official repo and pass score/import
  gates.
- Baseline hyperparameters come from the official default/recommended setting,
  unless an override is explicitly recorded for protocol compatibility.
- Our method's gates, weights, and hyperparameters are selected on validation or
  fixed before test.
- Full-finetune and retuned-baseline variants are not mixed into the primary
  main comparison table.
- External score files have exact unique `source_event_id,user_id,item_id`
  coverage with finite numeric scores.
- Calibration method and C-CRP weights are selected on validation or fixed
  before test.
- `calibration_split_metadata.csv` has zero user overlap unless the table is
  explicitly labeled non-main.
- Candidate protocol audit exists for the domain/split.
- Statistical tests provide paired confidence intervals for the supported claim.

## Safe wording

Use "observed best" for results whose confidence interval crosses zero or whose
paired test is not significant after correction. Use "winner" only for results
that pass the configured statistical rule.

## C-CRP v3 Progress (2026-05-31)

### Multi-Domain Results

C-CRP v3 uses a profile-enhanced prompt that asks the LLM to infer user
preferences before scoring each candidate. All domains use 10k users,
101 candidates (1 positive + 100 negative), Qwen3-8B via vLLM.

| Domain | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR | vs Best Baseline |
|--------|------|-------|-------|--------|---------|---------|-----|------------------|
| beauty (973u) | 0.157 | 0.229 | 0.369 | 0.111 | 0.134 | 0.169 | 0.128 | #2 (ProEx=0.253) |
| books | 0.374 | **0.476** | 0.592 | 0.300 | **0.333** | 0.362 | 0.306 | historical rank 1 (+0.8% vs LLMEmb) |
| electronics | 0.218 | **0.299** | 0.418 | 0.157 | **0.183** | 0.213 | 0.168 | historical rank 1 (+22% vs LLMEmb) |
| movies | 0.145 | 0.208 | 0.331 | 0.108 | 0.128 | 0.159 | 0.127 | #5 (LLMEmb=0.334) |
| sports | 0.275 | 0.382 | 0.517 | 0.198 | 0.233 | 0.267 | 0.208 | domain gate PASS |
| toys | 0.317 | 0.396 | 0.506 | 0.245 | 0.271 | 0.298 | 0.250 | domain gate PASS |
| home | 0.156 | 0.226 | 0.351 | 0.110 | 0.132 | 0.164 | 0.126 | domain gate PASS |
| tools | 0.194 | 0.270 | 0.393 | 0.142 | 0.166 | 0.197 | 0.156 | domain gate PASS |

Status: `completed_result` for beauty/books/electronics/movies/sports/toys/home/tools.
The C-CRP v3 batch completed without FAILED/OOM/Traceback markers in
`ccrp_v3_all_domains.log`; new-domain official baselines entered Phase 2 on
2026-05-31 with a sports single-domain production run.
For sports/toys/home/tools, each report records `n_users=10000`,
`n_prompts=1010000`, the test same-candidate `data_path`, `scores.csv` with
1,010,000 scored candidate rows plus header, and `user_ranks.jsonl` with
10,000 rows.
Toys-specific audit checkpoint (2026-06-02 07:18 CST): the C-CRP core result
under `outputs/toys_large10000_100neg_ccrp_v3` is metric-complete
(HR@5/10/20 `0.3172 / 0.3964 / 0.5059`, NDCG@5/10/20
`0.2451904009717959 / 0.27079859856897753 / 0.298341205798594`, MRR
`0.2503049488607351`, `n_users=10000`, `scores.csv` `1,010,001` lines, and
`user_ranks.jsonl` `10,000` rows). At 2026-06-02 17:09 CST the raw scores were
imported into
`outputs/toys_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
through the existing same-candidate importer without `--allow_partial_scores`.
The import reported `score_coverage_rate=1.000000`. After the final toys
`llmesr_sasrec` official row completed, the follow-up toys domain gate recorded
`ccrp_ok=true`, `official_ok_count=8`, `official_all_ok=true`, and
`gate_ok=true`. The toys comparison/statistical gate then recorded C-CRP rank
1 on all seven metrics and all 56 C-CRP-vs-official paired tests positive and
Holm-significant. The local lightweight C-CRP import package keeps the imported
tables and gate summary; the large imported prediction JSONL remains
server-only.
Original-domain C-CRP v3 formal reports are present under
`outputs/ccrp_v3_formal/<domain>/report.json`, and the old four-domain
official-baseline comparison is present at
`outputs/ccrp_v3_formal/main_comparison_table.csv`.
Artifact audit note (2026-05-31): the old four-domain C-CRP reports were not
missing; earlier searches that only checked `outputs/*ccrp_v3/report.json`
missed the `outputs/ccrp_v3_formal/<domain>/report.json` layout. The old
four-domain 8-baseline comparison table is metric-complete, but some
method-specific old-domain baseline directories are table-only under the
current strict evidence gate because final provenance/audit files are not
co-located with the imported tables. Reconcile those evidence packs before
paper submission; do not rerun or relabel them without a provenance decision.

### Historical Rank-First Strategy

C-CRP v3 ranked first on books and electronics under the historical comparison
table, and sports/toys/home/tools now pass domain-level official-baseline and
paired-test gates. Full-catalog or universal SOTA wording is still forbidden;
paper-facing wording must stay scoped to controlled same-candidate ranking until
ARIS claim/citation audits pass, the paper-critical observation/ablation/
hyperparameter/figure modules are complete, and each completed module has passed
the per-module tri-reviewer review-as-you-go pass (Codex GPT xhigh + GPT-5.5
xhigh + a second Claude Opus 4.8) with no unresolved major objection. This
per-module cadence replaces the prior single end-of-project GPT-5.5/Codex xhigh
≥ 8/10 gate.

2026-06-12 hyperparameter-module status: the module is closed as scoped
supplementary stability/sensitivity evidence. The
saved-signal entrypoint
`scripts/analysis/main_build_ccrp_hyperparameter_sweep.py` produces valid/test
sweep CSVs and provenance from already audited full-scale C-CRP signal rows
without LLM re-query, score import, or retained per-grid prediction/score
dumps. The main controls are `eta` and `weight_grid_label`; `confidence_weight`
is diagnostic-only for `confidence_plus_evidence` and must not be used as a
full-mode main C-CRP curve. Sports, Toys, Home, and Tools each now have
complete local and server packages under
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_{sports,toys,home,tools}/`
with Phase 2.5 package audits reporting `ok=true`, `paper_claim_ready=true`,
and `failures=[]`; the hardened manifest gate covers 11/11 lightweight files
per domain. The four-domain aggregate at
`outputs/summary/paper_critical/ccrp_signal_generation_plan_post_performance_gate_20260606/ccrp_hyperparameter_four_domain/`
reports `ok=true`, `paper_claim_ready=true`, `all_controls_stable=true`, and
`table_eligibility=supplementary_hyperparameter_stability_only`. On NDCG@10,
`eta` is stable in `4/4` domains with max relative drop
`1.2462949826291314e-05`, and `weight_grid_label` is stable in `4/4` domains
with max relative drop `0.0002974344039075195`. The aggregate includes command
provenance, `run_config.json`, `log_snippets.md`, output hashes/sizes, and a
local/server manifest comparison covering 13/13 lightweight files. GPT-5.5
xhigh protocol review rated the module **CONDITIONAL PASS, 8.3/10**; Codex GPT
xhigh engineering review rated it **CONDITIONAL PASS, 8.0/10** after the
aggregate packaging/tolerance/manifest fixes. Claude Opus review tooling failed
with `Claude CLI did not return JSON output`, so that reviewer perspective is
missing. Paper wording may say validation-selected C-CRP `eta` and weight-grid
settings remain within a 5% relative NDCG@10 drop of the test-best grid point
across four full-scale domains. It must not claim test-selected tuning,
main-table SOTA evidence, all-metric robustness, universal optima, or that the
risk penalty is necessary.

### Remaining for paper submission

1. Consolidate the completed canonical 8-official-baseline gates for
   sports/toys/home/tools into the final paper comparison evidence. Do not
   launch more official baseline rows unless a later audit finds a concrete
   invalid row. `scripts/run_baselines_new_domains.sh` remains aligned to
   exclude SETRec while it is blocked/supplementary, supports single-domain
   production via `DOMAINS_OVERRIDE`, and audits/imports complete
   `@5/@10/@20 + MRR` same-candidate metrics after each completed score file.
   Sports, toys, home, and tools are now 8/8 complete and have passed their
   domain/comparison/paired-test gates. Tools `proex_profile` launched at
   2026-06-04 14:25 CST as the first Tools row and completed at
   2026-06-04 16:08 CST with `implementation_status=official_completed`,
   `blockers=[]`, exact `score_coverage_rate=1.0`, score audit/imported
   tables, server-final audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit all passing. Full metrics over
   10,000 users and 101 candidates are HR@5/10/20
   `0.0602 / 0.1177 / 0.2329`, NDCG@5/10/20
   `0.037281705859706714 / 0.055676376797898205 / 0.08437492971571317`, and
   MRR `0.06071849976691817`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `proex_official_model.pt` are covered by
   `server_large_artifact_manifest.sha256` and
   `prediction_deletion_manifest.json`. The completed intermediate adapter was
   removed after final evidence and local backup passed, recovering `/` to
   about `12G` free / `95%` used while preserving final evidence. Tools
   `promax_profile` launched at 2026-06-04 16:46 CST as the second single-row
   Tools official baseline with runner PID `3279573`, adapter PID `3279582`,
   and log `baselines_new_domains_tools_promax_20260604_164630.log`, then
   completed at 2026-06-04 19:59 CST with `implementation_status=official_completed`,
   `blockers=[]`, exact `score_coverage_rate=1.0`, score audit/imported
   tables, server-final audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit all passing. Full metrics over
   10,000 users and 101 candidates are HR@5/10/20
   `0.056 / 0.1046 / 0.2018`, NDCG@5/10/20
   `0.03468275603534166 / 0.05029722685396016 / 0.07458228366305956`, and MRR
   `0.056527355267188224`; `scores.csv` has `1,010,001` lines, predictions
   had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `promax_official_model.pt` are covered by
   `server_large_artifact_manifest.sha256` and
   `prediction_deletion_manifest.json`; the manifest records sha256 values
   `2a9797b945fef73e76a1db18efe7ef037f2b47732c29ebaaf44ece01b33ac781`
   for scores, `b2123ea945285b9ee7ca940819382191fbae6af945cee09fb741b7c5ca95c717`
   for the deleted prediction JSONL, and
   `c0e17d003ba1e055e65d38dfac4dc96483f4a8744e201e3daf168a71f31890fc`
   for the final model. The completed ProMax intermediate adapter was removed
   after final evidence and local backup passed, with cleanup manifests
   `outputs/summary/tools_promax_completed_adapter_cleanup_manifest_20260604.sha256`
   and `outputs/summary/tools_promax_completed_adapter_cleanup_du_20260604.txt`;
   final scores, provenance, audits, imported tables, and model were preserved.
   Tools `elmrec_graph` launched at 2026-06-04 20:46 CST as the third
   single-row Tools official baseline with runner PID `3301337`, adapter PID
   `3301345`, and log `baselines_new_domains_tools_elmrec_20260604_204602.log`.
   It completed at 2026-06-04 22:39 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact sha256 manifest, lightweight local sync, and
   local-light audit all passing. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0501 / 0.101 / 0.2101`, NDCG@5/10/20
   `0.029656030656687697 / 0.045870649973376774 / 0.07316592297455926`, and
   MRR `0.05237582779698271`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, deleted prediction metadata, and
   `elmrec_official_model.pt` are covered by `server_large_artifact_manifest.*`
   and `prediction_deletion_manifest.json`; the manifest records sha256 values
   `13e8aa52ed5c69fa8c9b04006907b907043dd5fbc26b1829e3542d2ed58b050c` for
   scores, `d49457a2d7fc7a5877f565f51913f7110fac438bd6cadd871a8e2da68237c4fd`
   for the deleted prediction JSONL, and
   `637e253d09007a93dfa1fc3f78ba4209b2d052b76a22f629e6e1ad8bf375a22d` for the
   final model. The completed ElmRec intermediate adapter was removed after
   final evidence and local backup passed, with cleanup manifests
   `outputs/summary/tools_elmrec_completed_adapter_cleanup_manifest_20260604.sha256`
   and `outputs/summary/tools_elmrec_completed_adapter_cleanup_du_20260604.txt`;
   final scores, provenance, audits, imported tables, and model were preserved.
   Tools `llmemb` launched at 2026-06-04 23:10 CST as the fourth single-row
   Tools official baseline with runner PID `3317251`, adapter PID `3317260`,
   and log `baselines_new_domains_tools_llmemb_20260604_231030.log`; it
   completed at 2026-06-05 00:43 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, and passing server-final audit, server
   large-artifact manifest, lightweight local sync, and local-light audit.
   Full metrics over 10,000 users and 101 candidates are HR@5/10/20
   `0.1365 / 0.2257 / 0.3637`, NDCG@5/10/20
   `0.087457824217457 / 0.11594350972806679 / 0.15050644138929892`, and MRR
   `0.10649354669900822`; `scores.csv` has `1,010,001` lines, predictions had
   `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_llmemb_official_qwen3base_same_candidate/`.
   The server-only scores, deleted prediction metadata, and
   `llmemb_official_model.pt` are covered by `server_large_artifact_manifest.*`
   and `prediction_deletion_manifest.json`; cleanup manifests are
   `outputs/summary/tools_llmemb_completed_adapter_cleanup_manifest_20260605.sha256`
   and `outputs/summary/tools_llmemb_completed_adapter_cleanup_du_20260605.txt`.
   During the run, a completed Home LLM2Rec checkpoint was deleted under
   emergency disk approval after sha256/size manifesting; local record:
   `outputs/summary/home_llm2rec_checkpoint_deletion_manifest_20260604.json`.
   Final scores, provenance, audits, imported tables, and model were preserved,
   and post-cleanup `/` returned to about `15G` free / `93%` used. After a
   fresh preflight, Tools
   `irllrec_intent` launched at 2026-06-05 01:04 CST as the fifth single-row
   Tools official baseline with runner PID `3326805`, adapter PID `3326813`,
   log `baselines_new_domains_tools_irllrec_20260605_0058.log`, and heartbeat
   `monitor-tools-irllrec`; its launch snapshot
   `outputs/summary/tools_irllrec_launch_monitor_20260605.json` shows Qwen3
   embedding progress `2056/269711`, GPU `95%`, disk `13.34G` free, and no
   failure markers. It completed normally at 2026-06-05 05:19 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact manifest, lightweight local sync, and local-light
   audit all passing. Full metrics over 10,000 users and 101 candidates are
   HR@5/10/20 `0.102 / 0.1651 / 0.3095`, NDCG@5/10/20
   `0.06504709833452535 / 0.08525707170530923 / 0.12100829406900945`, and MRR
   `0.08670154590435823`; `scores.csv` has `1,010,001` lines, predictions have
   `10,000` lines, and `tables/ranking_eval_records.csv` has `10,001` lines.
   The local lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`.
   Server-only `scores.csv`, predictions, and `irllrec_official_model.pt` are
   covered by `server_large_artifact_manifest.*`; provenance records the
   IRLLRec scalability bridge `deterministic_node_cap`. The completed
   intermediate adapter was removed after gates and local backup passed, with
   cleanup manifest
   `outputs/summary/tools_irllrec_completed_adapter_cleanup_manifest_20260605.sha256`,
   recovering `/` to about `14G` free / `93%` used while preserving final
   evidence. Tools is now 5/8 complete and still needs `rlmrec_graphcl`,
   `llm2rec_sasrec`, and `llmesr_sasrec` plus the Tools
   domain/comparison/paired-test gates. After a fresh preflight, Tools
   `rlmrec_graphcl` launched at 2026-06-05 05:41 CST as the sixth single-row
   Tools official baseline with runner PID `3347729`, adapter PID `3347738`,
   log `baselines_new_domains_tools_rlmrec_20260605_054158.log`, heartbeat
   `monitor-tools-rlmrec`, guarded completion-gate plan
   `outputs/summary/official_completion_gate_plan/tools_rlmrec_graphcl_completion_gates_20260605.{json,ps1}`,
   and launch snapshot `outputs/summary/tools_rlmrec_launch_monitor_20260605.json`.
   The launch snapshot found one matching adapter process, no completion or
   failure markers, adapter dir about `1000M`, final output placeholder-only,
   and disk about `12.35G` free / `94%` used. A 2026-06-05 05:53 CST monitor
   check found runner PID `3347729` and adapter PID `3347738` still alive, one
   matching adapter process, Qwen3 embedding progress `35360/269711`, GPU
   active, disk about `12.35G` free / `94%` used, no failure markers, and no
   final scores/provenance/tables yet. A 2026-06-05 06:03 CST follow-up found
   the same PIDs alive and unique, Qwen3 embedding progress `79528/269711`,
   disk still about `12.35G` free / `94%` used, and no completion/failure
   markers or final scores/provenance/tables. A 2026-06-05 06:15 CST follow-up
   found Qwen3 embedding progress `128944/269711`, the same PIDs alive and
   unique, disk still about `12.35G` free / `94%` used, and no
   completion/failure markers or final scores/provenance/tables. A
   2026-06-05 06:25 CST follow-up found Qwen3 embedding progress
   `172352/269711`, the same PIDs alive and unique, disk still about `12.35G`
   free / `94%` used, and no completion/failure markers or final
   scores/provenance/tables. A 2026-06-05 06:33 CST follow-up found Qwen3
   embedding progress `209840/269711`, the same PIDs alive and unique, disk
   still about `12.35G` free / `94%` used, and no completion/failure markers or
   final scores/provenance/tables. This is active monitor-only evidence, not a
   completed row.
   A 2026-06-05 07:02 CST follow-up found Qwen3 embedding complete
   (`269711/269711`) and official RLMRec training started, latest observed at
   `epoch=110` with train loss `1.519527`. The same runner/adapter PIDs were
   alive and unique, GPU was active, final RLMRec evidence remained
   placeholder-only, and no completion/failure marker or final
   scores/provenance/tables existed. Disk fell below the warning line to about
   `8.6G` free / `96%` used as the active adapter grew to about `5.2G`; after
   verifying server-final and local-light evidence for completed Tools IRLLRec,
   Codex removed only its server-side `predictions/rank_predictions.jsonl`
   with sha256 manifest
   `outputs/summary/tools_irllrec_prediction_cleanup_manifest_20260605.sha256`.
   Scores, provenance, audits, imported tables, and model were preserved; disk
   recovered only to about `9.3G` free, still below the 10G warning threshold.
   This is active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 08:01 CST follow-up confirmed the active Tools RLMRec run is
   on the expected official default `3000`-epoch path (`--rlmrec_epochs=3000`
   / trainer `--epochs=3000`) and had reached epoch `590/3000` with train loss
   `1.509551`. Runner PID `3347729` and adapter PID `3347738` were alive and
   unique, no completion/failure/final-artifact markers existed, final evidence
   remained placeholder-only, and disk remained warning-level at about `9.3G`
   free / `96%` used. A repeat large-file/cache/temp/archive audit found no
   safe deletion candidate. This remains active monitor-only evidence, not a
   completed RLMRec row.
   A 2026-06-05 08:46 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the first material monitor
   checkpoint at epoch `1000/3000` and train loss `1.505931`. Final RLMRec
   evidence still remained placeholder-only, with no final scores/provenance,
   score audit, imported tables, predictions, completion marker, OOM/no-space,
   killed, traceback, or error marker. Disk remained warning-level at about
   `9.3G` free / `96%` used and no new safe cleanup candidate was identified.
   This remains active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 09:45 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the halfway monitor
   checkpoint on the default `3000`-epoch path: latest observed epoch
   `1510/3000` had train loss `1.506936`, after epoch `1500` loss
   `1.507642`. Final RLMRec evidence still remained placeholder-only (`4.0K`)
   with no final scores, provenance, score audit, imported tables,
   predictions, completion marker, OOM/no-space, killed, traceback, or error
   marker. Disk remained warning-level at about `9.3G` free / `96%` used; a
   repeat large-file/cache/temp/archive and prediction cleanup audit found no
   safe meaningful deletion candidate. This remains active monitor-only
   evidence, not a completed RLMRec row.
   A 2026-06-05 10:36 CST follow-up found the same runner/adapter PIDs alive
   and unique, with exactly one matching RLMRec adapter and zero matching
   IRLLRec adapters. Official RLMRec training crossed the two-thirds monitor
   checkpoint on the default `3000`-epoch path at epoch `2000/3000` with train
   loss `1.506144`. Final RLMRec evidence still remained placeholder-only
   (`4.0K`) with no final scores, provenance, score audit, imported tables,
   predictions, completion marker, OOM/no-space, killed, traceback, or error
   marker. Disk remained warning-level at about `9.3G` free / `96%` used; a
   fresh cleanup audit found no safe deletion target because caches/tmp were
   small, no temp/archive/part/core files existed, and the largest files were
   active RLMRec intermediates, protected task splits, retained completed
   checkpoints/evidence, or historical C-CRP/fusion summaries. This remains
   active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 11:49 CST follow-up found the same runner/adapter PIDs alive
   and unique, with official RLMRec training past the five-sixths monitor
   checkpoint at epoch `2610/3000`; the epoch `2600` loss was `1.505511` and
   the latest epoch `2610` loss was `1.504630`. Final RLMRec evidence still
   remained placeholder-only (`4.0K`) with no final scores, provenance, score
   audit, imported tables, predictions, completion marker, OOM/no-space,
   killed, traceback, or error marker. Disk remained warning-level at about
   `9.3G` free / `96%` used; a repeat large-file/cache/temp/archive and
   prediction cleanup audit found no safe deletion target because visible
   reclaimable candidates were active RLMRec intermediates, protected task
   splits, retained completed checkpoints/evidence, or the legacy Electronics
   ELMRec prediction JSONL without server-final/local-light deletion proof.
   This remains active monitor-only evidence, not a completed RLMRec row.
   A 2026-06-05 12:49 CST completion/gate pass then recorded Tools
   `rlmrec_graphcl` as the sixth gated Tools official row. The wrapper log
   reached epoch `3000/3000` with final train loss `1.505858`, wrote
   `implementation_status=official_completed`, `blockers=0`, saved predictions
   and tables, reported `score_coverage_rate=1.000000`, and ended with
   `DONE rlmrec_graphcl on tools` / `All baseline runs complete`. Server-final
   audit, server large-artifact manifest, lightweight local sync, and
   local-light audit all passed. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0784 / 0.1354 / 0.2465`, NDCG@5/10/20
   `0.05017501611537314 / 0.06838865570840932 / 0.09599330874161652`, and MRR
   `0.07220064580885768`; `scores.csv` has `1,010,001` lines, predictions had
   `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local lightweight
   package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`.
   Post-gate cleanup removed only the server prediction JSONL and completed
   intermediate adapter directory with manifests
   `prediction_deletion_manifest.json`,
   `outputs/summary/tools_rlmrec_completed_adapter_cleanup_manifest_20260605.sha256`,
   and `outputs/summary/tools_rlmrec_completed_adapter_cleanup_du_20260605.txt`,
   preserving final scores, provenance, audits, imported tables, model,
   manifests, and local evidence. Disk recovered to about `15G` free / `93%`
   used. Tools is now 6/8 official rows gated and still needs `llm2rec_sasrec`,
   `llmesr_sasrec`, and the Tools domain/comparison/paired-test gates.
   A first Tools `llm2rec_sasrec` launch at 2026-06-05 13:02 CST then failed
   before a stable adapter process or score file existed. The traceback was a
   validation-export exact-one-positive failure for
   `source_event_id='AGEY75LYLXUAHG3KW5KF5ICMKA4A'`. Audit found the Tools
   valid `candidate_items.csv` was truncated: original sha256
   `4302712bb7dbe0a8cfde99b0a2727c8de0818d250b65e7cc3bc0b8ad01fa6f2b`,
   `587252` logical rows, partial final event, and a malformed user-id-only
   row. The CSV was rebuilt from `ranking_valid.jsonl` and independently
   validated as PASS with `1,010,000` rows, `10,000` events, `101`
   candidates/event, and exactly one positive/label per event. Manifests are
   `outputs/summary/tools_valid_candidate_items_repair_20260605T051943Z.json`
   and
   `outputs/summary/tools_valid_candidate_items_repair_validation_20260605T0520Z.json`.
   No Tools LLM2Rec row is active or table-eligible; disk is about `14G` free /
   `93%` used, so relaunch is pending a fresh preflight and storage-margin
   decision.
   A 2026-06-05 13:46 CST relaunch checkpoint then started exactly one Tools
   `llm2rec_sasrec` row after sidecar ARIS review of storage and duplicate-run
   risk. Runner PID `3413921`, adapter PID `3413930`, log
   `baselines_new_domains_tools_llm2rec_20260605_134355.log`, heartbeat
   `monitor-tools-llm2rec`, and launch snapshot
   `outputs/summary/tools_llm2rec_launch_monitor_20260605.json`. The snapshot
   found one adapter process, no failure markers, active Qwen3 embedding
   progress around `[hf_mean_pool] encoded 3888/345622`, GPU about `95%` /
   `16089 MiB`, adapter directory `1.1G`, final output placeholder-only, and
   disk about `12G` free / `94%` used. This is active monitor-only evidence,
   not a completed or table-eligible row.
   A 2026-06-05 13:57 CST follow-up found the same runner/adapter alive, cleared
   three stale diagnostic grep/bash processes that were causing false duplicate
   matches, and observed Qwen3 embedding progress around
   `[hf_mean_pool] encoded 53296/345622`; disk remained about `12G` free /
   `94%` used and no failure/completion marker was present. Snapshot:
   `outputs/summary/tools_llm2rec_monitor_checkpoint_20260605_1355.json`.
   A 2026-06-05 15:23 CST monitor confirmed the relaunch failed due disk
   exhaustion rather than completing. Runner PID `3413921` and adapter PID
   `3413930` had exited, GPU was idle, and
   `baselines_new_domains_tools_llm2rec_20260605_134355.log` ended with
   `OSError: [Errno 28] No space left on device` after Qwen3 embedding reached
   `345622/345622`. The LLM2Rec training log showed `torch.save` failed while
   saving the SASRec checkpoint, so the partial checkpoint was corrupt. There
   is no `fairness_provenance.json`, no `scores.csv`, and no score audit; the
   row is failed and not table-eligible. Emergency cleanup removed only failed
   or already-repaired artifacts: the corrupt validation CSV backup, failed
   adapter-side embedding copy, regenerable adapter CSVs, and corrupt partial
   checkpoint. It preserved the reusable upstream embedding at
   `/home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`.
   Post-cleanup disk is about `8.1G` free / `96%` used, still below the `10G`
   monitor threshold. Do not import this row and do not start the next baseline
   until additional audited space is freed; the recovery relaunch should pass
   the preserved path through `--llm2rec_item_embedding_path`.
   A 2026-06-05 16:05 CST recovery/storage checkpoint removed the completed
   Sports and Toys LLM2Rec full checkpoints with sha256/size manifests after a
   GPT-5.5 xhigh sidecar ARIS storage audit and live evidence checks. The
   manifests are
   `outputs/summary/sports_llm2rec_checkpoint_deletion_manifest_20260605.json`
   and `outputs/summary/toys_llm2rec_checkpoint_deletion_manifest_20260605.json`.
   Sports/Toys LLM2Rec scores, provenance, score audits, run summaries,
   imported tables, server-final audits, and local-light packages were
   preserved; only the full `.pth` checkpoints were deleted. The wrapper now
   supports `LLM2REC_ITEM_EMBEDDING_PATH_OVERRIDE` and the server copy passed
   `bash -n`. Exactly one Tools LLM2Rec recovery launched at 2026-06-05
   15:59 CST with runner PID `3423029`, adapter PID `3423037`, official
   training PID `3423221`, log
   `baselines_new_domains_tools_llm2rec_recovery_20260605_155904.log`, and
   heartbeat `monitor-tools-llm2rec-recovery`. Its adapter command includes
   `--llm2rec_item_embedding_path /home/ajifang/projects/LLM2Rec/item_info/ToolsSameCandidate100Neg/pony_qwen3_8b_title_item_embs.npy`,
   so Qwen3 embedding is reused. At the 16:05 CST snapshot it was active around
   epoch `30+`, had saved a Tools checkpoint, and disk was about `10G` free /
   `95%` used. This is active monitor-only evidence, not a completed row.
   A 2026-06-05 17:04 CST monitor/storage checkpoint confirmed the recovery
   row was still active with runner PID `3423029`, adapter PID `3423037`, and
   training PID `3423221`; duplicate counts were exactly one LLM2Rec adapter
   and one `ToolsSameCandidate100Neg` training child. No final provenance,
   `scores.csv`, or imported ranking table existed yet. Training-log
   validation metrics were still improving at epoch `330`, latest observed
   HR@5/10/20
   `0.26969999074935913 / 0.3407999873161316 / 0.414900004863739` and
   NDCG@5/10/20
   `0.20475752651691437 / 0.22773440182209015 / 0.2465011030435562`; these
   are validation-only progress numbers, not final table metrics. Disk briefly
   crossed the monitor threshold at about `9.9G` free / `95%` used. A
   large-file/cache/temp/archive audit found no safe project-output deletion:
   large visible artifacts were active, final evidence/model artifacts,
   same-candidate task data, or an older Electronics ELMRec prediction whose
   server-final audit is `ok=false`. After approval, only conda package cache
   cleanup was run; no project outputs, evidence, checkpoints, embeddings, task
   splits, or other projects were deleted. Post-cleanup `/` was about `11G`
   free (`10,773,983,232` bytes) / `95%` used.
   A 2026-06-05 18:07-18:11 CST follow-up found the same recovery row still
   active with runner PID `3423029`, adapter PID `3423037`, and training PID
   `3423221`; duplicate counts remained exactly one LLM2Rec adapter and one
   `ToolsSameCandidate100Neg` training child. No final provenance,
   `scores.csv`, score audit, or imported ranking table existed yet.
   Training-log validation progress reached epoch `715`, latest observed
   validation HR@5/10/20
   `0.8866999745368958 / 0.9272000193595886 / 0.9584000110626221` and
   NDCG@5/10/20
   `0.8021516799926758 / 0.8153268098831177 / 0.8232611417770386`; these are
   validation-only progress numbers, not final table metrics. Disk was only
   about 34 MiB above the 10 GiB guard (`10,773,143,552` bytes free), so only
   non-project cache cleanup was run: conda reported no unused
   tarballs/packages/tempfiles, pip cache purge removed `17.0 MB`, and npm
   cache clean was attempted. No project outputs, evidence, checkpoints,
   embeddings, task splits, or other projects were deleted. The post-cleanup
   safety check confirmed the same PIDs alive, duplicate counts `1/1`, no final
   evidence, and disk `10,820,177,920` bytes free / `95%` used.
   At 2026-06-05 18:38 CST, Tools `llm2rec_sasrec` completed normally under
   the recovery run. The wrapper reported `implementation_status=official_completed`,
   `[2026-06-05 18:38:04] DONE llm2rec_sasrec on tools`, and `=== All baseline
   runs complete ===`; tracked PIDs exited and duplicate counts were `0/0`.
   Server-final audit passed with `ok=true`, `failures=[]`, `warnings=[]`;
   server large-artifact manifest passed; local-light sync and local-light
   audit both passed. Local lightweight evidence is under
   `outputs/baselines/official_adapters/tools_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
   Full metrics over `10,000` users and `101` candidates are HR@5/10/20
   `0.0957 / 0.1625 / 0.2954`, NDCG@5/10/20
   `0.060227320850546905 / 0.08147852827156639 / 0.11481218118869342`, and MRR
   `0.08101396652891538`. Score audit/import coverage passed with
   `candidate_rows=1010000`, `score_rows=1010000`, `matched_keys=1010000`,
   no missing/extra/duplicate keys, finite scores `1010000`, and
   `score_coverage_rate=1.0`. Row counts: `scores.csv` `1,010,001` lines,
   `predictions/rank_predictions.jsonl` `10,000` lines before post-gate
   cleanup, and `tables/ranking_eval_records.csv` `10,001` lines. Provenance
   has `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, no test-set model selection, and no extra
   baseline tuning. Post-gate cleanup preserved final scores, provenance,
   audits, imported tables, server-final audit, large-artifact manifest, SASRec
   checkpoint, upstream embedding, and compact adapter metadata. Because disk
   remained tight, only the server-side prediction JSONL was deleted with
   manifest
   `outputs/summary/tools_llm2rec_prediction_deleted_post_gate_20260605.{json,sha256}`
   and then the completed-row adapter staging CSVs were deleted with manifest
   `outputs/summary/tools_llm2rec_completed_adapter_staging_cleanup_20260605.{json,sha256}`.
   The deleted prediction had `10,000` lines, size `804,958,794` bytes, and
   sha256 `211a037a71020955e3488fdcd53f8d6710505bdad23e13c60e7a869d83e99148`;
   adapter staging cleanup removed `1,100,523,516` bytes. Post-cleanup disk is
   `11,772,899,328` bytes free / `95%` used. Tools is now 7/8 official rows
   gated; the remaining Tools official row is `llmesr_sasrec`.
   After a fresh no-process/GPU/disk/duplicate-output preflight, the final
   Tools official row `llmesr_sasrec` launched at 2026-06-05 18:52 CST with
   runner PID `3440278`, adapter PID `3440287`, log
   `baselines_new_domains_tools_llmesr_20260605_185250.log`, pidfile
   `outputs/summary/tools_llmesr_launch_20260605_185250.pid`, launch snapshot
   `outputs/summary/tools_llmesr_launch_monitor_20260605_185250.json`, and
   heartbeat `monitor-tools-llm-esr`. The 18:57 CST stable follow-up showed
   exactly one LLM-ESR adapter process, no training child yet, GPU `99%` /
   `16091 MiB`, embedding progress about `11288/269711`, no failure markers,
   final output placeholder-only, and adapter dir about `1005M`. Because
   adapter staging pushed disk below the strict 10 GiB guard, only non-project
   caches were cleaned (`.cache`, `.codex/.tmp`, Chrome GPU cache, and inactive
   `.cursor-server/bin` after verifying no Cursor process). No project
   evidence, checkpoints, embeddings, task splits, configs, source code, or
   other project files were deleted. Cache-cleanup record:
   `outputs/summary/tools_llmesr_launch_cache_cleanup_20260605.txt`.
   Post-cleanup disk was `10,866,053,120` bytes free / `95%` used. This
   LLM-ESR row is active monitor-only evidence until final score/provenance
   gates pass.
   A 2026-06-05 19:04-19:06 CST monitor/cache checkpoint found the same row
   still active with runner PID `3440278` and adapter PID `3440287`, duplicate
   counts exactly one adapter and zero training child, GPU about `91-96%` /
   `16189 MiB`, embedding progress about `47824/269711`, no final evidence,
   and no failure/OOM/no-space markers. Disk dipped just below the strict 10
   GiB guard (`10,726,035,456` bytes free), so only non-project Chrome
   cache/GPU cache and VSCode-server logs were cleared. No project outputs,
   evidence, checkpoints, embeddings, task splits, configs, source code, or
   other project files were deleted. Cleanup record:
   `outputs/summary/tools_llmesr_monitor_cache_cleanup_20260605_1904.txt`.
   Post-cleanup disk was `10,757,963,776` bytes free / `95%` used.
   A 2026-06-05 19:09-19:13 CST disk-emergency checkpoint found the active
   Tools `llmesr_sasrec` row still alive with runner PID `3440278` and adapter
   PID `3440287`, duplicate counts exactly one adapter and zero training child,
   GPU about `96%` / `16213 MiB`, embedding progress from about `59448/269711`
   to `77056/269711`, and no final evidence or failure markers. Disk was below
   the strict 10 GiB guard (`10,633,687,040` bytes free). No safe completed
   prediction deletion existed, and conda/pip/npm cache cleanup freed `0` bytes
   (`outputs/summary/tools_llmesr_monitor_conda_cache_cleanup_20260605_1909.txt`).
   After explicit destructive-action approval, the completed Tools LLM2Rec
   SASRec checkpoint was deleted only after verifying gated evidence and
   writing manifests:
   `outputs/summary/tools_llm2rec_completed_checkpoint_cleanup_for_llmesr_disk_20260605.{json,sha256}`.
   The deleted checkpoint was `5,665,876,357` bytes, sha256
   `8ad7ce0316befeb8ee6b3482546ffe3e301e42e9a6b1e10ee608689ea5ece414`.
   Final LLM2Rec scores, provenance, score audit, run summary, imported
   tables, server-final audit, large-artifact manifest, and local-light package
   were preserved; disk recovered to `16,293,425,152` bytes free / `92%` used.
   A 2026-06-05 19:55-20:01 CST monitor then found the active Tools
   `llmesr_sasrec` row still alive and unique with runner PID `3440278` and
   adapter PID `3440287`. The Qwen3 embedding pass completed
   (`269711/269711`) and the row entered official LLM-ESR training at
   `[llmesr] epoch=1 train_loss=1.398057`. There was still no final
   `scores.csv`, provenance, score audit, run summary, imported ranking table,
   or prediction JSONL, so this remains active monitor-only evidence. Disk was
   tight at `11,804,352,512` bytes free / `95%` used; a read-only storage audit
   found the main new pressure is the active adapter embedding
   `outputs/baselines/paper_adapters/tools_large10000_100neg_llmesr_official_adapter/llm_esr/handled/itm_emb_np.pkl`
   (`4.12G`), which is not safe to delete while training is active. Other large
   visible files are protected completed models/checkpoints/task splits or
   non-deletable legacy prediction evidence. No cleanup was performed.
   At 2026-06-05 21:19 CST, Tools `llmesr_sasrec` completed as the eighth
   Tools official row with wrapper markers `implementation_status=official_completed`,
   `blockers=0`, `DONE llmesr_sasrec on tools`, and
   `All baseline runs complete`. Server-final audit, server large-artifact
   manifest, local-light sync, and local-light audit all passed. Full metrics
   over 10,000 users and 101 candidates are HR@5/10/20
   `0.0711 / 0.1270 / 0.2219`, NDCG@5/10/20
   `0.042728964614829223 / 0.060602849768892623 / 0.08433244535733923`, and
   MRR `0.06334161303438132`; row counts are `scores.csv` `1,010,001`,
   prediction JSONL `10,000` before cleanup, and
   `tables/ranking_eval_records.csv` `10,001`. The local lightweight package is
   `outputs/baselines/official_adapters/tools_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   Post-gate cleanup removed only the completed server prediction JSONL and
   two adapter staging CSVs with manifest
   `outputs/summary/tools_llmesr_post_gate_cleanup_20260605.sha256`, preserving
   final scores, provenance, audits, imported tables, server-final certificate,
   `llmesr_official_model.pt`, and upstream embedding artifacts. The valid
   pre-cleanup Tools IRLLRec server-final certificate was restored from the
   local-light package after a post-cleanup failed audit had overwritten the
   server copy; the failed audit was preserved as
   `server_final_evidence_audit_post_cleanup_failed_20260605.json`.
   Tools C-CRP raw scores were imported into
   `outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
   with exact `score_coverage_rate=1.0`. The post-cleanup Tools domain gate
   `outputs/summary/tools_official_ccrp_gate_post_cleanup_20260605.{json,csv}`
   records `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
   `gate_ok=true`. Tools C-CRP metrics are HR@5/10/20
   `0.1937 / 0.2696 / 0.3931`, NDCG@5/10/20
   `0.14186375906171483 / 0.16611553052793934 / 0.19703986741872317`, and MRR
   `0.15585924577949772`. The comparison/statistical package
   `outputs/summary/tools_official_ccrp_20260605_*` records
   `claim_gate=tools_domain_pass`: C-CRP is observed-best on all seven metrics,
   all 56 C-CRP-vs-8-official paired tests have positive deltas and are
   Holm-significant, `min_delta=0.0294`, `min_ci_low=0.0173`, and
   `max_holm_p_value=9.7172307634557e-07`. After local table sync and the
   passed domain/comparison gates, the server-only imported C-CRP prediction
   JSONL was removed with
   `outputs/tools_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/prediction_deletion_manifest.json`;
   the post-cleanup gate still passes by certificate. This is a Tools-domain
   pass, not final paper readiness.
   A compact four-new-domain ARIS gate certificate was then built locally as
   `outputs/summary/new_domains_official_ccrp_cross_domain_20260605_*`. It
   covers Sports, Toys, Home, and Tools and records passing blocking checks:
   8/8 official-code-level baseline rows per domain, exact same-candidate gate
   status, C-CRP rank 1 on all seven metrics, and 56/56 positive
   Holm-significant C-CRP-vs-official paired tests per domain. The safe claim
   is limited to the four new 10k-user/101-candidate same-candidate domains.
   The certificate explicitly does not support paper-ready SOTA,
   full-catalog recommender SOTA, large-effect wording, universal winner
   wording, or a proven uncertainty-mechanism claim. It also records
   non-blocking local artifact gaps for Sports/Toys C-CRP raw event restat, so
   it is a comparison-gate summary rather than a self-contained local raw
   reproduction package.
   Home
   LLM2Rec completed at 2026-06-04 09:49 CST after a disk-full partial-copy
   recovery and passed score audit/import, server-final audit, server
   large-artifact manifest, lightweight local sync, and local-light audit.
   Full metrics over 10,000 users and 101 candidates are HR@5/10/20
   `0.0577 / 0.1101 / 0.2153`, NDCG@5/10/20
   `0.034207889197971464 / 0.05094946457092549 / 0.07719596686909931`, and
   MRR `0.0563865859396318`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llm2rec_sasrec_official_qwen3base_same_candidate/`.
   The post-gate server prediction deletion manifest is
   `outputs/summary/home_llm2rec_prediction_deleted_after_gates_20260604.sha256`.
   The completed LLM2Rec intermediate adapter and upstream symlink were then
   removed with sha256 cleanup manifest
   `outputs/summary/home_llm2rec_completed_adapter_cleanup_manifest_20260604.sha256`,
   recovering `/` to about `12G` free / `95%` used without touching final
   evidence. Home `llmesr_sasrec` launched at 2026-06-04 10:14 CST as the
   eighth Home row and completed at 2026-06-04 13:09 CST with
   `implementation_status=official_completed`, `blockers=[]`, exact
   `score_coverage_rate=1.0`, score audit/imported tables, server-final audit,
   server large-artifact sha256 manifest, lightweight local sync, and
   local-light audit all passing. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0621 / 0.1163 / 0.2139`, NDCG@5/10/20
   `0.037993209299003045 / 0.055376101596196485 / 0.0797502336556021`, and
   MRR `0.059737054548523474`; `scores.csv` has `1,010,001` lines,
   predictions had `10,000` lines before post-gate deletion, and
   `tables/ranking_eval_records.csv` has `10,001` lines. The local
   lightweight package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   The post-gate prediction/staging cleanup manifest is
   `outputs/summary/home_llmesr_post_gate_cleanup_20260604.sha256`, and final
   scores/provenance/audits/imported tables/`llmesr_official_model.pt` were
   preserved. The Home C-CRP raw scores were then imported into
   `outputs/home_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate`
   with exact coverage and NDCG@10 `0.13239420796539653`. The Home domain gate
   wrote `outputs/summary/home_official_ccrp_gate_20260604.{json,csv}` with
   `official_ok_count=8`, `official_all_ok=true`, `ccrp_ok=true`, and
   `gate_ok=true`. The comparison/statistical package
   `outputs/summary/home_official_ccrp_20260604_*` records C-CRP rank 1 on all
   seven metrics, all 56 C-CRP-vs-official paired tests positive and
   Holm-significant, `min_delta=0.0336`, `min_ci_low=0.0216`, and
   `max_holm_p_value=1.0216927255359559e-08`. This is a Home-domain pass only,
   not a paper-wide SOTA claim. Toys
   `llmesr_sasrec` completed at 2026-06-02 18:59 CST after a disk-full
   recovery as `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0`. Full metrics over 10,000 users and 101
   candidates are HR@5/10/20 `0.0637 / 0.1172 / 0.2203`,
   NDCG@5/10/20
   `0.037504900117522603 / 0.05456849726033091 / 0.08036871527121744`, and MRR
   `0.05844977379835533`; row counts are `scores.csv` `1,010,001`,
   predictions `10,000`, and `tables/ranking_eval_records.csv` `10,001`.
   Server-final audit, lightweight sync, and local-light audit passed. The
   local lightweight package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_llmesr_sasrec_official_qwen3base_same_candidate/`.
   The final toys domain gate and comparison/statistical gate are under
   `outputs/summary/toys_official_gate_final_20260602_1900.*` and
   `outputs/summary/toys_official_ccrp_20260602_1900_*`. C-CRP is rank 1 on
   all seven toys metrics; all 56 paired tests are positive and
   Holm-significant. Current server disk recovered from about `5.9G` to `17G`
   free after removing only verified completed upstream staging data for
   sports/toys LLMEmb and LLM-ESR, with cleanup manifest
   `outputs/summary/upstream_completed_sports_toys_llmemb_llmesr_cleanup_manifest_20260602.sha256`.
   Final official evidence and local lightweight packages were preserved.
   Home `proex_profile` launched at 2026-06-02 19:45 CST with runner PID
   `3004208`, adapter PID `3004218`, and log
   `baselines_new_domains_home_proex_20260602_1950.log`. It completed at
   2026-06-02 22:00 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0606/0.1177/0.2296`,
   NDCG@5/10/20=`0.03662857786324662/0.054867449700296195/0.08290060869107069`,
   and MRR=`0.05933326491258513`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed intermediate adapter was removed with cleanup
   manifest
   `outputs/summary/home_proex_completed_adapter_cleanup_manifest_20260602.sha256`,
   recovering disk from about `8.2G` to `16G` free without touching final
   evidence. Home `promax_profile` launched at 2026-06-02 22:14 CST with
   runner PID `3026043`, adapter PID `3026052`, and log
   `baselines_new_domains_home_promax_20260602_2215.log`. It completed at
   2026-06-03 02:53 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0514/0.1019/0.2076`,
   NDCG@5/10/20=`0.030788292596664168/0.04691808776215203/0.07326077825489297`,
   and MRR=`0.053474908740382465`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed intermediate adapter was removed with cleanup
   manifest
   `outputs/summary/home_promax_completed_adapter_cleanup_manifest_20260602.sha256`,
   recovering disk from about `7.5G` to `15G` free without touching final
   evidence. Home `elmrec_graph` launched at 2026-06-03 03:02 CST with runner
   PID `3061705`, adapter PID `3061714`, and log
   `baselines_new_domains_home_elmrec_20260603_0302.log`. It completed at
   2026-06-03 05:47 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0509/0.1021/0.2018`,
   NDCG@5/10/20=`0.029717257242599254/0.0460440741915887/0.0708856096588022`,
   and MRR=`0.05195852255617441`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   At that 2026-06-03 05:47 CST checkpoint, home had 3/8 completed official
   rows. At the 2026-06-03 05:48 CST
   checkpoint no Pony/C-CRP/baseline Python process was active, GPU was idle,
   and disk was tight at about `6.5G` free. The completed ElmRec intermediate
   adapter was removed after exact realpath checks and a 16-file sha256 cleanup
   manifest
   `outputs/summary/home_elmrec_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, and final scores,
   provenance, audits, predictions, imported tables, model, and local
   lightweight evidence were preserved. Disk recovered to about `14G` free.
   Home `llmemb` launched at 2026-06-03 06:08 CST as the fourth home row. The
   first run reached exact score export but failed during checkpoint save at
   `100%` disk, so the orphan score was quarantined and not imported as
   official evidence. After removing only failed-run/generated staging storage
   and patching LLMEmb handled embedding staging to symlink the large
   `itm_emb_np.pkl`, the true symlink rerun launched at 2026-06-03 09:16 CST
   with log `baselines_new_domains_home_llmemb_symlink_rerun_20260603_0920.log`.
   It completed as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.1079/0.1856/0.3169`,
   NDCG@5/10/20=`0.06899578967097944/0.09390612986107003/0.12674255822842873`,
   and MRR=`0.09012268660291177`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. At that checkpoint, Home had 4/8
   completed official rows before IRLLRec. The completed LLMEmb intermediate adapter was then removed
   with cleanup manifest
   `outputs/summary/home_llmemb_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, and disk recovered to
   about `7.1G` free. Home `irllrec_intent` then launched at 2026-06-03 13:55
   CST as the fifth home row after a clean preflight; runner PID `3147646`,
   adapter PID `3147655`, log
   `baselines_new_domains_home_irllrec_20260603_1355.log`. At the 13:57 CST
   checkpoint it was CPU-side active with about `6.1G` free and no final
   scores/provenance/imported tables, so no fifth home result row was eligible
   at that checkpoint. At the 2026-06-03 17:21 CST heartbeat, the active row had completed
   Qwen3 embedding and reached official training epoch `1220`, but disk fell to
   about `30M` free. Emergency cleanup removed only the completed Toys LLM2Rec
   intermediate adapter and two server-only prediction JSONLs from already
   gated toys rows (`rlmrec_graphcl` and `irllrec_intent`) after sha256
   manifests; scores, provenance, audits, imported tables, models, and
   local-light packages were preserved. Disk recovered to about `2.0G`, and the
   active home IRLLRec process continued to epoch `1280` by 17:27 CST. The
   cleanup manifests are
   `outputs/summary/toys_llm2rec_completed_adapter_cleanup_manifest_20260603_irllrec_disk.sha256`
   and
   `outputs/summary/toys_predictions_deleted_for_home_irllrec_disk_20260603.sha256`.
   At the 17:43 CST follow-up, disk had slipped to about `1.7G`, so the
   already gated Toys ProEx server-only prediction JSONL was removed after
   confirming server-final and local-light audits; scores, provenance, audits,
   imported tables, models, and local-light packages were preserved. The
   additional manifest is
   `outputs/summary/toys_proex_prediction_deleted_for_home_irllrec_disk_20260603.sha256`;
   disk recovered to about `2.5G`, and active training reached epoch `1480` by
   17:45 CST. At the 18:08 CST follow-up, home IRLLRec remained active through
   epoch `1740` with no fatal/OOM/no-space markers, but disk was still tight.
   After confirming sports ProMax server-final audit `ok=true` and local-light
   audit `ok=true`, only the already gated server-side Sports ProMax prediction
   JSONL was removed with sha256 manifest
   `outputs/summary/sports_promax_prediction_deleted_for_home_irllrec_disk_20260603.sha256`.
   Sports ProMax scores, provenance, audits, imported tables, model, and
   local-light package were preserved; disk recovered to about `3.1G`.
   At the 18:45 CST continuation check, Home IRLLRec was still active through
   epoch `2140` with no fatal/OOM/no-space markers, but disk remained tight.
   After confirming server-final audit `ok=true`, local-light audit `ok=true`,
   `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0` for completed Home `proex_profile`,
   `promax_profile`, `elmrec_graph`, and `llmemb`, only those four rows'
   server-side `predictions/rank_predictions.jsonl` files were removed under
   the documented disk-pressure exception. The sha256 manifest is
   `outputs/summary/home_completed_predictions_deleted_for_irllrec_disk_20260603.sha256`.
   Their scores, provenance, audits, imported tables, models, and local-light
   packages were preserved; disk recovered to about `6.0G`, and the active Home
   IRLLRec runner/adapter remained active at that checkpoint. Home
   `irllrec_intent` completed at
   2026-06-03 20:05 CST as `implementation_status=official_completed` with
   `blockers=[]`, exact `score_coverage_rate=1.0`, server-final audit PASS,
   lightweight sync PASS, and local-light audit PASS. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0821/0.1443/0.2878`,
   NDCG@5/10/20=`0.05108975290090002/0.07089752436733526/0.10662011242627113`,
   and MRR=`0.07424325424843974`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. After final evidence and local
   backup passed, the completed IRLLRec intermediate adapter was removed with
   cleanup manifest
   `outputs/summary/home_irllrec_completed_adapter_cleanup_manifest_20260603.sha256`;
   a post-cleanup server-final audit remained `ok=true`, final scores,
   provenance, audits, predictions, imported tables, model, and local-light
   package were preserved, and disk recovered from about `4.7G` to `12G` free.
   Home `rlmrec_graphcl` launched at 2026-06-03 20:28 CST as the sixth home row
   after fresh process/GPU/disk and duplicate-output preflight, then completed
   at 2026-06-04 06:49 CST as `implementation_status=official_completed`,
   `blockers=[]`, and `score_coverage_rate=1.0`. The launch log was
   `baselines_new_domains_home_rlmrec_20260603_2028.log`; the final training
   line reached `[rlmrec-official] epoch=3000 train_loss=1.546134`. RLMRec
   passed server-final evidence audit, server large-artifact sha256 manifest,
   lightweight local sync, and local-light audit. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0685/0.1268/0.2451`,
   NDCG@5/10/20=`0.04126795456098191/0.059869974785318684/0.08932211717259728`,
   and MRR=`0.06397404670906748`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/home_large10000_100neg_rlmrec_graphcl_official_qwen3base_same_candidate/`;
   server-only large artifacts remain covered by
   `server_large_artifact_manifest.sha256`. Home now has 6/8 completed official
   rows; remaining Home rows are `llm2rec_sasrec` and `llmesr_sasrec`. Home is
   still not domain-gate eligible until all eight official rows and C-CRP
   imported evidence pass the same checks. Before launching Home LLM2Rec, the
   completed non-final RLMRec intermediate adapter was removed after cleanup
   manifest
   `outputs/summary/home_rlmrec_completed_adapter_cleanup_manifest_20260604.sha256`,
   exact realpath check, and post-cleanup RLMRec server-final audit PASS; final
   RLMRec evidence was preserved. Disk recovered to about `19G` free / `91%`
   used. Home `llm2rec_sasrec` then launched at 2026-06-04 07:19 CST as the
   seventh home row with runner PID `3236678`, adapter PID `3236688`, and log
   `baselines_new_domains_home_llm2rec_20260604_071902.log`. At the first
   stable check it was in Qwen3 `hf_mean_pool` embedding at about `944/568891`,
   GPU was about `96%`, and `/` was about `18G` free / `91%` used. It is not
   table-eligible until final score/provenance/import/server-final,
   large-artifact manifest, local-light sync, local-light audit, and full
   metric/row-count gates pass.
   Historical sports run record: sports started from
   `baselines_new_domains_sports.log` with runner PID `2794722`; the active
   child at the 2026-05-31 22:32 CST checkpoint was sports `llmemb` PID
   `2794731`. It had completed the `hf_mean_pool` embedding pass
   (`233470/233470`) and entered `llmemb-sasrec` training, with log lines
   through epoch 35. By the 2026-05-31 22:52 CST checkpoint,
   `llmemb-sasrec` had completed 200 epochs and the `llmemb` training phase
   had reached epoch 175. GPU was about `83%`, `16301 MiB / 49140 MiB`; disk
   was about `36G` free. At 2026-05-31 22:56 CST, sports `llmemb` completed
   as `official_completed` with `blockers=[]`, `score_coverage_rate=1.0`, and
   exact audit `audit_ok=True`. Metrics: HR@5/10/20=`0.2124/0.3384/0.4900`,
   NDCG@5/10/20=`0.1388527216/0.1795004215/0.2176868359`,
   MRR=`0.1538831336`, `n_users=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner then advanced
   to sports `proex_profile`. At 2026-06-01 00:25 CST, sports `proex_profile`
   also completed as `official_completed` with `blockers=[]`,
   `score_coverage_rate=1.0`, and exact audit `audit_ok=True`. Metrics:
   HR@5/10/20=`0.0821/0.1527/0.2777`,
   NDCG@5/10/20=`0.0516826556/0.0741722663/0.1054064715`,
   MRR=`0.0742689715`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. At 2026-06-01 03:04 CST,
   sports `promax_profile` completed as `official_completed` with
   `blockers=[]`, `score_coverage_rate=1.0`, and exact audit `audit_ok=True`.
   Metrics: HR@5/10/20=`0.0825/0.1387/0.2370`,
   NDCG@5/10/20=`0.0541847954/0.0721533411/0.0967593591`,
   MRR=`0.0741052747`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner advanced to
   sports `elmrec_graph`; at the 2026-06-01 03:05 CST checkpoint disk was
   about `20G` free (`90%` used), so storage is a close watch item but not yet
   a blocker. At 2026-06-01 04:37 CST, sports `elmrec_graph` completed as
   `official_completed` with `blockers=[]`, `score_coverage_rate=1.0`, and
   exact audit `audit_ok=True`. Metrics: HR@5/10/20=`0.0532/0.1054/0.2013`,
   NDCG@5/10/20=`0.0317045493/0.0483716358/0.0723504733`,
   MRR=`0.0537009851`, `sample_count=10000`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Lightweight local evidence is under
   `outputs/baselines/official_adapters/sports_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`;
   server-only large artifacts remain on the server. The runner advanced to
   sports `irllrec_intent`; at the 2026-06-01 04:38 CST checkpoint disk was
   about `15G` free (`93%` used), so storage is a close watch item but no
   space/OOM/CUDA failure has been observed.
   At 2026-06-01 05:31 CST, local lightweight evidence for the four completed
   sports official rows was expanded to include inspect provenance, JSON/TXT
   score audits, run summaries, imported summary/metric/coverage/exposure
   tables, and per-event `tables/ranking_eval_records.csv`; local/server
   size and line-count checks matched. The four completed methods' server
   `outputs/baselines/paper_adapters/` working directories were then removed
   after path checks, while final scores/provenance/audits/tables/predictions
   and checkpoints were preserved. Disk recovered to about `33G` free.
   At 2026-06-01 06:36 CST, the same runner remains active on sports
   `irllrec_intent` (child PID `2835275`), with log progress through epoch
   `1190/3000`, latest train loss `0.625393`, GPU about `98%`, `16295 MiB /
   49140 MiB`, disk about `29G` free (`85%` used), and no fatal/OOM/no-space
   markers. Four sports official rows are complete and locally backed up;
   `irllrec_intent`, `rlmrec_graphcl`, `llm2rec_sasrec`, and
   `llmesr_sasrec` are still incomplete. The local package gate
   `scripts/audit/main_audit_official_evidence_package.py` passed on all four
   completed lightweight evidence packages and is now the required check
   before recording future copied official evidence as backed up.
   At 2026-06-01 06:44 CST, the same gate was run in `server_final` mode on
   `pony-rec-gpu` for the four completed sports output directories; all four
   passed, including final server-side `scores.csv`, predictions, provenance,
   full metrics, coverage/exposure tables, and per-event evaluation records.
   The active `irllrec_intent` process was left untouched.
   At 2026-06-01 07:18 CST, sports `irllrec_intent` remained active under
   runner PID `2794722` and child PID `2835275`; the log had reached epoch
   `2040/3000`, latest train loss `0.624872`, GPU was about `75%` with
   `16295 MiB / 49140 MiB`, disk remained about `29G` free (`85%` used), and
   fatal/OOM/CUDA/no-space/runtime-error scans remained clean. No fifth sports
   official row is final yet, so no new paper-facing metric row has been added.
   To prevent local evidence omissions after future rows finish,
   `scripts/audit/main_sync_official_evidence_package.py` now performs
   allowlist-based lightweight sync and size/sha256 verification before the
   `local_light` evidence audit.
   At 2026-06-01 07:32 CST, sports `irllrec_intent` was still active at epoch
   `2320/3000` with latest train loss `0.625049`; GPU/memory were about
   `69%` and `16295 MiB / 49140 MiB`, disk remained about `29G` free, and
   fatal scans were clean. No final IRLLRec score/provenance/table package
   exists yet, so paper-facing sports official evidence remains four completed
   rows.
   At 2026-06-01 08:10 CST, sports `irllrec_intent` became the fifth completed
   sports official row: `implementation_status=official_completed`,
   `blockers=[]`, `score_coverage_rate=1.0`, and exact score coverage passed.
   Metrics are HR@5/10/20=`0.1573/0.2215/0.4016`,
   NDCG@5/10/20=`0.10642150916142634/0.12691703149297534/0.17128490034441315`,
   and MRR=`0.12444202662842994` over 10,000 users and 101 candidates.
   Server-final audit, local lightweight sync, and local-light audit all
   passed; the row is eligible as official sports evidence, pending the full
   eight-baseline sports comparison and paired tests. The runner then advanced
   to sports `rlmrec_graphcl`. At 2026-06-01 08:24 CST, RLMRec remained active
   under runner PID `2794722` and child PID `2851207`, generating Qwen
   embeddings at `57992/233470`; GPU was about `95%` with
   `16285 MiB / 49140 MiB`, disk was `32G` free, and fatal/OOM/CUDA/no-space
   scans remained clean. A read-only scan found one empty malformed output
   directory named
   `outputs/sports_large10000_100neg_TRAIN_METHODS_OVERRIDE=_official_qwen3base_same_candidate/`.
   It contains no files and is not evidence; the local runner now validates
   method tokens before creating output directories so a misquoted override
   cannot create another malformed method directory. At 2026-06-01 08:34 CST,
   RLMRec was still in Qwen embedding generation at `105376/233470` with no
   fatal/OOM/CUDA/no-space markers; its output directory still had only
   `inspect_fairness_provenance.json`, so no sixth sports official result is
   available yet. At 2026-06-01 09:05 CST, RLMRec reached Qwen embedding
   progress `233470/233470`, but child PID `2851207` was still active after
   embedding completion and the output directory still contained only
   `inspect_fairness_provenance.json`; no final `scores.csv`, provenance,
   audit, run summary, or imported metrics table exists yet, so the sports
   official evidence count remains five completed rows. At 2026-06-01 09:08
   CST, RLMRec entered official training and logged epoch 10 train loss
   `1.675038`; GPU was `100%` with `19943 MiB / 49140 MiB`, disk was `28G`
   free, and no final artifacts existed yet. The zero-degree graph
   normalization warning is recorded as non-fatal because the implementation
   maps infinite inverse degrees to `0.0`, but final eligibility still depends
   on successful score/provenance/audit/table outputs. At 2026-06-01 09:19
   CST, RLMRec was still training at epoch 140 with train loss `1.490221`,
   clean fatal/OOM/no-space scans, and only `inspect_fairness_provenance.json`
   in the output directory; the sports official evidence count remains five.
   At 2026-06-01 09:53 CST, RLMRec passed epoch 500 (`train_loss=1.480699`)
   and reached epoch 510 (`train_loss=1.482085`) with clean fatal/OOM/CUDA/
   no-space scans and stable `28G` free disk. No final `scores.csv`, final
   provenance, audit, run summary, or imported metrics table exists yet, so
   the sports official evidence count remains five. At 2026-06-01 10:41 CST,
   RLMRec passed epoch 1000 (`train_loss=1.477797`) and reached epoch 1030
   (`train_loss=1.478778`) with clean fatal/OOM/CUDA/no-space scans and stable
   `28G` free disk. The output directory still contained only
   `inspect_fairness_provenance.json`, so the sports official evidence count
   remains five. At 2026-06-01 12:10 CST, RLMRec passed epoch 2000
   (`train_loss=1.476514`) with clean fatal/OOM/CUDA/no-space scans and stable
   `28G` free disk. The output directory still contained only
   `inspect_fairness_provenance.json`; no final score/provenance/audit/table
   package existed yet, so the sports official evidence count remained five.
   At 2026-06-01 13:50 CST, sports `rlmrec_graphcl` completed and passed all
   row gates: `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
   and local-light audit PASS. Metrics over 10,000 users and 101 candidates:
   HR@5/10/20=`0.1212/0.1879/0.3009`,
   NDCG@5/10/20=`0.078580389191345/0.10001773336299705/0.12818232277286493`,
   MRR=`0.09720456858848743`, `sample_count=10000`,
   `avg_candidates=101.0`, `score_rows=1010000`, and
   `candidate_rows=1010000`. Server row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has six
   completed official rows, but no sports SOTA claim is allowed until LLM2Rec,
   LLM-ESR, the complete eight-baseline comparison table, and paired tests
   finish. The runner then reached sports `llm2rec_sasrec` and stopped during
   adapter export because validation candidate histories were incorrectly
   looked up in the test-task train file. The local fix uses
   `valid_same_candidate/train_interactions.csv` for validation histories and
   passed a targeted unit test; it still needs server pull and resumed LLM2Rec
   execution before any LLM2Rec row is eligible. At 2026-06-01 14:03 CST, the
   fix had been deployed to the dirty server worktree without resetting
   unrelated experiment changes, and sports `llm2rec_sasrec` was resumed as a
   single-row job. The real sports run passed the previous adapter-export
   blocker and entered Qwen3 `hf_mean_pool` embedding generation at about
   `3432/283760` under PID `2870575`. This is progress only; the LLM2Rec row
   remains ineligible until final scores, provenance, audits, complete metrics,
   and row-count checks pass. At 2026-06-01 15:14 CST, sports
   `llm2rec_sasrec` had completed full Qwen3 item embedding
   (`283760/283760`) and produced matching 4,649,140,352-byte embedding files
   under the Pony adapter directory and the upstream LLM2Rec
   `item_info/SportsSameCandidate100Neg/` path. The run then stopped before
   official SASRec training with
   `FileNotFoundError: [Errno 2] No such file or directory: 'python'` from
   `_train_with_official_entrypoint`. This is a wrapper environment bug, not
   a completed or failed metric row. Local fix changes the subprocess command
   to start with `sys.executable` while preserving the official
   `evaluate_with_seqrec.py` entrypoint and SASRec arguments; targeted local
   tests passed (`tests/test_llm2rec_upstream_adapter.py`: 5 passed,
   `tests/test_llm2rec_same_candidate_export.py`: 3 passed). The LLM2Rec row
   remains not paper/table eligible until the server resume produces final
   `scores.csv`, provenance, score audit, imported full `@5/@10/@20 + MRR`
   metrics, row-count checks, server-final audit, and local-light evidence
   audit. At 2026-06-01 15:51 CST, the fixed runner had been verified on the
   server and sports `llm2rec_sasrec` was active again: adapter PID `2875446`
   and upstream official `evaluate_with_seqrec.py` PID `2875559`. The upstream
   official command uses the existing Qwen3 embedding file rather than
   regenerating it. The training log reached epoch 15 validation and saved
   checkpoints at epochs 5 and 10. This is a running checkpoint only; no final
   LLM2Rec row, metric table, provenance, or score audit is available yet.
   At 2026-06-01 15:56 CST, sports `llm2rec_sasrec` completed as an official
   row with `implementation_status=official_completed`, `blockers=[]`,
   `score_coverage_rate=1.0`, server-final audit PASS, lightweight sync PASS,
   and local-light audit PASS. Full same-candidate metrics over 10,000 users
   and 101 candidates are HR@5/10/20=`0.1105/0.206/0.3657`,
   NDCG@5/10/20=`0.06514778914391295/0.09566791850988236/0.13561659669926907`,
   and MRR=`0.08828933028385053`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has seven
   completed official rows; no sports SOTA claim is allowed until LLM-ESR,
   the full eight-baseline table, and paired tests finish. At 2026-06-01
   16:28 CST, the completed LLM2Rec intermediate adapter directory was removed
   after server-final/local-light audit checks, recovering about `5.3G`
   without touching final LLM2Rec evidence or the upstream embedding. Sports
   `llmesr_sasrec` was then launched as the eighth sports official row under
   runner PID `2877443` and adapter PID `2877452`; it is in Qwen3
   `hf_mean_pool` embedding at about `51472/233470`. This is a running
   checkpoint only: LLM-ESR has no final scores/provenance/imported metrics
   yet and is not table-eligible. At 2026-06-01 16:50 CST, LLM-ESR was still
   running under the same PIDs, with embedding progress about
   `141696/233470`, GPU sample `95%`, `16285 MiB / 49140 MiB`, and about
   `22G` free disk. No LLM-ESR `scores.csv`, final provenance, score audit,
   imported table, or predictions exist yet. The completed sports rows were
   also rechecked for the user's full-metric concern: the seven completed rows
   all have HR@5/@10/@20, NDCG@5/@10/@20, MRR, `sample_count=10000`,
   `avg_candidates=101.0`, exact 1,010,000/1,010,000 score coverage, final
   provenance, score audit, and imported `ranking_eval_records.csv`. The four
   earliest rows (`llmemb`, `proex_profile`, `promax_profile`, and
   `elmrec_graph`) lacked only the newer standardized
   `server_final_evidence_audit.json`; that audit was backfilled with
   `ok=true` on the server and local-light audits passed after copying the new
   JSONs into the local lightweight packages. No scores changed. At
   2026-06-01 17:15 CST, LLM-ESR completed Qwen3 embedding
   (`233470/233470`) and entered official training, with logged losses
   epoch 1 `1.374167` and epoch 5 `0.361412`. The row is still not
   table-eligible: no final LLM-ESR `scores.csv`, final provenance, score
   audit, predictions, imported tables, or local lightweight evidence package
   exists yet. Disk is a watch item at about `15G` free (`93%` used), but a
   read-only storage review found no meaningful safe cleanup while the active
   LLM-ESR adapter is running and completed-row final evidence is protected.
   At 2026-06-01 18:42 CST, sports `llmesr_sasrec` completed as the eighth
   sports official row with `implementation_status=official_completed`,
   `blockers=[]`, and `score_coverage_rate=1.0`. Server-final audit,
   lightweight sync, and local-light audit passed. Full metrics over 10,000
   users and 101 candidates are HR@5/10/20=`0.0916/0.1564/0.2650`,
   NDCG@5/10/20=`0.054919833257876506/0.0758115528438973/0.10310478593304104`,
   and MRR=`0.0751149958885503`. Row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). Sports now has all
   eight official rows complete, but sports SOTA is still not claimable until
   the full sports comparison table and paired tests are built. The completed
   LLM-ESR intermediate adapter directory was removed only after final evidence
   and local lightweight backup passed, recovering disk from `9.4G` to `14G`
   free without touching final outputs. At 2026-06-01 19:08 CST, a read-only
   domain gate with `scripts/audit/main_audit_domain_official_gate.py`
   generated `outputs/summary/sports_official_ccrp_gate_20260601.json` and
   `.csv` on the server and local. The gate returned
   `official_ok_count=8`, `ccrp_ok=true`, and `gate_ok=true`: all eight
   official rows plus `ccrp_v3_qwen3base_pointwise` have complete
   HR@5/@10/@20, NDCG@5/@10/@20, MRR, 10,000 users, 101 candidates,
   `score_coverage_rate=1.0`, expected row counts, and no gate failures.
   This confirms sports result completeness, not a SOTA claim; paired tests
   and the full comparison table remain required before any winner wording.
   At 2026-06-01 19:20 CST, `scripts/experiments/main_build_domain_official_comparison.py`
   built that sports comparison/statistical gate from existing server
   artifacts. The generated local/server outputs are
   `outputs/summary/sports_official_ccrp_20260601_comparison.csv`,
   `outputs/summary/sports_official_ccrp_20260601_comparison.md`,
   `outputs/summary/sports_official_ccrp_20260601_paired_tests.csv`, and
   `outputs/summary/sports_official_ccrp_20260601_paired_summary.json`.
   C-CRP is rank 1 and observed-best on all seven full metrics. The paired
   family has 56 tests (8 official baselines x HR@5/@10/@20, NDCG@5/@10/@20,
   and MRR), all with 10,000 paired events, positive deltas, paired-bootstrap
   95% CIs above zero, and Holm-significant p-values. Against the strongest
   official row (`llmemb`), the closest margin is HR@20 delta `0.0272` with
   CI `[0.0164, 0.0386]` and Holm p `1.219129314796352e-06`. This supports a
   sports-domain passed gate only; paper-wide SOTA wording still requires the
   declared domain set and ARIS review.
2. Import and audit each remaining new-domain baseline row with exact score coverage,
   full @5/@10/@20 metrics, provenance, and row-count checks. At
   2026-06-01 19:48 CST, the next-domain phase began with a storage preflight:
   disk was only `14G` free, so disposable user caches under
   `/home/ajifang/.cache` were removed after path verification, recovering
   about `5G` and leaving about `19G` free without deleting project outputs or
   protected evidence. Toys `proex_profile` was then launched as a
   single-domain/single-method official row with runner PID `2893793`, adapter
   PID `2893803`, PID file `baselines_new_domains_toys_proex.pid`, and log
   `baselines_new_domains_toys_proex_20260601_194414.log`. At the 19:48 check
   it was in Qwen3 `hf_mean_pool` embedding at about `7088/215034`, with GPU
   `95%` and disk about `18G` free. This row is running and not yet
   table-eligible; do not include it in claims until final score/provenance,
   audits, imported full metrics, row counts, local-light sync, and paired
   evidence gates pass. At 2026-06-01 21:13 CST, toys `proex_profile`
   completed as `official_completed` with `blockers=[]` and
   `score_coverage_rate=1.0`. Server-final audit, lightweight sync, and
   local-light audit passed. Full metrics over 10,000 users and 101 candidates
   are HR@5/10/20=`0.0895/0.1615/0.3017`,
   NDCG@5/10/20=`0.058141214365017416/0.0810170703641553/0.11607709818340411`,
   and MRR=`0.08121671352544663`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_proex_profile_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records the final large files while keeping
   `scores.csv`, predictions, and `proex_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `14G` to `18G` free without touching final evidence. Toys
   now has 1/8 completed official baselines. At 2026-06-01 21:31 CST, toys
   `promax_profile` was running as the next single-row official baseline under
   runner PID `2899989` and adapter PID `2899998`, with log
   `baselines_new_domains_toys_promax_20260601_212808.log`; it was in Qwen3
   embedding at about `1312/215034` and is not table-eligible until the same
   gates pass. At 2026-06-02 00:02 CST, toys `promax_profile` completed as
   `official_completed` with `blockers=[]` and `score_coverage_rate=1.0`.
   Server-final audit, lightweight sync, and local-light audit passed. Full
   metrics over 10,000 users and 101 candidates are
   HR@5/10/20=`0.0920/0.1435/0.2416`,
   NDCG@5/10/20=`0.06289618254810064/0.07937554863319267/0.10387644003990415`,
   and MRR=`0.08184625622431366`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_promax_profile_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records final large files while keeping
   `scores.csv`, predictions, and `promax_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `13G` to `17G` free without touching final evidence. Toys
   now has 2/8 completed official baselines. At 2026-06-02 00:10 CST, toys
   `elmrec_graph` was running as the next single-row official baseline under
   runner PID `2906447` and adapter PID `2906455`, with log
   `baselines_new_domains_toys_elmrec_20260602_000729.log`; at the 2026-06-02
   00:15 CST monitor check it was still active and had advanced to about
   `21872/215034` Qwen3 embeddings, with no fatal/OOM/no-space markers. It is
   not table-eligible until the same gates pass. At 2026-06-02 01:36 CST,
   toys `elmrec_graph` completed as `official_completed` with `blockers=[]` and
   `score_coverage_rate=1.0`. Server-final audit, lightweight sync, and
   local-light audit passed. Full metrics over 10,000 users and 101 candidates
   are HR@5/10/20=`0.0545/0.1043/0.2013`,
   NDCG@5/10/20=`0.03259298673054038/0.04856005753116525/0.07278039157879498`,
   and MRR=`0.05431081812612059`; row counts passed for `scores.csv`
   (`1,010,001` lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_elmrec_graph_official_qwen3base_same_candidate/`.
   A server-side sha256 manifest records final large files while keeping
   `scores.csv`, predictions, and `elmrec_official_model.pt` server-only. The
   completed intermediate adapter directory was removed after audits, recovering
   disk from about `12G` to `16G` free without touching final evidence. At
   2026-06-02 03:04 CST, toys `llmemb` completed as the fourth official row
   with `implementation_status=official_completed`, `blockers=[]`, and
   `score_coverage_rate=1.0`. Server-final and local-light audits passed with
   full metrics over 10,000 users and 101 candidates: HR@5/10/20
   `0.2499 / 0.3505 / 0.4866`, NDCG@5/10/20
   `0.17252113274887534 / 0.20485045979333913 / 0.23905481091819092`, and MRR
   `0.1813804118284203`. Row counts passed for `scores.csv` (`1,010,001`
   lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_llmemb_official_qwen3base_same_candidate/`;
   final `scores.csv`, predictions, and `llmemb_official_model.pt` remain
   server-only and covered by `server_large_artifact_manifest.sha256`. After
   audits and local sync passed, the completed intermediate adapter directory
   was removed, recovering disk from about `4.0G` to `8.3G` free without
   touching final evidence. Toys now has 4/8 completed official baselines. At
   2026-06-02 03:16 CST, toys `irllrec_intent` was launched as the fifth
   single-row official baseline under runner PID `2923429` and adapter PID
   `2923437`, with log
   `baselines_new_domains_toys_irllrec_20260602_031623.log`; at the 03:19 CST
   check it was in Qwen3 `hf_mean_pool` embedding at about `1400/215034`, with
   GPU `96%` and disk about `7.3G` free. This row is running and not
   table-eligible until final score/provenance, audits, imported full metrics,
   row counts, local-light sync, and paired evidence gates pass. At 04:12 CST,
   after embeddings completed and training reached epoch 30, disk had fallen to
   about `4.0G` free; only disposable pip cache/temp paths were removed after
   scope checks, recovering disk to about `4.4G` free without touching project
   evidence or active adapters. At 04:35 CST, IRLLRec was still running at
   epoch `500/3000`; GPU was active and disk remained about `4.4G` free. The
   final evidence directory still lacked scores/provenance/audits/imported
   tables, so toys remains 4/8 complete. A read-only cleanup audit did not find
   another safe project artifact to remove: the old
   `books_large10000_100neg_llmesr_adapter` contains historical adapter mapping
   data while the corresponding final books directory is table-only and no
   local lightweight package was found, so it was left intact. At 04:47 CST,
   IRLLRec had reached `epoch=760/3000` and still had no final evidence
   package. Three user-level cache directories
   (`.vscode-server/data/CachedExtensionVSIXs`, Chrome `component_crx_cache`,
   and Code `CachedData`) were removed after realpath allowlist checks,
   recovering disk from about `4.4G` to `4.6G` free without touching project
   outputs, active adapters, models, Python site-packages, or other projects.
   At 04:56 CST, IRLLRec had reached `epoch=940/3000` with no final evidence
   files and a clean error scan. Five inactive VSCode remote server cache
   directories under `.vscode-server/cli/servers/Stable-*` were removed after
   a no-process check and realpath prefix verification, recovering disk to
   about `6.4G` free without touching project outputs, final evidence, active
   adapters, models, conda/Python environments, or other projects.
   At 2026-06-02 06:35 CST, toys `irllrec_intent` completed as the fifth
   official row with `implementation_status=official_completed`, `blockers=[]`,
   and `score_coverage_rate=1.0`. Server-final audit and local-light audit
   passed with full metrics over 10,000 users and 101 candidates: HR@5/10/20
   `0.1565 / 0.2293 / 0.4098`, NDCG@5/10/20
   `0.11049209461545026 / 0.13380144693674725 / 0.1785851471792316`, and MRR
   `0.1311986744710446`. Row counts passed for `scores.csv` (`1,010,001`
   lines), predictions (`10,000` lines), and
   `tables/ranking_eval_records.csv` (`10,001` lines). The local lightweight
   package is
   `outputs/baselines/official_adapters/toys_large10000_100neg_irllrec_intent_official_qwen3base_same_candidate/`;
   final `scores.csv`, predictions, and `irllrec_official_model.pt` remain
   server-only and covered by `server_large_artifact_manifest.sha256`. After
   audits and local sync passed, the completed IRLLRec intermediate adapter
   directory was removed, recovering disk from about `5.4G` to `9.7G` free
   without touching final evidence. Toys now has 5/8 completed official
   baselines. At 2026-06-02 06:44 CST, toys `rlmrec_graphcl` was launched as
   the next single-row official baseline with runner PID `2937284`, adapter PID
   `2937292`, and log `baselines_new_domains_toys_rlmrec_20260602_064443.log`;
   at the 06:47 CST check it was in Qwen3 `hf_mean_pool` embedding at about
   `1664/215034`, GPU `99%`, and disk about `8.8G` free. This row is running
   and not table-eligible until final score/provenance, audits, imported full
   metrics, row counts, local-light sync, and paired evidence gates pass.
3. Full @5/@10/@20 comparison table across all domains
4. Statistical significance tests (paired t-test, 20+ seeds or bootstrap)
5. Paper writing
6. GPT-5.5/Codex review cycle (target: 8/10)
