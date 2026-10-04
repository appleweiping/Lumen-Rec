export const meta = {
  name: 'amendment2-implementation',
  description: 'Implement PREREG_AMENDMENT_2 code: prompt-variant bank, confirm panels, exhaustive decision table + gate-fix selection, CPU forensics, diagnosis battery + run scripts; review, repair, integrate',
  phases: [
    { title: 'Implement', detail: '5 disjoint groups' },
    { title: 'Review', detail: 'adversarial reviewer per group' },
    { title: 'Repair', detail: 'fix verified issues' },
    { title: 'Integrate', detail: 'full tests, contract audit, CPU dry run' },
  ],
}

const SCRATCH = 'D:/_Organized/Temp-Review/_RootDirs/temp/claude/D--/22c9b1b2-12a5-4956-a490-79bcb9beb0d8/scratchpad'
const COMMON = `You implement code for the Lumen-Rec SIGIR-2027 research project. Correctness and exact adherence to the pre-registration matter more than speed: a silent bug here corrupts registered scientific decisions.
Repository: D:/Research/Lumen (Windows).
ENVIRONMENT RULES (strict):
- The repo's .git is corrupted: NEVER run any git command; never touch .git.
- Use the PowerShell tool. Tests: cd D:/Research/Lumen; $env:PYTHONPATH='.'; python -m pytest <files> -q -p no:cacheprovider (only tests/test_confrec_*.py matter).
- No GPU/vLLM locally: import vllm lazily inside main(); tests must not need vllm or a real model (use fakes).
- Do NOT contact the remote server; do not download data.
- Edit ONLY files in your ownership list (others are being edited concurrently). src/confrec/stats.py, metrics.py, categories.py are shared and FROZEN (import, never edit).
- BINDING SPEC: idea-stage/PREREG_AMENDMENT_2.md (read it fully) plus idea-stage/PREREG_AMENDMENT_1.md. Variant definitions: idea-stage/deliberation_2026-10-02/gatefix_prompt_variants_v1.json is NOT in the repo; its copy is at ${SCRATCH}/deliberation/gatefix_prompt_variants_v1.json (read it). Judge rationale: idea-stage/deliberation_2026-10-02/judge.md.
- Local copies of the real Pilot-1 panels and reports (for byte-identity tests; read-only): ${SCRATCH}/pilot_panels/{ml1m_rated.jsonl, toys_rated.jsonl, sports_next_1k.jsonl, *.report.json, *.meta.json, toys_rated.brand_pop.json}. Pilot ML-1M report: questions like,dislike,like_para, hist_len 10, data_sha1 985494c7b44d010bec4b11ec62f35ad274dfe91f, prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 (prompts_sha1 = pyes_scorer.prompts_sha1 over the user-prompt strings of record_requests(rec, questions, hist_len) in record order).
- Shared helpers: src/confrec/stats.py (rankdata_avg, spearman, cluster_bootstrap, paired_bootstrap, tie_aware_rank, rank_bins, ols_slope, sigmoid, platt_fit/apply, user_halves, strict_json, percentile_ci); metrics.py (ece, brier, auroc, risk_coverage, bias_index, bias_index_ci, ndcg_from_rank).
- Style: match surrounding code; module docstrings with usage; argparse CLIs (python -m src.confrec.<mod>); strict JSON via stats.strict_json. Every behaviour change gets a test that would fail without it. Run your tests until they pass.

CROSS-GROUP CONTRACT:
1. prompting.py (group prompts) exports VARIANTS (keys V0,V1,V2,V3,V4,V5,V7 plus diagnostic key T0_probe), render(rec, i, question, variant="V0", panel_kind=None) -> (system: str|None, user: str), build_prompt (unchanged V0 wrapper), chat_ids(tok, user, system=None), yes_no_ids, digit_ids(tok) -> dict r->frozenset ids for "1".."5", read_yes_no, read_digits(top, digit_ids) -> (exp_rating, logit_45_vs_12, mass, censored). panel_kind is "rated" when rows carry candidate_labels+history_ratings, else "next_item"; auto-detected when None. Optional per-candidate field candidate_extras (list[str], "" = none) is rendered as an extra line "\\n{extra}" right after the description line in EVERY variant (absent field => byte-identical to before). Rated rows may carry history of up to 20 events plus history_meta (list[str] aligned with history_item_ids, e.g. "Genres: Action, Drama" or "Categories: ...") and domain_kind ("movie"|"product"). Variants with hist 10 use the last 10 history entries; V3/V7 use the last 20.
2. pyes_scorer.py flags (group prompts): existing flags + --variant V0 (default) + --readout yesno|digits (default yesno; digits writes extra column exp_rating, logit = log P(4|5) - log P(1|2)) ; --max_model_len default 4096. report.json records variant, readout, system prompt sha1; prompts_sha1 hashes system+"\\x1d"+user when a system message exists, else the user string exactly as before (so V0 reproduces 12e83c4f...). Output columns unchanged + exp_rating (NaN for yesno).
3. Panels (group panels): build_rated_panels adds history_meta, domain_kind; --hist_len 20 supported. scripts/sigir/build_confirm_panels.py writes outputs/confrec/gatefix/panels/{ml1m,toys}_{dev,confirm}_h20.jsonl and manifest outputs/confrec/gatefix/panels/manifest.json (eligible counts incl. games+sports rated, prefix sha1 checks, user-id list sha1s, n users per split).
4. pilot1_gate.py + pilot_mirror.py + gatefix_select.py (group decision): exhaustive table with INDETERMINATE; pilot_mirror adds the correlation-matched ensemble-null block; gatefix_select CLI: python -m src.confrec.gatefix_select dev --root outputs/confrec/gatefix/dev --variants V0,V1,V2,V3,V4,V5,V7 --out outputs/confrec/gatefix/dev/selection.json  (expects ROOT/{ml1m,toys}/{V}/scores.csv.gz + report.json) and python -m src.confrec.gatefix_select confirm --vstar_dir D --v0_dir D --selection selection.json --out gate.json.
5. forensics.py (group forensics): python -m src.confrec.forensics --pilot_dir outputs/confrec/pilot1_mirror --panels outputs/confrec/panels --raw data/raw --out outputs/confrec/forensics/forensics.json  (+ a local-only mode --ccrp_head_share for A8 reading local ranking_eval_records).
6. diag_battery.py + run scripts (group battery): python -m src.confrec.diag_battery make --kind {t0,t1,t2,starperm} --panel P --raw DIR --source ml1m|amazon [--domain d] --out P2 ; python -m src.confrec.diag_battery analyze --root outputs/confrec/diag --out diag.json ; scripts/sigir/run_gatefix.sh (stage0 panels+hash print, stage1 dev scoring + selection, stage2 confirm if FIX_FOUND; NEVER scores dislike/para/swap on dev) and scripts/sigir/run_diag_battery.sh.`

const GROUPS = [
  { key: 'prompts', files: 'src/confrec/prompting.py, src/confrec/pyes_scorer.py, src/confrec/train_lora_yesno.py, tests/test_confrec_scorer.py, tests/test_confrec_train_lora.py, NEW tests/test_confrec_variants.py',
    spec: `Implement contract items 1-2 exactly per the variants JSON (V6 omitted; next-item renderings implemented but next-item stays frozen at V0 by policy). T0_probe: item-only prompt, no history: user = "Movie: {title}{desc}\\n\\nIs this movie widely considered good? Answer with only Yes or No." for domain_kind movie, "Product: ..." / "Is this product widely considered good?" otherwise (desc rendered like V0). V2 split rule: liked = rating > 3, disliked = rating <= 3, empty list renders "- (none)". V4 history meta: ML-1M "Genres: ..." from history_meta; Amazon "Categories: ..."[:80]. V5/V7 system messages per the JSON (rated_movie vs rated_product by domain_kind; next_product for next-item). Threshold-family questions for V1/V7 (like/dislike/para strings exactly as JSON). Tests: (a) V0 render of the local pilot ML-1M panel with questions like,dislike,like_para reproduces prompts_sha1 12e83c4fcca40db398ac58acbba2a7eb9c770524 via pyes_scorer.prompts_sha1 + record_requests (skip with a clear message if the local file is absent); (b) the same holds when the panel rows carry 20 history entries (simulate by prepending 10 dummy older events to each row: V0 must still use the last 10); (c) one golden string per variant on a fixed fake row; (d) candidate_extras rendering; (e) read_digits math incl. censoring; (f) trainer and scorer produce identical ids per variant (fake tokenizer).` },
  { key: 'panels', files: 'src/confrec/build_rated_panels.py, NEW scripts/sigir/build_confirm_panels.py, tests/test_confrec_rated_panels.py, NEW tests/test_confrec_confirm_panels.py',
    spec: `Implement contract item 3. history_meta for ML-1M from movies.dat genres ("Genres: Action, Drama"); for Amazon from meta categories string[:80] ("Categories: ..."); "" when missing; domain_kind. Building with --hist_len 20 must select exactly the same users, candidates, candidate order and labels as --hist_len 10 (history is only longer) - test it. build_confirm_panels.py (G2): per source (ml1m; toys only if eligible >= 2000): build the full eligible panel at hist_len 10 into a temp file and assert sha1 of its first 1500 lines equals the pilot data_sha1 (ml1m 985494c7b44d010bec4b11ec62f35ad274dfe91f; toys 69252b4806bb4ffe05101967ea2a1d94ada57dee) - on mismatch fall back to user_id exclusion against the pilot panel file and log it in the manifest; then build the same at hist_len 20, split DEV = rows whose user_id is in the pilot panel, CONFIRM = the rest (assert disjoint, assert DEV users == pilot users), write the h20 panels, user-id lists (sorted, newline-joined) and their sha1s, eligible counts for ml1m/toys/games/sports rated (n_users large), and manifest.json. Must run on the server CPU with the real data (Toys 16M reviews: keep memory reasonable); tests on tiny fixtures.` },
  { key: 'decision', files: 'scripts/sigir/pilot1_gate.py, src/confrec/pilot_mirror.py, NEW src/confrec/gatefix_select.py, tests/test_confrec_pilot1_gate.py, tests/test_confrec_pilot_mirror.py, NEW tests/test_confrec_gatefix_select.py',
    spec: `Contract item 4. (a) pilot1_gate.decide: the label 'AMBIGUOUS' becomes 'INDETERMINATE' (not POSITIVE, no handoff); keep INCOMPLETE for undetermined (None) inputs and GATE_FAIL_UNINTERPRETABLE; add a test that enumerates EVERY combination of the decision conditions (True/False/None for each input condition) and asserts exactly one label results and that POSITIVE never co-occurs with a failed ensemble-null condition. (b) G7 ensemble null: pilot_mirror adds block ensemble_null = binormal-predicted UAUC of the sum of the two views (like and -dislike): per user compute AUC_like, AUC_negdislike and the within-user Pearson correlation rho of the two logits (pooled within-class correlation), d'_k = sqrt(2)*Phi^-1(AUC_k), predicted AUC = Phi(((d'_1+d'_2)/sqrt(2+2rho))/sqrt(2)); report mean predicted UAUC and dUAUC(mirror - predicted) with a user bootstrap (paired over users), plus the same construction for the placebo pair (like, like_para) as a sanity check. pilot1_gate POSITIVE additionally requires dUAUC(mirror - ensemble_null).lo > 0 in a domain where dUAUC(mirror - placebo).lo > 0 (record the value). (c) gatefix_select.py per amendment 2 G4/G5/G6 exactly: dev stage computes like UAUC per variant per panel (stats/metrics helpers; ties averaged), E1 (censored=2 share <= 0.5%, overlength = 0 from report.json n_overlength or censored=3 count, mean yes_no_mass >= 0.95), E2 (Toys dev UAUC(V) >= Toys dev UAUC(V0) - 0.010), V* with the tie rule (within 0.005 of max -> higher Toys dev UAUC -> simplicity order V0<V1<V5<V3<V4<V2<V7), one-sided paired user bootstrap lower bound of UAUC(V*)-UAUC(V0) on ML-1M dev (2000 resamples, seed 0, level 1-0.05/6), FIX_FOUND iff V*!=V0 and lower bound > 0, else F0 GATE_FAIL_AFTER_REMEDY(dev); publish the full 7x2 table. confirm stage: GATE_PASS iff UAUC(V*) >= 0.60 (point estimate) and E1 holds, else F1 GATE_FAIL_AFTER_REMEDY(confirm); report 95% user-bootstrap CI and V0 context. Exit codes: dev 0 = FIX_FOUND, 4 = F0; confirm 0 = PASS, 3 = F1.` },
  { key: 'forensics', files: 'NEW src/confrec/forensics.py, NEW tests/test_confrec_forensics.py',
    spec: `Contract item 5: implement amendment 2 section A checks A2-A8 (A1 belongs to the panels group; A9 is a report field you copy from report.json prompts/s): A2 UAUC of pi alone and L_nohist alone, split-half Spearman reliability of pi over items (donors are the rows of swap_prior.csv.gz per item in file order: donors 1-4 vs 5-8), 4-donor evidence UAUC and Spearman-Brown projection; A3 prior-only leave-user-out item mean (other users' ratings strictly BEFORE the candidate's timestamp; uses build_rated_panels loaders and candidate_timestamps when present, else rebuild timestamps from raw), leave-candidates-out biased MF (train on all ratings except every panel user's candidate events; SGD or ALS on CPU, dims 32, fixed seed; report UAUC of the MF score and of the personal residual MF - b_u - b_i), global AUC of the user's history-star mean; A4 MIRROR vs the binormal ensemble null (same definition as the decision group) on the pilot outputs; A5 ECE after intercept-only recalibration (fit b in sigmoid(L + b)), after temperature-only (sigmoid(L / T)), and user-centred ECE (subtract each user's mean logit before Platt); A6 error targets after Platt and per-user top-k errors (k = #likes) with detector AUROCs; A7 partial Spearman of pi and v against log-pop controlling for the prior-only item mean (+ release year parsed from the ML-1M title, + description length and has_store for Toys); A8 local-only: from D:/Research/Lumen/outputs/sports_large10000_100neg_ccrp_v3_qwen3base_pointwise_same_candidate/tables/ranking_eval_records.csv compute C-CRP top-10 head share and the expected top-10 head share under random ranking per event (from candidate_popularity_groups), restricted to the first 1000 events and to all 10000. Output strict JSON with every quantity labelled exploratory. Tests on synthetic fixtures with known answers (e.g. pi reliability on a planted signal, MF residual null when no personal signal, binormal prediction matches a simulated sum).` },
  { key: 'battery', files: 'NEW src/confrec/diag_battery.py, NEW scripts/sigir/run_gatefix.sh, NEW scripts/sigir/run_diag_battery.sh, NEW tests/test_confrec_diag_battery.py',
    spec: `Contract item 6 and amendment 2 sections G and D. diag_battery make: t0 = one row per unique panel item (history empty, candidate = the item, domain_kind kept) to be scored with --variant T0_probe --questions like (the probe text lives in the variant); t1 = copy of the panel with candidate_extras "(This user later rated this item r/5.)" using the candidate's true rating; t2 = candidate_extras "Average rating by other users before this date: m/5 (n ratings)" with the prior-only leave-user-out mean (strictly before the candidate timestamp; "no ratings yet" when n = 0); starperm = K=2 copies of the panel where the history "(rated r/5)" suffixes are permuted among the same history items (seeded, derangement where possible). diag_battery analyze computes the readings of amendment 2 section D (T0 Spearman vs prior-only item mean; T1 UAUC; T2 transmission ratio; T3 digit-readout UAUC vs yes/no; star permutation dUAUC and within-user SD of tau_P; T4 Llama UAUC) with the registered interpretive thresholds, all labelled interpretive-only. run_gatefix.sh: stage0 = build_confirm_panels.py, then print and write outputs/confrec/gatefix/FREEZE.txt with sha1 of idea-stage/PREREG_AMENDMENT_2.md, sha1 of the rendered prompt bank (python one-liner over prompting.render for every variant x first 50 rows of each dev panel x like) and the user-id list sha1s from the manifest, then STOP unless FREEZE_ACK=1 is set (the orchestrator records the hashes in PILOT_LOG first); stage1 = score like only for V0,V1,V2,V3,V4,V5,V7 on ml1m and toys dev h20 panels with pyes_scorer --variant (skip-if-done markers as in run_pilot1_mirror.sh; output root outputs/confrec/gatefix/dev/{ml1m,toys}/{V}), then gatefix_select dev; stage2 only if exit 0: score V* like and V0 like on ml1m confirm, then gatefix_select confirm. run_diag_battery.sh: T0..T3, starperm on burned ML-1M (+T2 on 500 burned Toys users), T4 = Llama-3.1-8B-Instruct (/root/autodl-tmp/lumen/models/Llama-3.1-8B-Instruct) V0 like on burned ML-1M, separate output root outputs/confrec/diag/, then analyze. Both scripts: conda activation as in run_pilot1_mirror.sh, quoted paths, set -euo pipefail, bash -n clean (Git Bash at C:/Program Files/Git/bin/bash.exe). Tests for make/analyze on synthetic fixtures.` },
]

const IMPL = {
  type: 'object',
  properties: {
    files_changed: { type: 'array', items: { type: 'string' } },
    done: { type: 'array', items: { type: 'string' } },
    not_done: { type: 'array', items: { type: 'object', properties: { item: { type: 'string' }, why: { type: 'string' } }, required: ['item', 'why'] } },
    tests_result: { type: 'string' },
    tests_passed: { type: 'boolean' },
    interface_notes: { type: 'string' },
  },
  required: ['files_changed', 'done', 'not_done', 'tests_result', 'tests_passed', 'interface_notes'],
}
const REVIEW = {
  type: 'object',
  properties: {
    issues: { type: 'array', items: { type: 'object', properties: {
      severity: { type: 'string', enum: ['critical', 'major', 'minor'] }, file: { type: 'string' }, title: { type: 'string' },
      evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'file', 'title', 'evidence', 'fix'] } },
    spec_gaps: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string', enum: ['clean', 'needs_repair'] },
  },
  required: ['issues', 'spec_gaps', 'verdict'],
}

phase('Implement')
const results = await pipeline(
  GROUPS,
  g => agent(`${COMMON}

YOUR GROUP: ${g.key}
OWNED FILES (edit only these): ${g.files}
SPEC:
${g.spec}
Return the structured summary; be honest in not_done.`, { label: `impl:${g.key}`, phase: 'Implement', schema: IMPL }),
  (impl, g) => agent(`${COMMON}

You are an independent ADVERSARIAL REVIEWER of group '${g.key}' (you did not write it). READ-ONLY on the repo; probe scripts only under ${SCRATCH}/probe2_${g.key}/.
Owned files: ${g.files}
Spec:
${g.spec}
Implementer summary (may be optimistic): ${JSON.stringify(impl)}
Check (1) every spec item and the amendment-2 text it implements (registered constants, tie rules, levels, user roles, MIRROR-blindness) against the code; (2) try to break it with probe scripts (byte identity, edge cases, NaNs, ties, empty lists, unicode, resume); (3) run the group's tests. Severity: critical = would corrupt a registered decision or waste a GPU run; major = wrong number or contract violation; minor = robustness. verdict needs_repair if any critical/major issue or spec gap.`, { label: `review:${g.key}`, phase: 'Review', schema: REVIEW })
    .then(rev => ({ impl, rev })),
  (r, g) => {
    const bad = (r.rev.issues || []).filter(i => i.severity !== 'minor').length + (r.rev.spec_gaps || []).length
    if (r.rev.verdict === 'clean' && bad === 0 && (r.rev.issues || []).length === 0) return { group: g.key, ...r, repair: null }
    return agent(`${COMMON}

YOUR GROUP: ${g.key} (REPAIR). OWNED FILES: ${g.files}
Spec:
${g.spec}
Implementer summary: ${JSON.stringify(r.impl)}
Reviewer report: ${JSON.stringify(r.rev)}
Verify each issue/gap against the code first (reviewers can be wrong); fix every verified critical/major issue and spec gap, and verified minor issues when small and safe; list rejected ones with reasons in not_done. Re-run tests until they pass.`, { label: `repair:${g.key}`, phase: 'Repair', schema: IMPL })
      .then(rep => ({ group: g.key, ...r, repair: rep }))
  },
)

phase('Integrate')
const INTEG = {
  type: 'object',
  properties: {
    tests_result: { type: 'string' }, tests_passed: { type: 'boolean' },
    contract_fixes: { type: 'array', items: { type: 'string' } },
    v0_byte_identity: { type: 'string' },
    dry_run: { type: 'array', items: { type: 'object', properties: { step: { type: 'string' }, ok: { type: 'boolean' }, note: { type: 'string' } }, required: ['step', 'ok', 'note'] } },
    files_changed: { type: 'array', items: { type: 'string' } },
    remaining_risks: { type: 'array', items: { type: 'string' } },
  },
  required: ['tests_result', 'tests_passed', 'contract_fixes', 'v0_byte_identity', 'dry_run', 'files_changed', 'remaining_risks'],
}
const integ = await agent(`${COMMON}

You are the INTEGRATION engineer; all 5 groups finished. You may edit any file under src/confrec, scripts/sigir, tests/test_confrec_* (not stats/metrics/categories unless a proven contract bug). Group outcomes: ${JSON.stringify(results.map(r => r && ({ group: r.group, impl: r.impl, review_verdict: r.rev && r.rev.verdict, repair: r.repair })))}
Tasks: (1) run ALL tests/test_confrec_*.py; fix failures. (2) Contract audit: run_gatefix.sh / run_diag_battery.sh flags exist in every target argparse; gatefix_select reads exactly what pyes_scorer writes; diag_battery panels render through prompting (candidate_extras, T0_probe); pilot1_gate reads pilot_mirror's ensemble_null keys. (3) V0 BYTE IDENTITY on the real local pilot panels: render V0 like,dislike,like_para on ${SCRATCH}/pilot_panels/ml1m_rated.jsonl and confirm prompts_sha1 == 12e83c4fcca40db398ac58acbba2a7eb9c770524; also for toys_rated.jsonl against its report.json prompts_sha1 and for sports_next_1k.jsonl (questions next,like,dislike,like_para, hist_len 5) against its report.json. Report the three results verbatim. (4) CPU dry run under ${SCRATCH}/dryrun2/ with tiny synthetic data and a FAKE scorer that honours --variant/--readout and writes the contract format: build_confirm_panels (tiny), the stage-1 dev selection end to end (fake scores for 7 variants), stage-2 confirm, diag_battery make+analyze, forensics on a fake pilot dir, pilot1_gate on outputs with the ensemble_null block. (5) bash -n all scripts/sigir/*.sh with C:/Program Files/Git/bin/bash.exe. Return the structured report with honest remaining risks.`, { label: 'integrate', phase: 'Integrate', schema: INTEG })

return { results, integ }
