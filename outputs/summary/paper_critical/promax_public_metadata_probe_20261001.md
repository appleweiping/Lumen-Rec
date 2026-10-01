# ProMax Public Metadata Probe

Generated: 2026-10-01T14:02:31.327660+00:00

- OK: `true`
- ProMax public metadata ready: `false`
- Final submission ready: `false`
- Network mode: `live`
- DOI: `10.1145/3805712.3809600`

## Blocker Checks

- `promax:final_page_range_missing_in_bib`: closed=`true`
  - closure: Add final ACM page range to the ProMax BibTeX entry.
- `promax:crossref_registry_not_visible`: closed=`true`
  - closure: Crossref /works DOI lookup returns 200 with matching DOI metadata.
- `promax:doi_resolver_not_visible`: closed=`false`
  - closure: DOI resolver returns a successful response for the expected DOI.

## Direct Checks

- Crossref status: `200`
- DOI resolver status: `403`
- ACM DL status: `403`

## Source Probes

- `arxiv_html_promax_acm_metadata`: ok=`true`, status=`200`, missing_patterns=[]
- `sigir2026_accepted_papers_promax`: ok=`true`, status=`200`, missing_patterns=[]
- `uq_author_profile_promax_sigir2026`: ok=`true`, status=`200`, missing_patterns=[]
- `author_publications_promax_sigir2026`: ok=`true`, status=`200`, missing_patterns=[]
- `uq_experts_profile_promax_sigir2026`: ok=`true`, status=`200`, missing_patterns=[]

## Remaining Blockers

- promax:doi_resolver_not_visible

## Warnings

- `acm_dl_not_accessible:status=403`

## Next Actions

- If all blocker checks are closed, update the external proceedings metadata audit and release-candidate stack.
- Do not set final_submission_ready=true from this probe alone; final readiness still requires the full final submission gate.
- If Crossref exposes a page range before BibTeX is updated, copy the final page range into Paper/references.bib and rerun the audits.
