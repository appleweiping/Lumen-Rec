# Pull the committed result files that feed the paper (the slot aliases of Paper/sigir2027/sections/experiments.tex, FILE
# ALIASES legend) from the GPU server into docs/sigir/results/<alias>/... and write docs/sigir/results/MANIFEST.json
# (relative path, size, sha1, server mtime) for every file found, plus the list of files that do not exist yet.
#
#   usage:  powershell -NoProfile -File scripts\sigir\pull_results.ps1 [-ListOnly] [-Target lumen-gpu]
#
# Read-only on the server: one `stat` listing through scripts\sigir\remote.ps1 -ScriptFile, then one scp per file. Key
# authentication through the ssh host alias (default lumen-gpu, BatchMode: no password prompt, no credential in any file).
# A missing server file is listed under "missing" in the manifest and is never an error; a local copy of it (from an
# earlier pull) is kept and flagged. Downloads go to a temporary name and replace the local file only when the size matches
# the server's. The manifest carries no wall-clock field, so the same server state gives a byte-identical manifest.
# The paper is then regenerated with:  python scripts\sigir\fill_paper.py   (see docs/sigir/PAPER_DATA_MAP.md)
param(
  [string]$Target = "lumen-gpu",
  [string]$ServerRoot = "/root/autodl-tmp/lumen-rec",
  [string]$RepoRoot = "",
  [switch]$ListOnly
)
$ErrorActionPreference = "Stop"
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$ResRoot = Join-Path $RepoRoot "docs\sigir\results"
$Remote = Join-Path $RepoRoot "scripts\sigir\remote.ps1"

# ---- the file list: alias, server path (relative to $ServerRoot), local path (relative to docs/sigir/results)
# Aliases cpu, ko and corr (rated panels) are fields of the grid reports (src/confrec/ftgrid_report.py writes the references,
# the knockout and the lemma/correction checks into report/<d>.json), so they have no file of their own.
$Files = New-Object System.Collections.Generic.List[object]
function Add-F([string]$alias, [string]$server, [string]$local) {
  $Files.Add([pscustomobject]@{ alias = $alias; server = $server; local = $local })
}
Add-F "sel"  "outputs/confrec/gatefix/dev/selection.json"   "sel/selection.json"
Add-F "gate" "outputs/confrec/gatefix/confirm/gate.json"    "gate/gate.json"
Add-F "gft"  "outputs/confrec/gateft/gate_ft.json"          "gft/gate_ft.json"
foreach ($d in "sports", "toys", "home", "tools", "summary") {
  Add-F "aud" "outputs/confrec/nextitem_audit/$d.json" "aud/$d.json"
}
foreach ($d in "sports", "toys", "home", "tools") {
  Add-F "aud2q" "outputs/confrec/nextitem_audit_z2/qwen_$d.json"  "aud2q/$d.json"
  Add-F "aud2l" "outputs/confrec/nextitem_audit_z2/llama_$d.json" "aud2l/$d.json"
}
foreach ($d in "ml1m", "toys", "games", "sports") {
  Add-F "grid" "outputs/confrec/ftgrid/report/$d.json"              "grid/qwen/$d.json"
  Add-F "grid" "outputs/confrec/ftgrid/report/${d}_tables.csv"      "grid/qwen/${d}_tables.csv"
  Add-F "grid" "outputs/confrec/ftgrid/panels/$d/ftgrid_split.json" "grid/qwen/${d}_split.json"
}
foreach ($d in "ml1m", "toys") {
  Add-F "grid" "outputs/confrec/ftgrid_llama/report/$d.json"              "grid/llama/$d.json"
  Add-F "grid" "outputs/confrec/ftgrid_llama/report/${d}_tables.csv"      "grid/llama/${d}_tables.csv"
  Add-F "grid" "outputs/confrec/ftgrid_llama/panels/$d/ftgrid_split.json" "grid/llama/${d}_split.json"
}
Add-F "prn"  "outputs/confrec/ftprune/pruning_ml1m.json" "prn/pruning_ml1m.json"
Add-F "prn"  "outputs/confrec/ftprune/pruning_ml1m.csv"  "prn/pruning_ml1m.csv"
Add-F "slot" "outputs/confrec/ftmethod/slot.json"        "slot/slot.json"
Add-F "slot" "outputs/confrec/ftmethod/slot_tables.csv"  "slot/slot_tables.csv"
foreach ($d in "ml1m", "toys", "games", "sports") {
  Add-F "slot" "outputs/confrec/ftmethod/$d/report.json" "slot/$d.json"
}
Add-F "mir"  "outputs/confrec/gatefix/stage3/decision.json" "mir/decision.json"
# PROPOSED by the skeleton, no producing script yet (listed so the manifest shows them as missing):
Add-F "corr" "outputs/confrec/ftgrid/corrections_sports.json" "corr/sports.json"

# ---- one read-only stat listing on the server
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("cd '$ServerRoot' || { echo 'NOROOT'; exit 0; }")
foreach ($f in $Files) {
  $lines.Add("if [ -f '$($f.server)' ]; then stat -c 'OK %Y %s %n' '$($f.server)'; else echo 'MISSING $($f.server)'; fi")
}
$tmpSh = [IO.Path]::Combine([IO.Path]::GetTempPath(), "pull_results_stat_$PID.sh")
[IO.File]::WriteAllText($tmpSh, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding $false))
try {
  $listing = & $Remote -ScriptFile $tmpSh -Target $Target
  $rc = $LASTEXITCODE
} finally {
  Remove-Item -LiteralPath $tmpSh -ErrorAction SilentlyContinue
}
if ($rc -ne 0 -or -not $listing) { throw "server listing failed (ssh $Target, exit $rc): nothing pulled" }
if (($listing | Select-Object -First 1) -eq "NOROOT") { throw "server root $ServerRoot not found on $Target" }
$stat = @{}
foreach ($ln in $listing) {
  if ($ln -match '^OK (\d+) (\d+) (.+)$') { $stat[$Matches[3]] = @{ mtime = [int64]$Matches[1]; size = [int64]$Matches[2] } }
}

# ---- download
$pulled = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[object]
$errors = New-Object System.Collections.Generic.List[object]
foreach ($f in $Files) {
  $dst = Join-Path $ResRoot ($f.local -replace '/', '\')
  if (-not $stat.ContainsKey($f.server)) {
    $missing.Add([ordered]@{ alias = $f.alias; server_path = $f.server; local_path = $f.local;
                             stale_local_copy_kept = [bool](Test-Path -LiteralPath $dst) })
    continue
  }
  $s = $stat[$f.server]
  $mt = [DateTimeOffset]::FromUnixTimeSeconds($s.mtime).UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
  if (-not $ListOnly) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    $part = "$dst.part"
    & scp -q -o BatchMode=yes "${Target}:$ServerRoot/$($f.server)" "$part"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $s.size) {
      Remove-Item -LiteralPath $part -ErrorAction SilentlyContinue
      $errors.Add([ordered]@{ alias = $f.alias; server_path = $f.server; error = "scp failed or size differs from the server's" })
      continue
    }
    Move-Item -LiteralPath $part -Destination $dst -Force
  }
  $sha = if (Test-Path -LiteralPath $dst) { (Get-FileHash -LiteralPath $dst -Algorithm SHA1).Hash.ToLowerInvariant() } else { $null }
  $pulled.Add([ordered]@{ alias = $f.alias; local_path = $f.local; server_path = $f.server; size = $s.size;
                          sha1 = $sha; server_mtime_utc = $mt })
}

$manifest = [ordered]@{
  schema = "pull_results_v1"
  server = "${Target}:$ServerRoot"
  note = ("cpu, ko and corr (rated panels) are fields of the grid reports; every number of the paper is read from these " +
          "files by scripts/sigir/fill_paper.py (docs/sigir/PAPER_DATA_MAP.md)")
  files = @($pulled | Sort-Object { $_.local_path })
  missing = @($missing | Sort-Object { $_.local_path })
  errors = @($errors | Sort-Object { $_.server_path })
}
if ($ListOnly) {
  "found on the server:"; $pulled | ForEach-Object { "  {0,-6} {1}  ({2} bytes, {3})" -f $_.alias, $_.server_path, $_.size, $_.server_mtime_utc }
  "missing:"; $missing | ForEach-Object { "  {0,-6} {1}" -f $_.alias, $_.server_path }
  exit 0
}
New-Item -ItemType Directory -Force -Path $ResRoot | Out-Null
$json = $manifest | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText((Join-Path $ResRoot "MANIFEST.json"), ($json -replace "`r", "") + "`n", (New-Object Text.UTF8Encoding $false))
"pulled {0} file(s), {1} missing, {2} error(s) -> {3}" -f $pulled.Count, $missing.Count, $errors.Count, (Join-Path $ResRoot "MANIFEST.json")
foreach ($e in $errors) { "  ERROR {0}: {1}" -f $e.server_path, $e.error }
foreach ($m in $missing) { "  missing {0,-6} {1}" -f $m.alias, $m.server_path }
exit 0
