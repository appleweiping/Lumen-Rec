# Sync the SIGIR-2027 work from the working folder (D:\Research\Lumen, whose .git is damaged) into the clean
# clone (D:\Research\_lumen_fresh, branch sigir2027), then commit + push. Copy-only: never touches Lumen\.git.
#   usage:  powershell -File scripts\sigir\sync_to_clean_clone.ps1 -Message "confrec: ..."  [-NoPush]
param(
  [Parameter(Mandatory = $true)][string]$Message,
  [string]$Src = "D:\Research\Lumen",
  [string]$Dst = "D:\Research\_lumen_fresh",
  [switch]$NoPush
)
$ErrorActionPreference = "Stop"
$dirs = @("docs\sigir", "idea-stage", "src\confrec", "scripts\sigir", "Paper\sigir2027", "outputs\confrec_pilot")
$files = @("RESEARCH_BRIEF.md", "refine-logs\EXPERIMENT_PLAN.md")
$globs = @("tests\test_confrec_*.py")

foreach ($d in $dirs) {
  if (Test-Path "$Src\$d") {
    # /MIR mirrors the SIGIR-owned folder only; LaTeX build products are excluded.
    robocopy "$Src\$d" "$Dst\$d" /MIR /NFL /NDL /NJH /NJS /NP /XF *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg /XD __pycache__ | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $d ($LASTEXITCODE)" }
  }
}
foreach ($f in $files) {
  New-Item -ItemType Directory -Force (Split-Path "$Dst\$f") | Out-Null
  Copy-Item "$Src\$f" "$Dst\$f" -Force
}
foreach ($g in $globs) {
  Get-ChildItem "$Src\$g" | ForEach-Object { Copy-Item $_.FullName "$Dst\tests\$($_.Name)" -Force }
}

Set-Location $Dst
if ((git branch --show-current) -ne "sigir2027") { throw "clean clone is not on branch sigir2027" }
git add -A -- docs/sigir idea-stage src/confrec scripts/sigir Paper/sigir2027 outputs/confrec_pilot RESEARCH_BRIEF.md refine-logs tests CLAUDE.md
$staged = git diff --cached --name-only
if (-not $staged) { Write-Output "nothing to commit"; exit 0 }
git commit -q -m "$Message" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
if (-not $NoPush) { git push -u origin sigir2027 2>&1 | Select-Object -Last 3 }
