# Run a multi-line bash script on the GPU server from PowerShell without quoting/encoding damage:
# the script is LF-normalised, base64-encoded (no BOM, no special characters) and decoded remotely.
#   usage (in-process):  & scripts\sigir\remote.ps1 -Text $bashScript      or  -ScriptFile local.sh
param(
  [string]$ScriptFile,
  [string]$Text,
  [string]$Target = "lumen-gpu"
)
if ($ScriptFile) { $Text = [IO.File]::ReadAllText($ScriptFile) }
if (-not $Text) { throw "give -ScriptFile or -Text" }
$bytes = (New-Object System.Text.UTF8Encoding $false).GetBytes(($Text -replace "`r", ""))
$b64 = [Convert]::ToBase64String($bytes)
ssh -o BatchMode=yes -o ServerAliveInterval=30 $Target "echo $b64 | base64 -d | bash"
exit $LASTEXITCODE
