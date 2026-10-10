# Build <team_name>_submission.zip in the layout the organisers' README asks for:
#   output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/, Documentation_template.md
# Validates the two TSVs against dataset/test first and refuses to package a failing submission.
#
#   .\make_submission.ps1 -Team <team_name>
param([Parameter(Mandatory = $true)][string]$Team)
$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
Set-Location $Root
$Match = "output/matching_results.tsv"
$Cand = "output/candidate_pairs.tsv"
$Code = "code/business_entity_resolution"
$Doc = "Documentation_template.md"
$Zip = Join-Path $Root "${Team}_submission.zip"

foreach ($f in @($Match, $Cand, $Doc, "$Code/README.md", "$Code/requirements.txt")) {
    if (-not (Test-Path $f -PathType Leaf)) { throw "missing: $f" }
}
if (-not (Test-Path "$Code/src" -PathType Container)) { throw "missing: $Code/src" }

$tbd = Select-String -Path $Doc -Pattern "TBD"
if ($tbd) {
    Write-Warning "$Doc still contains TBD placeholders:"
    $tbd | ForEach-Object { Write-Warning ("  line {0}: {1}" -f $_.LineNumber, $_.Line.Trim()) }
}

if (-not (Test-Path "dataset/test" -PathType Container)) { throw "dataset/test not found: cannot validate, refusing to package" }
if (-not (Test-Path "utils/validate_submission.py" -PathType Leaf)) { throw "utils/validate_submission.py not found: it is the organisers'' validator (not redistributed in this repo; copy it from the challenge''s student_resource to utils/validate_submission.py)" }
$env:PYTHONIOENCODING = "utf-8"
python utils/validate_submission.py --matching $Match --candidate $Cand --test-dir dataset/test
if ($LASTEXITCODE -ne 0) { throw "validator failed: not packaging" }

$Stage = Join-Path ([System.IO.Path]::GetTempPath()) ("sub_" + [guid]::NewGuid())
try {
    New-Item -ItemType Directory -Path "$Stage/output", "$Stage/code" | Out-Null
    Copy-Item $Match, $Cand "$Stage/output/"
    Copy-Item $Doc "$Stage/"
    Copy-Item $Code "$Stage/code/" -Recurse
    # no venv, caches or bytecode
    Get-ChildItem "$Stage/code" -Recurse -Force -Directory |
        Where-Object { $_.Name -in @(".venv", "__pycache__", ".pytest_cache") } |
        Remove-Item -Recurse -Force
    Get-ChildItem "$Stage/code" -Recurse -Force -Filter *.pyc | Remove-Item -Force

    if (Test-Path $Zip) { Remove-Item $Zip }
    Compress-Archive -Path "$Stage/output", "$Stage/code", "$Stage/$Doc" -DestinationPath $Zip
}
finally {
    Remove-Item $Stage -Recurse -Force -ErrorAction SilentlyContinue
}

$size = "{0:N1} MB" -f ((Get-Item $Zip).Length / 1MB)
Write-Host "`nwrote $Zip ($size)"
