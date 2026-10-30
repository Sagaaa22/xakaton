param(
    [Parameter(Position=0)]
    [string]$Target = "predict",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Rest
)

$inFile = "tickets.jsonl"
$outFile = "pred.jsonl"

foreach ($arg in $Rest) {
    if ($arg -match "^IN=(.+)$") { $inFile = $Matches[1] }
    elseif ($arg -match "^OUT=(.+)$") { $outFile = $Matches[1] }
}

switch ($Target.ToLower()) {
    "predict" { python -m app.cli predict --in $inFile --out $outFile; exit $LASTEXITCODE }
    "train"   { python -m app.cli train; exit $LASTEXITCODE }
    "evaluate"{ python -m app.cli evaluate --out $outFile; exit $LASTEXITCODE }
    "test"    { python -m unittest discover -s tests -p "test_*.py" -v; exit $LASTEXITCODE }
    "run"     { python -m uvicorn app.main:app --host 127.0.0.1 --port 8000; exit $LASTEXITCODE }
    default   { Write-Error "Unknown target: $Target"; exit 1 }
}

