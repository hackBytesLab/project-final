$ErrorActionPreference = "Stop"

$ModelPath = "models/lstm_fall_model_augsplit_acc_20260223_141719.h5"
$ThresholdsPath = "work_csv/compare/recommended_thresholds_20260223_141719.json"
$Labels = "Fall,No_Fall,Pre-Fall,Falling"

if (-not (Test-Path $ModelPath)) {
    throw "Model file not found: $ModelPath"
}

if (-not (Test-Path $ThresholdsPath)) {
    throw "Thresholds file not found: $ThresholdsPath"
}

Write-Host "[RUN] main.py with Pi camera + latest model + latest thresholds"
python main.py `
  --camera pi `
  --model $ModelPath `
  --labels $Labels `
  --thresholds-json $ThresholdsPath `
  --detect-people 4 `
  --inference-mode per-person `
  --normalize-geometry `
  --smooth-window 5 `
  --smooth-method mean
