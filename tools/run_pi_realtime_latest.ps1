$ErrorActionPreference = "Stop"

$ModelPath = "models/lstm_fall_model_enhanced_20260315_sample_fp16.tflite"
$ThresholdsPath = "work_csv/compare/recommended_thresholds_20260315_enhanced.json"
$Labels = "Fall,No_Fall,Pre-Fall,Falling"

if (-not (Test-Path $ModelPath)) {
    throw "Model file not found: $ModelPath"
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
  --smooth-method mean `
  --enhance-features
