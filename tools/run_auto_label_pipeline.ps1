param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [string]$ModelPath = "models/lstm_fall_model.h5",
    [string]$Labels = "Fall,No_Fall,Pre-Fall,Falling",
    [int]$Timesteps = 30,
    [int]$InferStep = 1,
    [int]$BatchSize = 64,
    [int]$DatasetStep = 15,
    [string]$ClipsDir = "data_videos",
    [string]$DatasetDir = "data",
    [int]$FrameSplitSize = 0,
    [int]$FrameSplitOverlap = 0,
    [int]$MaxPeople = 1,
    [int]$MaxHands = 2,
    [switch]$NormalizeGeometry,
    [double]$MinScore = 0.60,
    [double]$MinDuration = 0.50,
    [string]$OutputVideo = "work_csv/labeled_preview_named.mp4",
    [switch]$RunTrain,
    [string]$TrainReportsDir = "work_csv/eval_auto",
    [string]$ValidationMode = "holdout-kfold",
    [int]$NumFolds = 5,
    [double]$TestSize = 0.2,
    [string]$SplitUnit = "clip",
    [string]$BalanceMode = "none",
    [int]$TrainEpochs = 30,
    [int]$TrainBatchSize = 32,
    [string]$TrainOut = "models/lstm_fall_model_v2.h5",
    [string]$AugmentMode = "minority",
    [double]$AugmentFactor = 1.0,
    [double]$AugmentMinorityRatio = 0.9,
    [double]$AugmentNoiseStd = 0.01,
    [double]$AugmentScaleRange = 0.05,
    [int]$AugmentTimeShift = 2,
    [double]$AugmentFeatureDropout = 0.01,
    [double]$AugmentTimeMaskRatio = 0.10,
    [string]$LossFunction = "focal",
    [double]$FocalGamma = 2.0,
    [double]$FocalAlpha = 0.25,
    [string]$FocalAlphaMode = "balanced",
    [double]$FocalAlphaCap = 4.0
)

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Prepare workspace folders..."
python tools/prepare_workspace.py

Write-Host "[2/6] Run auto-label inference..."
python infer_video.py `
  --video $VideoPath `
  --model $ModelPath `
  --out-csv work_csv/segments_named.csv `
  --timesteps $Timesteps `
  --step $InferStep `
  --batch-size $BatchSize `
  --labels $Labels `
  --out-video $OutputVideo

Write-Host "[3/6] Filter segments by confidence..."
python tools/filter_segments.py `
  --input-csv work_csv/segments_named.csv `
  --output-csv work_csv/segments_filtered.csv `
  --min-score $MinScore `
  --expected-labels $Labels

Write-Host "[4/6] Cut clips into class folders..."
python tools/segments_to_clips.py `
  --video $VideoPath `
  --segments-csv work_csv/segments_filtered.csv `
  --output-dir $ClipsDir `
  --min-duration $MinDuration

Write-Host "[5/6] Build dataset from labeled clips..."
if (-not (Test-Path $DatasetDir)) {
  New-Item -ItemType Directory -Path $DatasetDir | Out-Null
}

$DatasetArgs = @(
  "video_to_dataset.py",
  "--input", $ClipsDir,
  "--output", $DatasetDir,
  "--timesteps", $Timesteps,
  "--step", $DatasetStep,
  "--labels", $Labels,
  "--max-people", $MaxPeople,
  "--max-hands", $MaxHands
)

if ($FrameSplitSize -gt 0) {
  $DatasetArgs += @("--frame-split-size", $FrameSplitSize)
  if ($FrameSplitOverlap -gt 0) {
    $DatasetArgs += @("--frame-split-overlap", $FrameSplitOverlap)
  }
}
if ($NormalizeGeometry) {
  $DatasetArgs += "--normalize-geometry"
}

python @DatasetArgs

if ($RunTrain) {
    Write-Host "[6/6] Train new model..."
    python train.py `
      --data-dir $DatasetDir `
      --validation-mode $ValidationMode `
      --num-folds $NumFolds `
      --test-size $TestSize `
      --split-unit $SplitUnit `
      --meta-csv "$DatasetDir/sample_meta.csv" `
      --balance-mode $BalanceMode `
      --epochs $TrainEpochs `
      --batch-size $TrainBatchSize `
      --out $TrainOut `
      --reports-dir $TrainReportsDir `
      --labels $Labels `
      --augment-mode $AugmentMode `
      --augment-factor $AugmentFactor `
      --augment-minority-ratio $AugmentMinorityRatio `
      --augment-noise-std $AugmentNoiseStd `
      --augment-scale-range $AugmentScaleRange `
      --augment-time-shift $AugmentTimeShift `
      --augment-feature-dropout $AugmentFeatureDropout `
      --augment-time-mask-ratio $AugmentTimeMaskRatio `
      --loss-function $LossFunction `
      --focal-gamma $FocalGamma `
      --focal-alpha $FocalAlpha `
      --focal-alpha-mode $FocalAlphaMode `
      --focal-alpha-cap $FocalAlphaCap
}
else {
    Write-Host "[6/6] Skipped training (use -RunTrain to enable)."
}

Write-Host "Pipeline completed."
