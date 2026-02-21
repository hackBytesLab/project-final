param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [string]$ModelPath = "models/lstm_fall_model.h5",
    [string]$Labels = "Fall,No_Fall,Pre-Fall,Falling",
    [int]$Timesteps = 30,
    [int]$InferStep = 1,
    [int]$BatchSize = 64,
    [double]$MinScore = 0.60,
    [double]$MinDuration = 0.50,
    [string]$OutputVideo = "work_csv/labeled_preview_named.mp4",
    [switch]$RunTrain,
    [int]$TrainEpochs = 30,
    [int]$TrainBatchSize = 32,
    [string]$TrainOut = "models/lstm_fall_model_v2.h5"
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
  --output-dir data_videos `
  --min-duration $MinDuration

Write-Host "[5/6] Build dataset from labeled clips..."
python video_to_dataset.py `
  --input data_videos `
  --output data `
  --timesteps $Timesteps `
  --step 15

if ($RunTrain) {
    Write-Host "[6/6] Train new model..."
    python train.py `
      --data-dir data `
      --epochs $TrainEpochs `
      --batch-size $TrainBatchSize `
      --out $TrainOut
}
else {
    Write-Host "[6/6] Skipped training (use -RunTrain to enable)."
}

Write-Host "Pipeline completed."
