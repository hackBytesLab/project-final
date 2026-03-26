# Fall Detection - Project Final

โปรเจกต์นี้ใช้ MediaPipe (Pose + Hand) + LSTM เพื่อแยกคลาสการเคลื่อนไหว:
- `Fall`
- `No_Fall`
- `Pre-Fall`
- `Falling`

รองรับ workflow:
`วิดีโอยาว -> Auto-label -> ตัดคลิปตามคลาส -> สร้าง dataset -> Retrain -> Deploy`

## 1) ติดตั้ง

```bash
pip install -r requirements.txt
```

Dependencies หลัก:
- `tensorflow`
- `mediapipe`
- `opencv-python`
- `numpy<2`
- `scikit-learn`
- `matplotlib`

## 2) ไฟล์สำคัญ

- `main.py`: real-time inference จากกล้อง + LINE alert
- `infer_video.py`: infer วิดีโอยาวและ export ช่วงเวลา
- `video_to_dataset.py`: แปลงวิดีโอในโฟลเดอร์คลาสเป็น `data/X.npy`, `data/y.npy`
- `train.py`: เทรนโมเดล LSTM + รองรับ `split / kfold / holdout-kfold` และรายงาน ROC/AUC/CM
- `tools/build_dataset_from_long_videos.py`: one-command pipeline สำหรับหลายวิดีโอ (infer -> filter -> cut -> dataset)
- `tools/test_line_alert.py`: ทดสอบส่ง LINE โดยตรง (ไม่ต้องเปิดกล้อง/โมเดล)
- `tools/run_auto_label_pipeline.ps1`: one-shot pipeline บน Windows

## 3) ไฟล์โมเดลที่ต้องมีใน `models/`

- `models/pose_landmarker_lite.task`
- `models/hand_landmarker.task`
- `models/lstm_fall_model.h5` (โมเดลเดิมสำหรับ auto-label)

## 4) ตั้งค่า Environment (อัปเดตล่าสุด)

1. สร้างไฟล์ env local จาก template:

```powershell
Copy-Item .evnv.example .evnv
```

2. ใส่ค่าที่จำเป็นใน `.evnv` (หรือใช้ `.env` ก็ได้):

```env
MODEL_PATH=models/lstm_fall_model.h5
CAMERA_MODE=pi
CAMERA_SOURCE=
LINE_CHANNEL_ACCESS_TOKEN=
LINE_USER_ID=
LINE_COOLDOWN_SECONDS=60
ALERT_CLASSES=Fall,Pre-Fall,Falling
LABELS=Fall,No_Fall,Pre-Fall,Falling
```

> สามารถใช้ alias env `LINE_TOKEN` (แทน `LINE_CHANNEL_ACCESS_TOKEN`) และ `LINE_TO` (แทน `LINE_USER_ID`) ได้ เผื่อใช้ script push แบบสั้น

3. กฎการโหลดค่า:
- `main.py` โหลด `.env` ก่อน แล้ว fallback ไป `.evnv`
- ถ้าใส่ CLI args ค่า CLI จะทับ env

4. ความปลอดภัย:
- เก็บ token เฉพาะไฟล์ local (`.env`/`.evnv`) เท่านั้น
- หาก token เคยถูกแชร์ ให้ rotate token ใหม่และ revoke ตัวเก่าทันที

## 5) ตัวอย่างการแจ้งเตือน LINE (เรียงตามการใช้งานจริง)

### 5.1 ทดสอบส่ง LINE โดยตรงก่อน (แนะนำ)

Broadcast (ไม่ต้องใช้ user id):

```bash
python tools/test_line_alert.py --mode broadcast --token "<LINE_CHANNEL_ACCESS_TOKEN>" --message "[LINE TEST] Broadcast OK"
```

Push (ส่งเฉพาะคน/กลุ่ม/ห้อง):

```bash
python tools/test_line_alert.py --mode push --token "<LINE_CHANNEL_ACCESS_TOKEN>" --user-id "<USER_OR_GROUP_ID>" --message "[LINE TEST] Push OK"
```

เช็กเฉพาะ config โดยไม่ยิง API:

```bash
python tools/test_line_alert.py --mode broadcast --token "dummy" --dry-run
```

### 5.2 รันแจ้งเตือนผ่าน `main.py`

Broadcast เมื่อเจอ `Fall`:

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5 --labels Fall,No_Fall,Pre-Fall,Falling --line-token "<LINE_CHANNEL_ACCESS_TOKEN>" --alert-classes Fall --line-cooldown-seconds 60
```

Push เมื่อเจอ `Fall`:

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5 --labels Fall,No_Fall,Pre-Fall,Falling --line-token "<LINE_CHANNEL_ACCESS_TOKEN>" --line-user-id "<USER_OR_GROUP_ID>" --alert-classes Fall --line-cooldown-seconds 60
```

ผลลัพธ์ที่ควรเห็นใน console:
- เปิดใช้ broadcast: `[LINE] Alert enabled via broadcast API`
- เปิดใช้ push: `[LINE] Alert enabled via push API`
- ส่งสำเร็จ: `[LINE] Alert sent`

## 6) Pipeline เทรนแบบ Step-by-step

### 6.0 Quick Start จาก `Data/train.mp4` + `Data/train2.mp4` (แนะนำ)

```bash
python tools/build_dataset_from_long_videos.py --videos Data/train.mp4,Data/train2.mp4 --model models/lstm_fall_model.h5 --labels Fall,No_Fall,Pre-Fall,Falling --backend auto
```

แล้วเทรนด้วย holdout + k-fold:

```bash
python train.py --data-dir data --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_v2.h5 --reports-dir work_csv/eval --labels Fall,No_Fall,Pre-Fall,Falling
```

### 6.1 เตรียมโฟลเดอร์งาน

```bash
python tools/prepare_workspace.py
```

### 6.2 ตรวจ class order ของโมเดลเดิม

1. เตรียมคลิปอ้างอิงใน `refs/`
2. สร้าง `refs/refs_manifest.csv` ตาม `refs/refs_manifest.example.csv`
3. รัน:

```bash
python tools/verify_class_order.py --manifest refs/refs_manifest.csv --model models/lstm_fall_model.h5 --out-dir work_csv --timesteps 30 --step 1 --batch-size 64
```

### 6.3 Auto-label วิดีโอยาว

```bash
python infer_video.py --video data_long/long_train.mp4 --model models/lstm_fall_model.h5 --out-csv work_csv/segments_named.csv --timesteps 30 --step 1 --batch-size 64 --labels Fall,No_Fall,Pre-Fall,Falling --out-video work_csv/labeled_preview_named.mp4
```

ถ้าต้องการ export รูปตัวอย่างตอน test/infer แยกตามทั้ง 4 คลาส:

```bash
python infer_video.py --video data_long/long_train.mp4 --model models/lstm_fall_model.h5 --out-csv work_csv/segments_named.csv --timesteps 30 --step 1 --batch-size 64 --labels Fall,No_Fall,Pre-Fall,Falling --out-video work_csv/labeled_preview_named.mp4 --save-class-frames-dir work_csv/test_class_frames --max-frames-per-class 10
```

ผลลัพธ์จะถูกบันทึกเป็นโฟลเดอร์แยกคลาส เช่น `00_Fall/`, `01_No_Fall/`, `02_Pre-Fall/`, `03_Falling/` พร้อมไฟล์ `saved_frames.csv`

### 6.4 Filter ตามความมั่นใจ

```bash
python tools/filter_segments.py --input-csv work_csv/segments_named.csv --output-csv work_csv/segments_filtered.csv --min-score 0.60 --expected-labels Fall,No_Fall,Pre-Fall,Falling
```

แก้ช่วงที่ผิดด้วยสายตา และบันทึกเป็น `work_csv/segments_final.csv`

### 6.5 ตัดคลิปลงโฟลเดอร์คลาส (รองรับ ffmpeg/OpenCV)

```bash
python tools/segments_to_clips.py --video data_long/long_train.mp4 --segments-csv work_csv/segments_final.csv --output-dir data_videos --min-duration 0.5 --max-duration 8 --backend auto --filename-prefix long_train
```

### 6.6 สร้าง dataset

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15 --labels Fall,No_Fall,Pre-Fall,Falling
```

ผลลัพธ์ที่ต้องได้:
- `data/X.npy`
- `data/y.npy`
- `data/class_map.json`
- `data/sample_meta.csv`

### 6.7 เทรนโมเดลรอบใหม่

```bash
python train.py --data-dir data --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_v2.h5 --reports-dir work_csv/eval --labels Fall,No_Fall,Pre-Fall,Falling
```

### 6.8 Deploy และรัน real-time

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5 --labels Fall,No_Fall,Pre-Fall,Falling
```

## 7) One-shot Pipeline (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_auto_label_pipeline.ps1 -VideoPath data_long/long_train.mp4 -ModelPath models/lstm_fall_model.h5 -Labels Fall,No_Fall,Pre-Fall,Falling -RunTrain -TrainOut models/lstm_fall_model_v2.h5
```

## 8) Checklist ก่อนถือว่าใช้งานได้

1. ทดสอบ `tools/test_line_alert.py` แล้วส่งเข้า LINE สำเร็จ
2. มี `data/X.npy` และ `data/y.npy`
3. มี `data/sample_meta.csv`
4. เทรนแล้วได้ `models/lstm_fall_model_v2.h5`
5. ได้รายงานครบใน `work_csv/eval/` เช่น:
- `work_csv/eval/cv/fold_1/metrics_summary.json` ... `fold_5`
- `work_csv/eval/holdout/confusion_matrix_raw.png`
- `work_csv/eval/holdout/confusion_matrix_norm.png`
- `work_csv/eval/holdout/roc_curve.png`
- `work_csv/eval/holdout/auc_summary.json`
- `work_csv/eval/summary/overview.json`
- `work_csv/eval/summary/overview.md`
- `work_csv/eval/summary/learning_curves.png`
6. รัน `main.py` แล้วพยากรณ์ได้ต่อเนื่องและมี alert ตามคลาสที่ตั้ง

## 9) หมายเหตุสำคัญ

- ใช้ label เดียวกันตลอด pipeline: `Fall,No_Fall,Pre-Fall,Falling`
- `timesteps` ต้องสอดคล้องกันใน `infer_video.py`, `video_to_dataset.py`, `train.py`
- Auto-label เป็น pseudo-label ควรตรวจและแก้ก่อน retrain
- อย่า hardcode token ในโค้ด ให้ส่งผ่าน env หรือ CLI เท่านั้น

## 10) Multi-person + Geometry Update (2026-02-22)

This update adds runtime support for both single-person and multi-person workflows.

### 10.1 Runtime Modes in `main.py`

`--inference-mode` has 3 options:
- `auto`: uses `per-person` automatically when model input is 150 features and `--detect-people > 1`; otherwise uses `frame`
- `frame`: one prediction for the whole frame sequence
- `per-person`: one prediction per tracked person (P1..P4)

Examples:

```bash
# Single-person runtime (stable baseline)
python main.py --model models/lstm_fall_model.h5 --detect-people 1 --inference-mode frame
```

```bash
# Multi-person runtime with single-person (150-feature) model
python main.py --model models/lstm_fall_model.h5 --detect-people 4 --inference-mode per-person
```

```bash
# Auto mode (recommended default)
python main.py --model models/lstm_fall_model.h5 --detect-people 4 --inference-mode auto
```

### 10.2 Person Tracking Controls (`main.py`)

In `per-person` mode, track association is center-distance based.

- `--track-max-distance` (default `0.20`): max normalized center distance for matching detections to tracks
- `--track-max-missed` (default `15`): how many missed frames before resetting a track

Example:

```bash
python main.py --model models/lstm_fall_model.h5 --detect-people 4 --inference-mode per-person --track-max-distance 0.20 --track-max-missed 15
```

### 10.3 Geometry Normalization

`--normalize-geometry` is available in:
- `main.py`
- `infer_video.py`
- `video_to_dataset.py`
- `tools/build_dataset_from_long_videos.py`

Important:
- Use the same normalization setting in dataset generation, training, and inference.
- If you train without normalization, do not enable it at runtime for that model.

Example dataset/training/runtime with normalization:

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15 --labels Fall,No_Fall,Pre-Fall,Falling --max-people 1 --max-hands 2 --normalize-geometry
python train.py --data-dir data --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_norm.h5 --reports-dir work_csv/eval --labels Fall,No_Fall,Pre-Fall,Falling
python main.py --model models/lstm_fall_model_norm.h5 --detect-people 4 --inference-mode per-person --normalize-geometry
```

### 10.4 Pipeline Updates

`tools/build_dataset_from_long_videos.py` now supports:
- `--max-people`
- `--max-hands`
- `--normalize-geometry`

Example:

```bash
python tools/build_dataset_from_long_videos.py --videos Data/train.mp4,Data/train2.mp4 --model models/lstm_fall_model.h5 --labels Fall,No_Fall,Pre-Fall,Falling --backend auto --max-people 1 --max-hands 2 --normalize-geometry
```

### 10.5 `.evnv.example` New Keys

New env keys:
- `INFERENCE_MODE=auto`
- `NORMALIZE_GEOMETRY=0`
- `TRACK_MAX_DISTANCE=0.20`
- `TRACK_MAX_MISSED=15`

## 11) Recommended 5-Fold Training From `train.mp4` (2026-02-22)

Use this flow when you need `num-folds=5` to run reliably with full report artifacts.

### 11.1 Source Video

If `Data/train.mp4` is unavailable, use:

`C:\Users\PC\AppData\Local\CapCut\Videos\work\train\train.mp4`

### 11.2 Build `segments_final_train.csv`

Run auto-label and filtering first:

```bash
python infer_video.py --video "C:\Users\PC\AppData\Local\CapCut\Videos\work\train\train.mp4" --model models/smoke_model.h5 --out-csv work_csv/segments_named_train.csv --timesteps 30 --step 1 --batch-size 64 --labels Fall,No_Fall,Pre-Fall,Falling --out-video work_csv/labeled_preview_train.mp4
python tools/filter_segments.py --input-csv work_csv/segments_named_train.csv --output-csv work_csv/segments_filtered_train.csv --min-score 0.25 --expected-labels Fall,No_Fall,Pre-Fall,Falling
```

Then review and save to:

`work_csv/segments_final_train.csv`

### 11.3 Cut Clips

```bash
python tools/segments_to_clips.py --video "C:\Users\PC\AppData\Local\CapCut\Videos\work\train\train.mp4" --segments-csv work_csv/segments_final_train.csv --output-dir data_videos_chunked --min-duration 0.1 --max-duration 8 --backend auto --filename-prefix train_chunk
```

### 11.4 Build Dataset For 5-Fold

Use `step=1` to increase sample count for minority classes:

```bash
python video_to_dataset.py --input data_videos_chunked --output data_5fold_chunked --timesteps 30 --step 1 --labels Fall,No_Fall,Pre-Fall,Falling --max-people 1 --max-hands 2
```

Frame split example (increase groups/samples from long clips):

```bash
python video_to_dataset.py --input data_videos_chunked --output data_5fold_chunked_split --timesteps 30 --step 1 --frame-split-size 180 --frame-split-overlap 60 --labels Fall,No_Fall,Pre-Fall,Falling --max-people 1 --max-hands 2
```

### 11.5 Train + Evaluate (`num-folds=5`)

```bash
python train.py --data-dir data_5fold_chunked --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data_5fold_chunked/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_v2_chunked.h5 --reports-dir work_csv/eval_chunked --labels Fall,No_Fall,Pre-Fall,Falling --balance-mode class_weight
```

Train with augmentation example:

```bash
python train.py --data-dir data_5fold_chunked_split --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data_5fold_chunked_split/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_aug.h5 --reports-dir work_csv/eval_aug --labels Fall,No_Fall,Pre-Fall,Falling --balance-mode class_weight --augment-mode minority --augment-factor 1.0 --augment-minority-ratio 0.9 --augment-noise-std 0.01 --augment-scale-range 0.05 --augment-time-shift 2 --augment-feature-dropout 0.01 --augment-time-mask-ratio 0.10
```

High-accuracy recipe (frame split + focal + augmentation):

```bash
python train.py --data-dir data_5fold_chunked_split --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data_5fold_chunked_split/sample_meta.csv --epochs 30 --batch-size 32 --patience 7 --out models/lstm_fall_model_augsplit_acc.h5 --reports-dir work_csv/eval_augsplit_acc --labels Fall,No_Fall,Pre-Fall,Falling --balance-mode none --augment-mode minority --augment-factor 1.0 --augment-minority-ratio 0.85 --augment-noise-std 0.006 --augment-scale-range 0.03 --augment-time-shift 1 --augment-feature-dropout 0.003 --augment-time-mask-ratio 0.05 --loss-function focal --focal-gamma 1.2 --focal-alpha 0.25 --focal-alpha-mode fixed --focal-alpha-cap 4.0
```

### 11.6 Expected Outputs

- `models/lstm_fall_model_v2_5fold.h5`
- `work_csv/eval_5fold/cv/fold_1` ... `fold_5`
- `work_csv/eval_5fold/holdout/confusion_matrix_raw.png`
- `work_csv/eval_5fold/holdout/confusion_matrix_norm.png`
- `work_csv/eval_5fold/holdout/roc_curve.png`
- `work_csv/eval_5fold/holdout/auc_summary.json`
- `work_csv/eval_5fold/summary/overview.md`

## 12) Focal Loss + Threshold Tuning (2026-02-22)

### 12.1 Train with Focal Loss

```bash
python train.py --data-dir data_5fold_chunked --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data_5fold_chunked/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_focal.h5 --reports-dir work_csv/eval_focal --labels Fall,No_Fall,Pre-Fall,Falling --balance-mode class_weight --loss-function focal --focal-gamma 2.0 --focal-alpha 0.25
```

Notes:
- `--loss-function categorical_crossentropy|focal`
- `--focal-gamma` and `--focal-alpha` are used only when `--loss-function focal`
- `--focal-alpha-mode fixed|balanced` (`balanced` builds per-class alpha from train-set frequency)
- `--focal-alpha-cap` caps large alpha values in balanced mode

Example (balanced focal alpha):

```bash
python train.py --data-dir data_5fold_chunked --validation-mode holdout-kfold --num-folds 5 --test-size 0.2 --split-unit clip --meta-csv data_5fold_chunked/sample_meta.csv --epochs 30 --batch-size 32 --out models/lstm_fall_model_focal_balanced.h5 --reports-dir work_csv/eval_focal_balanced --labels Fall,No_Fall,Pre-Fall,Falling --balance-mode none --loss-function focal --focal-gamma 2.0 --focal-alpha-mode balanced --focal-alpha-cap 4.0
```

### 12.2 Recommend Per-Class Thresholds from ROC

```bash
python tools/recommend_class_thresholds.py --eval-dir work_csv/eval_focal --labels Fall,No_Fall,Pre-Fall,Falling --output-json work_csv/compare/recommended_thresholds.json --output-md work_csv/compare/recommended_thresholds.md
```

### 12.3 Runtime with Thresholds

```bash
python main.py --camera pi --model models/lstm_fall_model_focal.h5 --labels Fall,No_Fall,Pre-Fall,Falling --thresholds-json work_csv/compare/recommended_thresholds.json
```

Compatibility note:
- Runtime/inference now loads Keras models with `compile=False`, so focal-loss models can be used without custom loss registration.

## 13) Raspberry Pi 5 Optimization (TFLite)

Use TFLite model on Pi 5 to reduce memory and speed up inference.

### 13.0 Pre-deploy check (run before copying model to board)

```bash
python3 tools/predeploy_board_check.py --model models/<your_model>.h5 --labels Fall,No_Fall,Pre-Fall,Falling --check-imports
```

For TFLite:

```bash
python3 tools/predeploy_board_check.py --model models/<your_model>.tflite --labels Fall,No_Fall,Pre-Fall,Falling --thresholds-json work_csv/compare/recommended_thresholds.json --check-imports
```

Interpretation:
- Exit code `0` = all checks pass
- Exit code `1` = warning only (review before deploy)
- Exit code `2` = fail (must fix before deploy)

### 13.0.1 One-command deploy to board (scp + install + precheck + run)

```bash
chmod +x tools/deploy_tflite_to_board.sh
tools/deploy_tflite_to_board.sh \
  --board-user <user> \
  --board-host <ip_or_hostname> \
  --board-dir ~/project-final \
  --model-local models/<your_model>.tflite \
  --thresholds-local work_csv/compare/recommended_thresholds.json \
  --labels Fall,No_Fall,Pre-Fall,Falling \
  --model-threads 4
```

Options:
- `--skip-install` to skip `pip install` steps on board
- `--allow-precheck-warn` to continue when precheck exits `1`

Test scenarios:
- Missing model:
  - set `--model-local` to a wrong path -> script stops before upload
- Threshold mismatch warning:
  - use a thresholds file with unknown keys, then add `--allow-precheck-warn` if you want to continue
- Missing runtime dependency:
  - uninstall `mediapipe`/`tensorflow` on board and run without `--skip-install` (install step should recover)

### 13.1 Export `.h5` to `.tflite` for Pi 5 (built-in ops only)

```bash
python tools/export_tflite.py --keras-model models/lstm_all_combined_216_sub8k_prefallboost.h5 --output models/lstm_all_combined_216_sub8k_prefallboost_pi5.tflite --quantization float16 --lite-friendly-rnn --fixed-timesteps 30 --require-builtins-only
```

What this does:
- rebuilds the Sequential LSTM with fixed `30` timesteps
- exports with `unroll=True` so the Pi 5 runtime can use built-in TFLite ops only
- verifies that the final `.tflite` contains no Flex ops

If you still want a generic TFLite export that allows Flex / `SELECT_TF_OPS`, use:

```bash
python tools/export_tflite.py --keras-model models/lstm_all_combined_216_sub8k_prefallboost.h5 --output models/lstm_all_combined_216_sub8k_prefallboost.tflite --quantization float16 --select-tf-ops
```

### 13.2 Install runtime on Pi 5

```bash
pip install tflite-runtime mediapipe opencv-python numpy
```

Notes:
- For the new `*_pi5.tflite` export above, `tflite-runtime` is enough because it does not require Flex ops.
- If you run an older `.tflite` exported with `--select-tf-ops`, install `tensorflow` instead of `tflite-runtime`.

### 13.3 Run real-time on Pi 5 with the Pi 5 TFLite runtime

```bash
python pi5_runtime.py --camera picamera2
```

USB camera:

```bash
python pi5_runtime.py --camera index --camera-index 0
```

Notes:
- Default model is `models/lstm_all_combined_216_sub8k_prefallboost_pi5.tflite`
- Expected input is fixed at `30 x 216`
- The runtime applies the same enhanced feature pipeline used by this export path
- Tune `--model-threads` (Pi 5 usually `2-4` is a good range)
