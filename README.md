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
ALERT_CLASSES=Fall
LABELS=Fall,No_Fall,Pre-Fall,Falling
```

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

### 6.4 Filter ตามความมั่นใจ

```bash
python tools/filter_segments.py --input-csv work_csv/segments_named.csv --output-csv work_csv/segments_filtered.csv --min-score 0.60 --expected-labels Fall,No_Fall,Pre-Fall,Falling
```

แก้ช่วงที่ผิดด้วยสายตา และบันทึกเป็น `work_csv/segments_final.csv`

### 6.5 ตัดคลิปลงโฟลเดอร์คลาส (รองรับ ffmpeg/OpenCV)

```bash
python tools/segments_to_clips.py --video data_long/long_train.mp4 --segments-csv work_csv/segments_final.csv --output-dir data_videos --min-duration 0.5 --backend auto --filename-prefix long_train
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
