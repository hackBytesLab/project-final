# Fall Detection - Project Final

โปรเจกต์นี้ใช้ MediaPipe (Pose + Hand) + LSTM เพื่อแยกคลาสการเคลื่อนไหว:
- `Fall`
- `No_Fall`
- `Pre-Fall`
- `Falling`

รองรับ workflow แบบ:
วิดีโอยาวไฟล์เดียว -> Auto-label -> ตัดคลิปตามคลาส -> สร้าง dataset -> Retrain -> Deploy

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

หมายเหตุ: ใช้ `numpy<2` เพื่อให้เข้ากับ TensorFlow/MediaPipe ใน environment ปัจจุบัน

## 2) ไฟล์สำคัญ

- `main.py`: real-time inference จากกล้อง (`iriun|pi|rtsp|index`) + LINE alert
- `infer_video.py`: auto-label วิดีโอยาวและ export `segments.csv`
- `video_to_dataset.py`: แปลงวิดีโอที่แยกคลาสเป็น `data/X.npy`, `data/y.npy`
- `train.py`: เทรนโมเดล LSTM
- `tools/prepare_workspace.py`: สร้างโฟลเดอร์งานมาตรฐาน
- `tools/verify_class_order.py`: ตรวจ `class_id` ของโมเดลจาก reference clips
- `tools/filter_segments.py`: filter segments ตาม `avg_score`
- `tools/segments_to_clips.py`: ตัดคลิปลงโฟลเดอร์คลาสด้วย ffmpeg
- `tools/run_auto_label_pipeline.ps1`: one-shot pipeline (Windows)

## 3) ไฟล์โมเดลที่ต้องมีใน `models/`

- `models/pose_landmarker_lite.task`
- `models/hand_landmarker.task`
- `models/lstm_fall_model.h5` (โมเดลเดิมสำหรับ auto-label)

## 4) ค่า default ใน `.env`

ตอนนี้ตั้งค่าเริ่มต้นไว้เป็น Pi Camera:
- `CAMERA_MODE=pi`
- `MODEL_PATH=models/lstm_fall_model.h5`
- `ALERT_CLASSES=Fall`
- `LINE_COOLDOWN_SECONDS=60`

มีตัวแปร LINE:
- `LINE_CHANNEL_ACCESS_TOKEN=`
- `LINE_USER_ID=`

หมายเหตุ:
- `main.py` จะโหลดค่าจาก `.env` อัตโนมัติ (ถ้ามี)
- CLI arguments จะ override ค่าจากไฟล์เสมอ

## 5) Step-by-step ตั้งแต่เริ่มต้น

### Step 1: เตรียมโฟลเดอร์งาน

```bash
python tools/prepare_workspace.py
```

จะได้โฟลเดอร์หลัก:
- `data_long/`
- `work_csv/`
- `data_videos/Fall`
- `data_videos/No_Fall`
- `data_videos/Pre-Fall`
- `data_videos/Falling`
- `refs/`

### Step 2: ตรวจ class order ของโมเดลเดิม

1. เตรียมคลิปอ้างอิงที่รู้คลาสแน่ชัดใน `refs/`
2. สร้าง `refs/refs_manifest.csv` ตามตัวอย่าง `refs/refs_manifest.example.csv`

ตัวอย่าง:

```csv
class_name,video_path
Fall,refs/fall_ref.mp4
No_Fall,refs/no_fall_ref.mp4
Pre-Fall,refs/pre_fall_ref.mp4
Falling,refs/falling_ref.mp4
```

3. รันตรวจ:

```bash
python tools/verify_class_order.py --manifest refs/refs_manifest.csv --model models/lstm_fall_model.h5 --out-dir work_csv --timesteps 30 --step 1 --batch-size 64
```

ผลลัพธ์:
- `work_csv/class_order_report.csv`
- `work_csv/class_order.json`
- `work_csv/suggested_labels.txt`

### Step 3: Auto-label วิดีโอยาว

```bash
python infer_video.py --video data_long/long_train.mp4 --model models/lstm_fall_model.h5 --out-csv work_csv/segments_named.csv --timesteps 30 --step 1 --batch-size 64 --labels Fall,No_Fall,Pre-Fall,Falling --out-video work_csv/labeled_preview_named.mp4
```

### Step 4: Filter ตามความมั่นใจ

```bash
python tools/filter_segments.py --input-csv work_csv/segments_named.csv --output-csv work_csv/segments_filtered.csv --min-score 0.60 --expected-labels Fall,No_Fall,Pre-Fall,Falling
```

จากนั้นตรวจด้วยสายตาและแก้ช่วงผิด ถ้าต้องการ:
- ใช้ `work_csv/labeled_preview_named.mp4` + `work_csv/segments_filtered.csv`
- เซฟสุดท้ายเป็น `work_csv/segments_final.csv`

### Step 5: ตัดคลิปลงโฟลเดอร์คลาส (ต้องมี ffmpeg)

```bash
python tools/segments_to_clips.py --video data_long/long_train.mp4 --segments-csv work_csv/segments_final.csv --output-dir data_videos --min-duration 0.5
```

### Step 6: สร้าง dataset

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15 --labels Fall,No_Fall,Pre-Fall,Falling
```

ผลลัพธ์:
- `data/X.npy`
- `data/y.npy`
- `data/class_map.json`

### Step 7: เทรนโมเดลรอบใหม่

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --out models/lstm_fall_model_v2.h5 --eval-dir work_csv --labels Fall,No_Fall,Pre-Fall,Falling
```

### Step 8: Deploy และรัน real-time (Pi Camera)

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5 --labels Fall,No_Fall,Pre-Fall,Falling
```

## 6) LINE Alert (เมื่อพบการล้ม)

ตัวอย่างส่งแจ้งเตือนเมื่อเจอ `Fall`:

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5 --labels Fall,No_Fall,Pre-Fall,Falling --line-token "<LINE_CHANNEL_ACCESS_TOKEN>" --line-user-id "<USER_OR_GROUP_ID>" --alert-classes Fall --line-cooldown-seconds 60
```

ตัวเลือกสำคัญ:
- `--line-token`: Channel Access Token (จำเป็นถ้าจะส่ง LINE)
- `--line-user-id`: ถ้าใส่จะใช้ Push API
- ถ้าไม่ใส่ `--line-user-id` จะใช้ Broadcast API
- `--alert-classes`: คลาสที่ให้ส่งแจ้งเตือน (คั่นด้วย comma ได้)
- `--line-cooldown-seconds`: กันแจ้งเตือนถี่เกินไป

## 7) One-shot Pipeline (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_auto_label_pipeline.ps1 -VideoPath data_long/long_train.mp4 -ModelPath models/lstm_fall_model.h5 -Labels Fall,No_Fall,Pre-Fall,Falling -RunTrain -TrainOut models/lstm_fall_model_v2.h5
```

## 8) Test Checklist

- มี `work_csv/segments_named.csv` และคอลัมน์ครบ
- หลัง filter ยังมีคลาสสำคัญที่ต้องการ
- `data_videos/<class>/` มีไฟล์จริง
- `data/X.npy` เป็น 3 มิติ และ `X.shape[2] == 150`
- ตรวจ `data/class_map.json` ว่า order ตรงกับ `--labels`
- เทรนแล้วได้ `models/lstm_fall_model_v2.h5`
- ได้ metric report ที่ `work_csv/`:
  - `confusion_matrix.csv`
  - `classification_report.json`
  - `classification_report.txt`
  - `metrics_summary.json` (รวม accuracy, precision, recall, f1)
- รัน `main.py` แล้วพยากรณ์ได้ต่อเนื่อง

## 9) หมายเหตุ

- Auto-label คือ pseudo-label ควรตรวจแก้ก่อน retrain
- `timesteps` ต้องสอดคล้องกันใน infer/dataset/train
- ให้ใช้ label เดียวกันตลอด pipeline (infer/dataset/main): `Fall,No_Fall,Pre-Fall,Falling`
- ถ้า class order ไม่ชัด ให้รัน `tools/verify_class_order.py` ก่อนทุกครั้ง
- อย่า hardcode token ในโค้ด ให้ส่งผ่าน argument หรือ environment variable
