# Fall Detection - Project Final

โปรเจกต์นี้ใช้ MediaPipe (Pose + Hand) ดึง landmark จากวิดีโอ แล้วใช้ LSTM แยกคลาส:
- `Fall`
- `No_Fall`
- `Pre-Fall`
- `Falling`

รองรับงานแบบวิดีโอยาวไฟล์เดียว -> Auto-label -> Retrain -> Deploy

## สถานะงานปัจจุบัน

ตอนนี้มีเครื่องมือครบตาม workflow แล้ว:
- `main.py` รัน real-time inference (`iriun|pi|rtsp|index`)
- `infer_video.py` แยกช่วงคลาสจากวิดีโอยาวและ export CSV
- `video_to_dataset.py` แปลงคลิปที่ติดป้ายแล้วเป็น `X.npy`, `y.npy`
- `train.py` เทรนโมเดล LSTM
- `tools/prepare_workspace.py` สร้างโฟลเดอร์งานอัตโนมัติ
- `tools/verify_class_order.py` ตรวจ `class_id` ของโมเดลจาก reference clips
- `tools/filter_segments.py` คัด segment ด้วย `avg_score`
- `tools/segments_to_clips.py` ตัดคลิปลงโฟลเดอร์คลาสด้วย ffmpeg
- `tools/run_auto_label_pipeline.ps1` รัน pipeline ต่อเนื่องบน Windows

## ติดตั้ง

```bash
pip install -r requirements.txt
```

แพ็กเกจหลัก:
- `tensorflow`
- `mediapipe`
- `opencv-python`
- `numpy<2`
- `scikit-learn`

หมายเหตุสำคัญ: ใช้ `numpy<2` เพื่อให้เข้ากับ TensorFlow/MediaPipe ใน environment ปัจจุบัน

## ไฟล์ที่ต้องมีใน `models/`

- `models/pose_landmarker_lite.task`
- `models/hand_landmarker.task`
- `models/lstm_fall_model.h5` (โมเดลเดิมสำหรับ auto-label)

## แผนตั้งแต่เริ่มต้น (Step-by-step)

### Step 1) เตรียม workspace

```bash
python tools/prepare_workspace.py
```

จะได้โฟลเดอร์หลัก:
- `data_long/` วิดีโอยาวต้นฉบับ
- `work_csv/` ไฟล์ segment ระหว่างทาง
- `data_videos/Fall`
- `data_videos/No_Fall`
- `data_videos/Pre-Fall`
- `data_videos/Falling`
- `refs/` คลิปอ้างอิงสำหรับตรวจ class order

### Step 2) ตรวจ class order ของโมเดลเดิม (ห้ามข้าม)

1. วางคลิปอ้างอิงที่รู้คลาสแน่ชัดไว้ใน `refs/`
2. สร้าง `refs/refs_manifest.csv` โดยใช้ตัวอย่าง `refs/refs_manifest.example.csv`

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

ใช้ค่าจาก `suggested_labels.txt` กับ `--labels` ในขั้น auto-label

### Step 3) Auto-label วิดีโอยาว

```bash
python infer_video.py --video data_long/long_train.mp4 --model models/lstm_fall_model.h5 --out-csv work_csv/segments_named.csv --timesteps 30 --step 1 --batch-size 64 --labels Fall,No_Fall,Pre-Fall,Falling --out-video work_csv/labeled_preview_named.mp4
```

### Step 4) คัดกรอง segment ตามความมั่นใจ

```bash
python tools/filter_segments.py --input-csv work_csv/segments_named.csv --output-csv work_csv/segments_filtered.csv --min-score 0.60 --expected-labels Fall,No_Fall,Pre-Fall,Falling
```

จากนั้นตรวจด้วยสายตา:
- เปิด `work_csv/labeled_preview_named.mp4`
- เปิด `work_csv/segments_filtered.csv`
- แก้ช่วงที่ผิด และบันทึกเป็น `work_csv/segments_final.csv`

ถ้าไม่แก้ ใช้ `segments_filtered.csv` ต่อได้

### Step 5) ตัดคลิปลงโฟลเดอร์คลาส (ต้องมี ffmpeg)

```bash
python tools/segments_to_clips.py --video data_long/long_train.mp4 --segments-csv work_csv/segments_final.csv --output-dir data_videos --min-duration 0.5
```

เงื่อนไข:
- segment ที่สั้นกว่า `0.5s` จะถูกข้าม
- ชื่อไฟล์ออกเป็น `seg_000001.mp4`, `seg_000002.mp4`, ...

### Step 6) สร้าง dataset สำหรับเทรน

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15
```

ต้องได้:
- `data/X.npy`
- `data/y.npy`

### Step 7) เทรนโมเดลรอบใหม่

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --out models/lstm_fall_model_v2.h5
```

### Step 8) Deploy และทดสอบ real-time

Pi Camera:

```bash
python main.py --camera pi --model models/lstm_fall_model_v2.h5
```

## รันแบบคำสั่งเดียว (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_auto_label_pipeline.ps1 -VideoPath data_long/long_train.mp4 -ModelPath models/lstm_fall_model.h5 -Labels Fall,No_Fall,Pre-Fall,Falling -RunTrain -TrainOut models/lstm_fall_model_v2.h5
```

Flow นี้จะทำ:
1. เตรียมโฟลเดอร์
2. auto-label
3. filter segment
4. cut คลิปเข้าคลาส
5. สร้าง dataset
6. เทรน (เมื่อใส่ `-RunTrain`)

## Test Checklist

- `work_csv/segments_named.csv` ถูกสร้างและมีคอลัมน์ครบ
- หลัง filter แล้วยังมีคลาสที่ต้องการ
- `data_videos/<class>/` มีคลิปจริง
- `data/X.npy` เป็น 3 มิติ และ `X.shape[2] == 150`
- เทรนจบแล้วได้ `models/lstm_fall_model_v2.h5`
- `main.py` โหลดโมเดลใหม่และพยากรณ์ได้

## หมายเหตุ

- Auto-label คือ pseudo-label ควรตรวจแก้ก่อน retrain
- `timesteps` ต้องสอดคล้องกันทั้ง infer/dataset/train
- ถ้า class order ไม่ชัด ให้รัน `tools/verify_class_order.py` ก่อนทุกครั้ง
- มีไฟล์ `.evnv` สำหรับค่า default อ้างอิง แต่สคริปต์ยังไม่โหลดอัตโนมัติ
