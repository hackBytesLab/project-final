# Fall Detection - Project Final

โปรเจกต์นี้ใช้ MediaPipe (Pose + Hand) เพื่อดึง landmark จากวิดีโอ แล้วเทรนโมเดล LSTM สำหรับแยกคลาส:
- `Fall`
- `No_Fall`
- `Pre-Fall`
- `Falling`

## การติดตั้ง

```bash
pip install -r requirements.txt
```

แพ็กเกจหลัก:
- `tensorflow`
- `mediapipe`
- `opencv-python`
- `numpy`
- `scikit-learn`

## โครงสร้างไฟล์สำคัญ

- `main.py`: รันตรวจจับแบบ real-time จากกล้อง/สตรีม
- `video_to_dataset.py`: แปลงวิดีโอที่ติดป้ายแล้วเป็น `X.npy`, `y.npy`
- `train.py`: เทรนโมเดล LSTM
- `infer_video.py`: แยกคลาสจากวิดีโอยาวและส่งออกช่วงเวลาเป็น CSV
- `lstm_model.py`: นิยามโมเดล
- `Iriun_Webcam.py`: helper สำหรับ Iriun webcam

ไฟล์โมเดล MediaPipe ที่ต้องมีใน `models/`:
- `pose_landmarker_lite.task`
- `hand_landmarker.task`

## ไฟล์ตั้งค่า `.evnv`

ในโปรเจกต์มีไฟล์ `.evnv` สำหรับเก็บค่า default เช่น:
- `MODEL_PATH`
- `CAMERA_MODE`
- `CAMERA_SOURCE`
- `TIMESTEPS`
- `STEP`
- `BATCH_SIZE`
- `LABELS`

หมายเหตุ: ตอนนี้สคริปต์ยังไม่ได้อ่าน `.evnv` อัตโนมัติ ใช้เป็นค่าอ้างอิงเวลาใส่ argument

## 1) เตรียมวิดีโอที่ติดป้ายคลาส

จัดไฟล์แบบแยกโฟลเดอร์ต่อคลาส:

```text
data_videos/
  Fall/
    fall_001.mp4
  No_Fall/
    nofall_001.mp4
  Pre-Fall/
    prefall_001.mp4
  Falling/
    falling_001.mp4
```

## 2) แปลงวิดีโอเป็น Dataset

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15
```

ผลลัพธ์:
- `data/X.npy` รูปแบบ `(n_samples, timesteps, num_features)`
- `data/y.npy` รูปแบบ `(n_samples,)`

ค่าเริ่มต้นที่ใช้ในโปรเจกต์:
- `timesteps=30`
- `num_features=150`

## 3) เทรนโมเดล

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --out models/lstm_fall_model.h5
```

ทดสอบ pipeline แบบเร็ว:

```bash
python train.py --generate-sample --data-dir data
```

## 4) Auto Label วิดีโอยาวด้วยโมเดลที่เทรนแล้ว

สามารถใช้ `infer_video.py` แยกช่วงคลาสจากวิดีโอยาวได้ (pseudo-label):

```bash
python infer_video.py ^
  --video path\to\long_video.mp4 ^
  --model models/lstm_fall_model.h5 ^
  --out-csv segments.csv ^
  --timesteps 30 --step 1 --batch-size 64 ^
  --labels Fall,No_Fall,Pre-Fall,Falling ^
  --out-video labeled.mp4
```

ผลลัพธ์:
- `segments.csv`: `class_id,class_name,start_time_s,end_time_s,avg_score`
- `labeled.mp4`: วิดีโอที่วาด label บนภาพ (ถ้าใส่ `--out-video`)

คำแนะนำ:
- ใช้ผล auto-label เป็น “ร่างแรก” แล้วตรวจแก้ด้วยคนก่อนนำไปเทรนรอบถัดไป
- ลำดับชื่อใน `--labels` ต้องตรงกับ class id ของโมเดล

## 5) รันแบบ Real-time (`main.py`)

`main.py` รองรับหลายโหมดกล้องผ่าน `--camera`

### Iriun

```bash
python main.py --camera iriun --model models/lstm_fall_model.h5
```

### Raspberry Pi camera (รันบน Pi)

```bash
python main.py --camera pi --model models/lstm_fall_model.h5
```

### RTSP stream

```bash
python main.py --camera rtsp --source rtsp://<pi-ip>:8554/stream --model models/lstm_fall_model.h5
```

### กล้องตาม index

```bash
python main.py --camera index --source 0 --model models/lstm_fall_model.h5
```

## หมายเหตุสำคัญ

- ห้ามเทรนจากวิดีโอยาวที่ยังไม่ติดป้ายโดยตรง
- ให้ใช้ `timesteps` เดียวกันระหว่างแปลงข้อมูล, เทรน, และ inference
- ก่อนรัน inference ต้องมี `models/lstm_fall_model.h5`
