# Fall Detection - Project Final

โปรเจกต์นี้ใช้ MediaPipe Pose + Hand landmarks แปลงวิดีโอเป็น sequence features แล้วเทรนโมเดล LSTM เพื่อแยกคลาส:
- `Fall`
- `No_Fall`
- `Pre-Fall`
- `Falling`

## 1) ติดตั้ง

```bash
pip install -r requirements.txt
```

Dependencies หลักใน `requirements.txt`:
- `tensorflow`
- `mediapipe`
- `opencv-python`
- `numpy`
- `scikit-learn`

## 2) โครงสร้างไฟล์สำคัญ

- `video_to_dataset.py`: แปลงวิดีโอที่แยกคลาสแล้วเป็น `X.npy` / `y.npy`
- `train.py`: เทรนโมเดล LSTM
- `infer_video.py`: ใช้โมเดลที่เทรนแล้วแยกคลาสจากวิดีโอยาว และ export segments
- `main.py`: รัน real-time detection จากกล้อง
- `lstm_model.py`: นิยามโมเดลและฟังก์ชันพยากรณ์
- `models/pose_landmarker_lite.task`, `models/hand_landmarker.task`: MediaPipe task files

## 3) เตรียมข้อมูลวิดีโอสำหรับเทรน

ต้องจัดวางวิดีโอแยกโฟลเดอร์ตามคลาสก่อน เช่น:

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

จากนั้นแปลงเป็น dataset:

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15
```

ผลลัพธ์:
- `data/X.npy` shape `(n_samples, timesteps, num_features)`
- `data/y.npy` shape `(n_samples,)`

ค่าเริ่มต้น:
- `timesteps=30`
- `num_features=150` (pose 33 จุด x,y + hand 2 มือ x 21 จุด x,y)

## 4) เทรนโมเดล

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --out models/lstm_fall_model.h5
```

โมเดลที่ดีที่สุดจะถูกบันทึกไปที่ `--out`

### สร้างข้อมูลตัวอย่าง (ทดสอบ pipeline)

```bash
python train.py --generate-sample --data-dir data
```

## 5) ใช้โมเดลแยกคลาสจากวิดีโอยาว

```bash
python infer_video.py \
  --video path/to/long_video.mp4 \
  --model models/lstm_fall_model.h5 \
  --out-csv segments.csv \
  --timesteps 30 --step 1 --batch-size 64 \
  --labels Fall,No_Fall,Pre-Fall,Falling \
  --out-video labeled.mp4
```

ผลลัพธ์:
- `segments.csv` มีคอลัมน์ `class_id,class_name,start_time_s,end_time_s,avg_score`
- `labeled.mp4` (ถ้าระบุ `--out-video`)

## 6) รันแบบ real-time จากกล้อง (`main.py`)

`main.py` โหลดโมเดลที่เทรนแล้วผ่าน `--model` และรองรับกล้องหลายแบบผ่าน `--camera`

### Iriun Webcam

```bash
python main.py --camera iriun --model models/lstm_fall_model.h5
```

### Raspberry Pi camera (รันบน Pi)

```bash
python main.py --camera pi --model models/lstm_fall_model.h5
```

### RTSP stream (เช่นสตรีมจาก Pi)

```bash
python main.py --camera rtsp --source rtsp://<pi-ip>:8554/stream --model models/lstm_fall_model.h5
```

### กล้องจาก camera index

```bash
python main.py --camera index --source 0 --model models/lstm_fall_model.h5
```

## 7) หมายเหตุสำคัญ

- ถ้าวิดีโอยังไม่ติดป้ายคลาส ห้ามเทรนตรงๆ ให้แยกคลาสก่อน
- ให้ใช้ `timesteps` เดียวกันในขั้นตอน train และ inference
- ชื่อคลาสใน `--labels` ควรเรียงให้ตรงกับ class id ของโมเดล
- ต้องมีไฟล์ต่อไปนี้ในโฟลเดอร์ `models/`:
  - `pose_landmarker_lite.task`
  - `hand_landmarker.task`
  - `lstm_fall_model.h5` (หลังเทรน)
