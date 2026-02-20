# Fall Detection — Project Final

เอกสารสั้น ๆ สำหรับการฝึกและใช้งานโมเดล LSTM ในโปรเจคนี้

**ภาพรวม:**
- โค้ดหลักตรวจจับ pose และ hand landmarks อยู่ใน `main.py`
- โมเดล LSTM ถูกนิยามใน `lstm_model.py`
- สคริปต์ฝึกถูกเพิ่มเป็น `train.py`

**Dependencies:**
ติดตั้งแพ็กเกจที่ต้องการด้วย:

```bash
pip install -r requirements.txt
```

ไฟล์ `requirements.txt` มีรายการหลัก: `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`.

**รูปแบบข้อมูลสำหรับการฝึก:**
- โฟลเดอร์ข้อมูล: `data/`
- ไฟล์ที่ต้องมี: `X.npy`, `y.npy`
  - `X.npy`: shape = `(n_samples, timesteps, num_features)` — ค่าเริ่มต้นโค้ดสมมติใช้ `timesteps=30` และ `num_features=150` (33 pose points ×2 + 21×2 hands ×2)
  - `y.npy`: shape = `(n_samples,)` — กำหนดเป็น integer class ids (0..C-1)

**สร้างข้อมูลตัวอย่าง (smoke test):**

```bash
python train.py --generate-sample --data-dir data
```

คำสั่งนี้จะสร้าง `data/X.npy` และ `data/y.npy` แบบสุ่มเพื่อทดสอบ pipeline

**รันการฝึก:**

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --out models/lstm_fall_model.h5
```

- โมเดลที่ดีที่สุดจะถูกบันทึกเป็นไฟล์ `models/lstm_fall_model.h5`
- ปรับ `--epochs` และ `--batch-size` ตามต้องการ

**การใช้งานโมเดลที่ฝึกแล้วใน `main.py`:**
- ปัจจุบัน `main.py` สร้างสถาปัตยกรรมโมเดลด้วย `build_lstm_model(...)` แต่ไม่ได้โหลดน้ำหนักจากไฟล์
- หลังฝึกเสร็จ ให้เปลี่ยนการสร้างโมเดลใน `main.py` เป็นการโหลดโมเดลที่บันทึกไว้ เช่น:

```python
from tensorflow.keras.models import load_model
model = load_model('models/lstm_fall_model.h5')
```

หรือถ้าต้องการโหลดน้ำหนักเท่านั้น (เมื่อยังต้องสร้างสถาปัตยกรรมด้วยโค้ด):

```python
model = build_lstm_model(num_features, num_classes)
model.load_weights('models/lstm_fall_model.h5')
```

**ข้อควรระวัง:**
- ตรวจสอบให้แน่ใจว่า `num_features` และ `timesteps` ในข้อมูลตรงกับที่โมเดลคาดไว้
- หากใช้ `load_model` ให้แน่ใจว่าไฟล์ `.h5` ถูกบันทึกด้วยสถาปัตยกรรมที่เข้ากัน

**ขั้นตอนถัดไปที่ผมช่วยได้:**
- รัน smoke test ฝึกสั้น ๆ ให้ (ผมสามารถรัน `python train.py --generate-sample` และฝึก 1-2 epochs)
- สร้างสคริปต์แปลง landmarks → `X.npy`/`y.npy` จากวิดีโอหรือ CSV ของคุณ

---

ไฟล์สำคัญ:
- `train.py` — สคริปต์ฝึก
- `lstm_model.py` — นิยามโมเดล
- `main.py` — แอพเรียลไทม์ (ต้องแก้ไขเพื่อโหลดโมเดลหลังฝึก)



การแปลงวิดีโอเป็นชุดข้อมูล (Video -> dataset)
---------------------------------------------
- วางวิดีโอโดยแยกโฟลเดอร์ตามคลาส: ตัวอย่างโครงสร้าง

```
data_videos/
  Fall/
    fall_001.mp4
    fall_002.mp4
  No_Fall/
    nofall_001.mp4
  Pre-Fall/
  Falling/
```

- สคริปต์ที่ใช้: `video_to_dataset.py`
- ตัวอย่างการรัน:

```bash
python video_to_dataset.py --input data_videos --output data --timesteps 30 --step 15
```

- ผลลัพธ์: จะได้ `data/X.npy` และ `data/y.npy` ซึ่งเป็น input สำหรับ `train.py` (ดูรายละเอียดรูปแบบข้อมูลด้านบน)

การรัน inference บนวิดีโอยาว (segment detection)
-------------------------------------------------
- สคริปต์ที่ใช้: `infer_video.py`
- ใช้โมเดลที่ฝึกแล้ว (.h5) ในการคาดคลาสของแต่ละ window แล้วรวมเป็น segments
- ตัวอย่างการรัน (รวมการสร้างวิดีโอที่มีป้ายชื่อ):

```bash
python infer_video.py \
  --video path/to/long_video.mp4 \
  --model models/lstm_fall_model.h5 \
  --out-csv segments.csv \
  --timesteps 30 --step 1 --batch-size 64 \
  --labels Fall,No_Fall,Pre-Fall,Falling \
  --out-video labeled.mp4
```

- ผลลัพธ์:
  - `segments.csv` — แถว: `class_id, class_name, start_time_s, end_time_s, avg_score`
  - ถ้าใส่ `--out-video` จะได้วิดีโอที่วาดป้ายคลาสบนเฟรม

วางไฟล์ที่จำเป็นสำหรับ Mediapipe
---------------------------------
- โปรเจคนี้ใช้ไฟล์ task ของ Mediapipe ที่อยู่ในโฟลเดอร์ `models/`:
  - `models/pose_landmarker_lite.task`
  - `models/hand_landmarker.task`
- ต้องมีไฟล์สองไฟล์นี้ไว้ในโปรเจค เพื่อให้ `video_to_dataset.py`, `main.py`, และ `infer_video.py` ทำงาน

ข้อแนะนำสั้น ๆ
-----------------
- ถ้าวิดีโอสั้นกว่า `--timesteps` จะถูกข้ามในการแปลง
- ปรับ `--timesteps` และ `--step` เพื่อควบคุมความยาว window และ overlap
- หากต้องการความเรียบ (smoothing) หรือ threshold ก่อนสร้าง segments ให้ขอผมเพิ่มฟังก์ชัน filter ได้
