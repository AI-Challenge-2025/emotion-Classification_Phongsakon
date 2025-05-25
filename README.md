# เว็ปเเอพจำเเนกอารมณ์จากรูปภาพ
จุดประสงค์ของโปรเจกต์
โปรเจกต์นี้มีจุดมุ่งหมายเพื่อพัฒนา ระบบจำแนกอารมณ์จากใบหน้า (Facial Emotion Recognition) ด้วยโมเดล Deep Learning เพื่อใช้ในการตรวจจับและวิเคราะห์อารมณ์ของผู้ใช้งานแบบอัตโนมัติผ่านกล้องหรือภาพถ่าย
ระบบนี้สามารถ ต่อยอด ไปยังแอปพลิเคชันได้หลายด้าน เช่น

แอปสุขภาพจิต : ใช้วิเคราะห์อารมณ์ของผู้ใช้งานในแต่ละวัน เพื่อช่วยประเมินภาวะความเครียด ความเศร้า หรือความสุขในระยะยาว เชื่อมต่อกับระบบช่วยเหลือหรือคำแนะนำด้านสุขภาพจิตตามอารมณ์ที่ตรวจพบ กระตุ้นให้ผู้ใช้ดูแลสุขภาพใจด้วยวิธีที่เหมาะสม

การวิเคราะห์ความพึงพอใจของลูกค้า : ตรวจจับอารมณ์ของลูกค้าระหว่างรับบริการ เพื่อประเมินคุณภาพการให้บริการ ใช้ในการปรับปรุงประสบการณ์ลูกค้า  อย่างแม่นยำ ลดการใช้แบบสอบถามแบบเดิม เพิ่มความสะดวกและความถูกต้อง

# เเหล่งข้อมูล(Data set) 
FER-2013 Dataset 

https://www.kaggle.com/datasets/msambare/fer2013?resource=download

# โมเดลที่ใช้
โมเดลที่ใช้ชื่อว่า BetterCNN ซึ่งเป็น Convolutional Neural Network (CNN) ที่ถูกออกแบบมาให้ทำงานกับภาพขาวดำขนาด 48x48 พิกเซล เพื่อจำแนกอารมณ์ต่างๆ

![1_kkyW7BR5FZJq4_oBTx3OPQ](https://github.com/user-attachments/assets/2a8c8871-43a5-47c0-89f1-c9acf41f1f8d)


อารมณ์ที่สามารถตรวจจับได้

Angry (โกรธ)

Disgust (รังเกียจ)

Fear (กลัว)

Happy (มีความสุข)

Sad (เศร้า)

Surprise (ตกใจ)

Neutral (เฉย ๆ)
# การ labelling ข้อมูล 
โดยการเเยกข้อมูลที่ดาวน์โหลดมา โดยเเยกเป็น 7 โฟล์เดอร์ 7 คลาส โดยมี 

![image](https://github.com/user-attachments/assets/f5e296bd-e395-48c9-95da-35dcecb0de24)

จากนั้นนำมาเทรน ด้วย Google Colab
โดยดาวน์โหลดไฟล์ Zip ด้วยโค้ด

```
from google.colab import files
uploaded = files.upload()
```
เเละเเตกไฟล์ด้วยด้วยโค้ด
 ```
!unzip archive.zip -d archive
```
# การเทรนโมเดลเเละตั้งค่าพารามิเตอร์ต่างๆ

ฝึก โมเดล CNN เพื่อจำแนกอารมณ์จากใบหน้า (เช่น angry,disgust,fear,happy,neutral,sad,surprise ) ด้วย ภาพขนาด 48x48 grayscale โดยใช้ PyTorch


โหลดไลบรารีที่ใช้สำหรับการฝึกโมเดลภาพ และแสดงผลภาพ
```
import os                                # สำหรับจัดการไฟล์และโฟลเดอร์
import torch                             # PyTorch สำหรับ deep learning
import torch.nn as nn                    # สำหรับสร้างเลเยอร์ของโมเดล
import torch.nn.functional as F          # สำหรับฟังก์ชัน activation และอื่น ๆ
from torchvision import datasets, transforms  # สำหรับโหลดชุดข้อมูลภาพ และแปลงภาพ
from torch.utils.data import DataLoader  # สำหรับสร้างตัวโหลดข้อมูลแบบ batch
import matplotlib.pyplot as plt          # สำหรับแสดงกราฟและภาพ

```

ตรวจสอบว่าใช้ GPU ได้หรือไม่ (ถ้าไม่ได้จะใช้ CPU)
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

สำหรับ ข้อมูลฝึก train โดยเพิ่มเทคนิค augmentation(การเพิ่มข้อมูลเทียม) เพื่อให้โมเดลเรียนรู้หลากหลายขึ้น

```
# แปลงภาพสำหรับชุดฝึก (training set) โดยเพิ่มเทคนิค augmentation
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),             # ปรับขนาดภาพให้เป็น 48x48 พิกเซล
    transforms.RandomHorizontalFlip(),       # สุ่มกลับภาพในแนวนอน เพื่อเพิ่มความหลากหลาย
    transforms.RandomRotation(10),           # สุ่มหมุนภาพไม่เกิน ±10 องศา
    transforms.Grayscale(),                  # แปลงภาพเป็นขาวดำ (1 ช่องสัญญาณ)
    transforms.ToTensor(),                   # แปลงภาพให้เป็น tensor (ข้อมูลที่ PyTorch ใช้งาน)
    transforms.Normalize((0.5,), (0.5,))     # ปรับค่าความสว่างให้อยู่ในช่วง [-1, 1]
])

])
```
DataLoader ใช้แบ่งข้อมูลเป็น batch ละ 64 รูป
ใช้อัตราการเรียนรู้ ของโมเดลไว้ที่ LR = 0.0005


ลูปการฝึกทั้งหมด 30 ครั้ง

![image](https://github.com/user-attachments/assets/d638e8fc-ee18-4732-816c-8a0e228e7c4c)


# ประเมินความแม่นยำบนชุดทดสอบ
Test Accuracy: 62.58%


# การสร้างเว็ปเเอพโดยใช้ Streamlit
โค้ดสร้าง UI
```
st.set_page_config(page_title="Emotion Detection", layout="centered")  # ตั้งชื่อและ layout หน้าเว็บ
st.title("\U0001F603 ตรวจจับอารมณ์จากใบหน้า")  # หัวข้อหลัก
st.markdown("""
อัปโหลดรูปภาพที่มีใบหน้า และระบบจะทำการตรวจจับและวิเคราะห์อารมณ์ เช่น **มีความสุข**, **โกรธ**, **เศร้า**, **ตกใจ** เป็นต้น
""")  # คำอธิบาย


# UI สำหรับอัปโหลดภาพ
uploaded_file = st.file_uploader("\U0001F4F7 เลือกไฟล์รูปภาพของคุณ (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# ถ้ามีภาพที่อัปโหลดเข้ามา
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # เปิดภาพและแปลงเป็น RGB
    img_np = np.array(image)  # แปลงเป็น numpy array
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # แปลงเป็นภาพขาวดำสำหรับตรวจจับใบหน้า
```

# การตรวจจับใบหน้าด้วย (Face Detection)
ในโปรเจกต์นี้ใช้เทคนิค Haar Cascade Classifier จาก OpenCV เพื่อทำการตรวจจับใบหน้าในภาพที่ผู้ใช้ส่งเข้ามา Haar Cascade คือ อัลกอริทึมที่ใช้เทคนิค “การตรวจจับคุณลักษณะเฉพาะของวัตถุ ผ่านการเปรียบเทียบกับแบบรูปที่ฝึกไว้ล่วงหน้า 

![image](https://github.com/user-attachments/assets/8f94e6af-2b1d-4b7b-90d3-fdab59b9cab3)

โหลดโมเดล Haar Cascade ของ OpenCV สำหรับตรวจจับใบหน้า
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # ตรวจหาใบหน้า

    if len(faces) == 0:
        st.warning("\U000026A0\ufe0f ไม่พบใบหน้าในภาพ กรุณาอัปโหลดภาพที่มีใบหน้าเด่นชัด")  # ถ้าไม่เจอใบหน้า
    else:
        st.subheader("\U0001F4C8 ผลลัพธ์การตรวจจับอารมณ์")  # หัวข้อผลลัพธ์
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # ตัดเฉพาะส่วนใบหน้า
            face_pil = Image.fromarray(face).resize((48, 48))  # แปลงเป็น PIL และ resize
            face_tensor = transform(face_pil).unsqueeze(0).to(device)  # แปลงเป็น tensor และส่งเข้าโมเดล

            with torch.no_grad():  # ไม่คำนวณ gradient
                outputs = model(face_tensor)  # พยากรณ์อารมณ์
                _, predicted = torch.max(outputs, 1)  # หาคลาสที่มีค่าสูงสุด
                label = class_names[predicted.item()]  # แปลง index เป็น label

            # วาดกรอบสี่เหลี่ยมบนภาพใบหน้า + label
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # แสดงภาพที่มีกรอบและ label อารมณ์
        st.image(img_np, caption="ภาพพร้อมผลลัพธ์การวิเคราะห์อารมณ์", channels="RGB", use_container_width=True)

```


# การทำงานของเว็ปเเอพ ตรวจจับอารมณ์จากใบหน้า

1.ผู้ใช้กดปุ่มเพิ่มรูปภาพ

2.รอการประมวลผล

3.เเสดงผลลัพธ์

![image](https://github.com/user-attachments/assets/56426ad0-ad0c-4a6c-ace6-ac4e17286359)




# การทดสอบตรวจจับหลายคน
![image](https://github.com/user-attachments/assets/fb0e9fe3-260c-4dda-b898-6e6210bcbfb0)






# วิธีการทำงานของระบบ
1.ผู้ใช้ทำการอัปโหลดรูปภาพ

2.ระบบใช้ OpenCV ตรวจจับใบหน้าภายในภาพ (Haar Cascade)

3.เฉพาะส่วนใบหน้าจะถูกตัดออก → resize → grayscale → normalize

4.นำใบหน้าแต่ละใบเข้าสู่โมเดล CNN

5.โมเดลประเมินอารมณ์ และแสดงผลด้วยกรอบและชื่ออารมณ์บนภาพ

https://github.com/user-attachments/assets/f768fc7a-bb54-44e0-b24c-8b9b36d0d8b4

# วิธีการติดตั้งและใช้งาน

1. Clone โปรเจกต์
```
git clone https://github.com/yourusername/emotion-detection-app.git
cd emotion-detection-app
```

2. ติดตั้งไลบรารีที่จำเป็น

```
pip install -r requirements.txt
```

3. รันแอป
```
streamlit run app.py
```

#ไลบรารีที่ใช้
Streamlit — สำหรับสร้าง UI เว็บแอป

PyTorch — สำหรับสร้างและโหลดโมเดล CNN

OpenCV — สำหรับการตรวจจับใบหน้า (Haar Cascade)

Pillow — จัดการรูปภาพ

NumPy — จัดการ array ของภาพ


# อ้างอิง

```
https://medium.com/@natthawatphongchit/%E0%B8%A1%E0%B8%B2%E0%B8%A5%E0%B8%AD%E0%B8%87%E0%B8%94%E0%B8%B9%E0%B8%A7%E0%B8%B4%E0%B8%98%E0%B8%B5%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%84%E0%B8%B4%E0%B8%94%E0%B8%82%E0%B8%AD%E0%B8%87-cnn-%E0%B8%81%E0%B8%B1%E0%B8%99-e3f5d73eebaa

https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d

https://www.youtube.com/watch?v=BoGNyWW9-mE
```












