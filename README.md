# เว็ปเเอพจำเเนกอารมณ์จากรูปภาพ
จุดประสงค์ของโปรเจกต์
โปรเจกต์นี้มีจุดมุ่งหมายเพื่อพัฒนา ระบบจำแนกอารมณ์จากใบหน้า (Facial Emotion Recognition) ด้วยโมเดล Deep Learning เพื่อใช้ในการตรวจจับและวิเคราะห์อารมณ์ของผู้ใช้งานแบบอัตโนมัติผ่านกล้องหรือภาพถ่าย
ระบบนี้สามารถ ต่อยอด ไปยังแอปพลิเคชันได้หลายด้าน เช่น

แอปสุขภาพจิต : ใช้วิเคราะห์อารมณ์ของผู้ใช้งานในแต่ละวัน เพื่อช่วยประเมินภาวะความเครียด ความเศร้า หรือความสุขในระยะยาว เชื่อมต่อกับระบบช่วยเหลือหรือคำแนะนำด้านสุขภาพจิตตามอารมณ์ที่ตรวจพบ กระตุ้นให้ผู้ใช้ดูแลสุขภาพใจด้วยวิธีที่เหมาะสม

การวิเคราะห์ความพึงพอใจของลูกค้า : ตรวจจับอารมณ์ของลูกค้าระหว่างรับบริการ เพื่อประเมินคุณภาพการให้บริการ ใช้ในการปรับปรุงประสบการณ์ลูกค้า  อย่างแม่นยำ ลดการใช้แบบสอบถามแบบเดิม เพิ่มความสะดวกและความถูกต้อง

เเหล่งข้อมูล(Data set)
https://www.kaggle.com/datasets/msambare/fer2013?resource=download

# การ labelling ข้อมูล 
โดยการเเยกข้อมูลที่ดาวน์โหลดมา โดยเเยกเป็น 7 โฟล์เดอร์ 7 คลาส โดยมี 

![image](https://github.com/user-attachments/assets/f5e296bd-e395-48c9-95da-35dcecb0de24)

จากนั้นนำมาเทรน ด้วย Google Colab
โดยดาวน์โหลดไฟล์ Zip ด้วยโค้ด

from google.colab import files
uploaded = files.upload()

เเละเเตกไฟล์ด้วยด้วยโค้ด
 
!unzip archive.zip -d archive

# การเทรนโมเดลเเละตั้งค่าพารามิเตอร์ต่างๆ

ฝึก โมเดล CNN เพื่อจำแนกอารมณ์จากใบหน้า (เช่น angry,disgust,fear,happy,neutral,sad,surprise ) ด้วย ภาพขนาด 48x48 grayscale โดยใช้ PyTorch


โหลดไลบรารีที่ใช้สำหรับการฝึกโมเดลภาพ และแสดงผลภาพ
```
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

ตรวจสอบว่าใช้ GPU ได้หรือไม่ (ถ้าไม่ได้จะใช้ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

สำหรับ ข้อมูลฝึก (train) (augmentation) เพื่อให้โมเดลเรียนรู้หลากหลายขึ้น

```
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
DataLoader ใช้แบ่งข้อมูลเป็น batch ละ 64 รูป
ใช้ LR = 0.0005


ลูปการฝึกทั้งหมด 30 ครั้ง

![image](https://github.com/user-attachments/assets/d638e8fc-ee18-4732-816c-8a0e228e7c4c)


# ประเมินความแม่นยำบนชุดทดสอบ
Test Accuracy: 54.96%














































