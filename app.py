
import streamlit as st  # สำหรับสร้างเว็บแอปแบบ interactive
import torch  # ใช้สำหรับโหลดและรันโมเดล deep learning ที่ฝึกไว้
import torch.nn as nn  # สำหรับสร้างโครงสร้างโมเดล
import torchvision.transforms as transforms  # สำหรับแปลงรูปภาพให้เหมาะกับการป้อนเข้าโมเดล
import cv2  # OpenCV สำหรับการประมวลผลภาพ เช่น ตรวจจับใบหน้า
from PIL import Image  # สำหรับจัดการไฟล์ภาพ
import numpy as np  # ใช้สำหรับจัดการข้อมูลในรูปแบบอาเรย์


# สร้างคลาสโมเดล CNN ที่เราเคยฝึกไว้แล้ว
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        # convolution layer 1: รับภาพ grayscale (1 channel) → 32 ฟิลเตอร์ ขนาด 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # convolution layer 2: 32 → 64 ฟิลเตอร์
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # convolution layer 3: 64 → 128 ฟิลเตอร์
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling เพื่อลดขนาดภาพ
        self.pool = nn.MaxPool2d(2, 2)
        # dropout เพื่อป้องกัน overfitting
        self.dropout = nn.Dropout(0.4)
        # fully connected layer 1 → เอาผลจาก conv มาแปลงเป็นเวกเตอร์เข้าไปยัง FC
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        # fully connected layer สุดท้าย → ออกผลลัพธ์ 7 class (7 อารมณ์)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        # โยนข้อมูลผ่านแต่ละเลเยอร์พร้อม ReLU และ pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # reshape ข้อมูลให้กลายเป็นเวกเตอร์ 1 มิติ
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# โหลดโมเดลที่ฝึกแล้ว
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ใช้ GPU ถ้ามี
model = BetterCNN().to(device)  # สร้างโมเดล
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))  # โหลดน้ำหนัก
model.eval()  # ตั้งค่าโมเดลให้ทำงานในโหมดประเมินผล (ไม่ใช้ dropout/batch norm)


# กำหนด label ของอารมณ์ 7 ประเภท
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# กำหนดการแปลงภาพให้เป็น input สำหรับโมเดล
transform = transforms.Compose([
    transforms.Resize((48, 48)),         # ย่อขนาดให้เป็น 48x48 พิกเซล
    transforms.ToTensor(),               # แปลงภาพเป็น tensor
    transforms.Normalize((0.5,), (0.5,)) # ปรับค่า pixel ให้อยู่ในช่วง [-1, 1]
])


# เริ่มสร้างหน้า Streamlit UI
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

    # โหลดโมเดล Haar Cascade ของ OpenCV สำหรับตรวจจับใบหน้า
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
