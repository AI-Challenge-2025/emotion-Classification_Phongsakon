import os  # ใช้สำหรับจัดการไฟล์และโฟลเดอร์
import torch  # ไลบรารีหลักสำหรับ deep learning
import torch.nn as nn  # สำหรับสร้างโครงสร้างของ neural network
import torch.nn.functional as F  # สำหรับใช้ฟังก์ชันภายในเช่น ReLU
from torchvision import datasets, transforms  # โหลดข้อมูลภาพ + ทำ data augmentation
from torch.utils.data import DataLoader  # ใช้สำหรับโหลดข้อมูลเข้าโมเดลเป็น batch
import matplotlib.pyplot as plt  # สำหรับการ plot กราฟต่างๆ (เช่น loss)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# สำหรับข้อมูล train: มีการทำ augmentation เพื่อช่วยให้โมเดลเจอข้อมูลหลากหลาย
transform_train = transforms.Compose([
    transforms.Resize((48, 48)),  # ปรับขนาดภาพให้เป็น 48x48
    transforms.RandomHorizontalFlip(),  # สลับซ้ายขวาสุ่ม
    transforms.RandomRotation(10),  # หมุนภาพสุ่ม ±10 องศา
    transforms.Grayscale(),  # แปลงเป็นภาพขาวดำ (1 channel)
    transforms.ToTensor(),  # แปลงเป็น Tensor
    transforms.Normalize((0.5,), (0.5,))  # ปรับค่าพิกเซลให้อยู่ในช่วง [-1, 1]
])

# สำหรับข้อมูลทดสอบ: ไม่มีการสุ่ม เปรียบเทียบแบบนิ่งๆ
transform_test = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#โหลดข้อมูลภาพ
train_dataset = datasets.ImageFolder("archive/train", transform=transform_train)
test_dataset = datasets.ImageFolder("archive/test", transform=transform_test)

# จัด batch ข้อมูลและสลับข้อมูลในชุดฝึก
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# เก็บชื่อของ class เช่น ['Angry', 'Happy', 'Sad', ...]
class_names = train_dataset.classes


# สร้างโครงสร้างโมเดล CNN
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()

        # Convolution Layer 1: input = 1 channel (grayscale), output = 32 filters
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # Convolution Layer 2: input = 32, output = 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Convolution Layer 3: input = 64, output = 128
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max pooling layer (ลดขนาดภาพครึ่งหนึ่ง)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout เพื่อป้องกัน overfitting
        self.dropout = nn.Dropout(0.4)

        # Fully connected layers (flatten 128x6x6 → 256 → 7 classes)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, len(class_names))

    def forward(self, x):
        # Block 1: Conv + ReLU + Pool → ขนาด: 48x48 → 24x24
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: → 12x12
        x = self.pool(F.relu(self.conv2(x)))
        # Block 3: → 6x6
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten tensor จาก 4D เป็น 2D เพื่อป้อนเข้า fully connected
        x = x.view(-1, 128 * 6 * 6)

        # Dropout + Fully Connected Layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# เตรียมโมเดลสำหรับฝึก
model = BetterCNN().to(device)  # ส่งโมเดลไปยัง GPU หรือ CPU
criterion = nn.CrossEntropyLoss()  # Loss function สำหรับ multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # ใช้ Adam optimizer กับ learning rate 0.0005


# เริ่มการฝึกโมเดล
for epoch in range(30):  # เทรนทั้งหมด 30 รอบ
    model.train()  # ตั้งโมเดลให้อยู่ในโหมดฝึก
    running_loss = 0.0  # ใช้เก็บค่า loss สะสมแต่ละ epoch

    for inputs, labels in train_loader:
        # ส่งข้อมูลและ label ไปยังอุปกรณ์ (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # เคลียร์ gradient ก่อน backward
        outputs = model(inputs)  # ส่งข้อมูลเข้าโมเดล
        loss = criterion(outputs, labels)  # คำนวณ loss
        loss.backward()  # คำนวณ gradient
        optimizer.step()  # ปรับน้ำหนักของโมเดล

        running_loss += loss.item()  # บวกค่า loss ของแต่ละ batch เข้ารวม

    # แสดงค่า loss เฉลี่ยต่อ epoch
    print(f"Epoch {epoch+1}/30, Loss: {running_loss/len(train_loader):.4f}")
